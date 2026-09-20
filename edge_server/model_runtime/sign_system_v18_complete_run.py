"""静态与动态手语统一序列识别工程。

数据目录：
    静态动作\\0..3：他、你、好、我，每个视频按单帧特征复制成 30 帧序列。
    动态动作\\4..8：吃饭、喝水、帮助、帮我、想，每个视频按连续帧生成序列。

完整流程：视频扫描 -> MediaPipe 特征 -> debug 图片 -> 序列数据 ->
Leave-One-Video-Out 验证 -> GRU 训练 -> 保存模型 -> 摄像头识别。
"""

import os
import random
import sys
import time
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PROJECT_ROOT / "静态动作"
DYNAMIC_ROOT = PROJECT_ROOT / "动态动作"
DEBUG_DIR = PROJECT_ROOT / "dataset_debug"

SEQUENCES_FILE = PROJECT_ROOT / "sign_sequences.npy"
SEQUENCE_LABELS_FILE = PROJECT_ROOT / "sign_sequence_labels.npy"
SEQUENCE_GROUPS_FILE = PROJECT_ROOT / "sign_sequence_groups.npy"
MODEL_FILE = PROJECT_ROOT / "sign_gru_final.pth"
DYNAMIC_SEQUENCES_FILE = PROJECT_ROOT / "sign_dynamic_sequences.npy"
DYNAMIC_LABELS_FILE = PROJECT_ROOT / "sign_dynamic_labels.npy"
DYNAMIC_GROUPS_FILE = PROJECT_ROOT / "sign_dynamic_groups.npy"
DYNAMIC_MODEL_FILE = PROJECT_ROOT / "sign_dynamic_gru_final.pth"
STATIC_FEATURES_FILE = PROJECT_ROOT / "sign_static_features.npy"
STATIC_LABELS_FILE = PROJECT_ROOT / "sign_static_labels.npy"
STATIC_GROUPS_FILE = PROJECT_ROOT / "sign_static_groups.npy"
STATIC_MODEL_FILE = PROJECT_ROOT / "sign_static_mlp_final.pth"
IMAGE_FEATURES_FILE = PROJECT_ROOT / "sign_image_frames.npy"
IMAGE_LABELS_FILE = PROJECT_ROOT / "sign_image_labels.npy"
IMAGE_GROUPS_FILE = PROJECT_ROOT / "sign_image_groups.npy"
IMAGE_MODEL_FILE = PROJECT_ROOT / "sign_image_cnn_final.pth"
IMAGE_ONNX_FILE = PROJECT_ROOT / "sign_image_cnn_final.onnx"
RICH_FEATURES_FILE = PROJECT_ROOT / "sign_rich_features.npy"
RICH_LABELS_FILE = PROJECT_ROOT / "sign_rich_labels.npy"
RICH_GROUPS_FILE = PROJECT_ROOT / "sign_rich_groups.npy"
RICH_MODEL_FILE = PROJECT_ROOT / "sign_rich_mlp_final.pth"
RICH_PAIR_CLASSES = (12, 13)
RICH_PAIR_MODEL_FILE = PROJECT_ROOT / "sign_rich_you_have_pair_mlp.pth"
RICH_CONFUSION_CLASSES = (6, 8, 12, 13)
RICH_CONFUSION_MODEL_FILE = PROJECT_ROOT / "sign_rich_confusion4_mlp.pth"

CLASS_NAMES = {
    0: "他",
    1: "你",
    2: "好",
    3: "我",
    4: "吃饭",
    5: "喝水",
    6: "帮助",
    7: "帮我",
    8: "想",
    9: "不",
    10: "不是",
    11: "是",
    12: "要",
    13: "有",
}

STATIC_CLASS_IDS = tuple(range(4))
DYNAMIC_CLASS_IDS = tuple(range(4, 9))
DYNAMIC_CLASS_NAMES = {
    0: "吃饭",
    1: "喝水",
    2: "帮助",
    3: "帮我",
    4: "想",
}
STATIC_NINE_VIDEO_LABELS = {
    "他.mp4": 0,
    "你.mp4": 1,
    "好.mp4": 2,
    "我.mp4": 3,
    "吃饭.mp4": 4,
    "喝水.mp4": 5,
    "帮助.mp4": 6,
    "帮我.mp4": 7,
    "想.mp4": 8,
    "不.mp4": 9,
    "不是.mp4": 10,
    "是.mp4": 11,
    "要.mp4": 12,
    "有.mp4": 13,
}
STATIC_NINE_CLASS_NAMES = CLASS_NAMES
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SEQ_LEN = 30
HEAD_FEATURE_DIM = 22
HAND_FEATURE_DIM = 86
FEATURE_DIM = 202 + HEAD_FEATURE_DIM
RICH_FEATURE_DIM = 380
DYNAMIC_WINDOW_SECONDS = 2.0
DYNAMIC_STRIDE_SECONDS = 0.5
CAMERA_BUFFER_SECONDS = 2.5
STATIC_FRAME_STRIDE = 3
IMAGE_FRAME_STRIDE = 3
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANNELS = 3
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = int(os.environ.get("SIGN_EPOCHS", "100"))
VALIDATION_EPOCHS = int(
    os.environ.get("SIGN_VALIDATION_EPOCHS", str(min(EPOCHS, 40)))
)
LEARNING_RATE = 0.003

FONT_PATHS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


@lru_cache(maxsize=None)
def get_cn_font(size):
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("找不到中文字体，请安装微软雅黑、黑体或宋体")


def put_cn_text(frame, text, org, font_size=36, color=(255, 255, 255)):
    """使用系统中文字体在 OpenCV BGR 图像上绘制文字。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    rgb_color = (color[2], color[1], color[0])
    draw.text(
        org,
        str(text),
        font=get_cn_font(font_size),
        fill=rgb_color,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def put_cn_text_right(frame, text, y=10, font_size=32, color=(255, 255, 255)):
    """在画面右上角绘制中文状态。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = get_cn_font(font_size)
    text_box = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    x = max(10, frame.shape[1] - text_width - 20)
    rgb_color = (color[2], color[1], color[0])
    draw.text(
        (x, y),
        str(text),
        font=font,
        fill=rgb_color,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros_like(vector)


def angle(a, b, c):
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return float(np.arccos(np.clip(cosine, -1, 1)))


def hand_feature(hand):
    features = list(hand.flatten())

    # 食指方向、拇指方向、手掌法向量。
    features.extend(normalize(hand[8] - hand[5]))
    features.extend(normalize(hand[4] - hand[1]))
    features.extend(
        normalize(
            np.cross(
                hand[5] - hand[0],
                hand[17] - hand[0],
            )
        )
    )

    joints = [
        (1, 2, 3),
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20),
    ]
    for a, b, c in joints:
        features.append(angle(hand[a], hand[b], hand[c]))

    for a, b in [(4, 8), (8, 12), (8, 16), (8, 20)]:
        features.append(float(np.linalg.norm(hand[a] - hand[b])))

    return features


def _hand_scale(hand):
    wrist = hand[0]
    palm_points = hand[[5, 9, 13, 17]]
    scale = float(np.mean(np.linalg.norm(palm_points - wrist, axis=1)))
    return max(scale, 1e-6)


def rich_hand_feature(hand):
    """提取不依赖画面位置的五指朝向、弯曲程度和手型特征。"""
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape != (21, 3) or not np.any(np.abs(hand)):
        return [0.0] * 130

    wrist = hand[0]
    scale = _hand_scale(hand)
    relative_landmarks = ((hand - wrist) / scale).flatten()
    index_direction = normalize(hand[8] - hand[5])
    middle_direction = normalize(hand[12] - hand[9])
    thumb_direction = normalize(hand[4] - hand[1])
    ring_direction = normalize(hand[16] - hand[13])
    little_direction = normalize(hand[20] - hand[17])
    palm_normal = normalize(np.cross(hand[5] - wrist, hand[17] - wrist))
    directions = np.concatenate(
        [
            thumb_direction,
            index_direction,
            middle_direction,
            ring_direction,
            little_direction,
            palm_normal,
        ]
    )
    base_bend_angles = []
    for mcp, pip in [(1, 2), (5, 6), (9, 10), (13, 14), (17, 18)]:
        base_bend_angles.append(angle(wrist, hand[mcp], hand[pip]))
    angles = []
    for a, b, c in [
        (1, 2, 3),
        (2, 3, 4),
        (5, 6, 7),
        (6, 7, 8),
        (9, 10, 11),
        (10, 11, 12),
        (13, 14, 15),
        (14, 15, 16),
        (17, 18, 19),
        (18, 19, 20),
    ]:
        angles.append(angle(hand[a], hand[b], hand[c]))
    distances = [
        float(np.linalg.norm(hand[4] - hand[8]) / scale),
        float(np.linalg.norm(hand[8] - hand[12]) / scale),
        float(np.linalg.norm(hand[8] - hand[16]) / scale),
        float(np.linalg.norm(hand[8] - hand[20]) / scale),
    ]
    finger_tips = [hand[4], hand[8], hand[12]]
    finger_relations = []
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        vector = (finger_tips[second] - finger_tips[first]) / scale
        finger_relations.extend(vector)
        finger_relations.append(float(np.linalg.norm(vector)))
    finger_directions = [thumb_direction, index_direction, middle_direction]
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        finger_relations.append(
            float(
                np.arccos(
                    np.clip(
                        np.dot(finger_directions[first], finger_directions[second]),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
    finger_mcps = [hand[1], hand[5], hand[9], hand[13], hand[17]]
    finger_tips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
    finger_extension_ratios = [
        float(
            np.linalg.norm(tip - wrist)
            / max(np.linalg.norm(mcp - wrist), 1e-6)
        )
        for mcp, tip in zip(finger_mcps, finger_tips)
    ]
    palm_center = np.mean(hand[[0, 5, 9, 13, 17]], axis=0)
    fingertip_wrist_distances = [
        float(np.linalg.norm(tip - wrist) / scale) for tip in finger_tips
    ]
    fingertip_palm_distances = [
        float(np.linalg.norm(tip - palm_center) / scale) for tip in finger_tips
    ]
    extra_finger_geometry = np.asarray(
        finger_extension_ratios
        + fingertip_wrist_distances
        + fingertip_palm_distances
    )
    return np.concatenate(
        [
            relative_landmarks,
            directions,
            np.asarray(base_bend_angles),
            np.asarray(angles),
            np.asarray(distances),
            np.asarray(finger_relations),
            extra_finger_geometry,
        ]
    ).tolist()


def rich_palm_orientation_feature(hand, body, pose_points):
    """提取掌心朝向相对肩轴、身体轴和脸部轴的方向分量。"""
    hand = np.asarray(hand, dtype=np.float32)
    body = np.asarray(body, dtype=np.float32)
    pose_points = np.asarray(pose_points, dtype=np.float32)
    if (
        hand.shape != (21, 3)
        or not np.any(np.abs(hand))
        or body.shape != (6, 3)
        or not np.any(np.abs(body))
        or pose_points.shape != (33, 3)
        or not np.any(np.abs(pose_points))
    ):
        return [0.0] * 3

    wrist = hand[0]
    palm_normal = normalize(np.cross(hand[5] - wrist, hand[17] - wrist))
    shoulder_center = (body[0] + body[1]) / 2.0
    shoulder_axis = normalize(body[1] - body[0])
    body_axis = normalize(pose_points[0] - shoulder_center)
    mouth = (pose_points[9] + pose_points[10]) / 2.0
    face_axis = normalize(pose_points[0] - mouth)
    return [
        float(np.dot(palm_normal, shoulder_axis)),
        float(np.dot(palm_normal, body_axis)),
        float(np.dot(palm_normal, face_axis)),
    ]


def rich_arm_feature(body):
    """提取肩、肘、腕的相对位置、方向、长度和关节角度。"""
    body = np.asarray(body, dtype=np.float32)
    if body.shape != (6, 3) or not np.any(np.abs(body)):
        return [0.0] * 60

    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist = body
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = max(float(np.linalg.norm(right_shoulder - left_shoulder)), 1e-6)
    body_relative = ((body - shoulder_center) / shoulder_width).flatten()

    segment_vectors = []
    segment_lengths = []
    joint_angles = []
    for shoulder, elbow_point, wrist in [
        (left_shoulder, left_elbow, left_wrist),
        (right_shoulder, right_elbow, right_wrist),
    ]:
        shoulder_to_elbow = (elbow_point - shoulder) / shoulder_width
        elbow_to_wrist = (wrist - elbow_point) / shoulder_width
        segment_vectors.extend([shoulder_to_elbow, elbow_to_wrist])
        segment_lengths.extend(
            [
                float(np.linalg.norm(elbow_point - shoulder) / shoulder_width),
                float(np.linalg.norm(wrist - elbow_point) / shoulder_width),
            ]
        )
        joint_angles.append(angle(shoulder, elbow_point, wrist))

    wrist_body_relations = []
    for wrist in (left_wrist, right_wrist):
        for anchor in body[:4]:
            wrist_body_relations.extend((wrist - anchor) / shoulder_width)

    return np.concatenate(
        [
            body_relative,
            np.asarray(segment_vectors).flatten(),
            np.asarray(segment_lengths),
            np.asarray(joint_angles),
            np.asarray(wrist_body_relations),
        ]
    ).tolist()


def rich_face_feature(pose_points, left_hand, right_hand):
    """提取手腕/指尖与嘴、鼻子的相对位置，连接手臂、手和脸。"""
    pose_points = np.asarray(pose_points, dtype=np.float32)
    if pose_points.shape != (33, 3) or not np.any(np.abs(pose_points)):
        return [0.0] * 54

    shoulder_width = float(np.linalg.norm(pose_points[12] - pose_points[11]))
    if shoulder_width <= 1e-6:
        return [0.0] * 54
    nose = pose_points[0]
    mouth = (pose_points[9] + pose_points[10]) / 2.0
    face_shape = [
        _normalized_distance(pose_points[9], pose_points[10], shoulder_width),
        _normalized_distance(nose, mouth, shoulder_width),
        _normalized_distance(pose_points[3], pose_points[7], shoulder_width),
        _normalized_distance(pose_points[6], pose_points[8], shoulder_width),
        _normalized_distance(nose, pose_points[3], shoulder_width),
        _normalized_distance(nose, pose_points[6], shoulder_width),
    ]
    hand_face = []
    for hand in (left_hand, right_hand):
        hand = np.asarray(hand, dtype=np.float32)
        if hand.shape != (21, 3) or not np.any(np.abs(hand)):
            hand_face.extend([0.0] * 24)
            continue
        for index in (0, 8, 12):
            point = hand[index]
            hand_face.extend((point - mouth) / shoulder_width)
            hand_face.extend((point - nose) / shoulder_width)
            hand_face.append(_normalized_distance(point, mouth, shoulder_width))
            hand_face.append(_normalized_distance(point, nose, shoulder_width))
    return np.concatenate([np.asarray(face_shape), np.asarray(hand_face)]).tolist()


def build_rich_feature(left_hand, right_hand, body, pose_points):
    """组合 380 维相对手型、手臂和手脸关系特征。"""
    feature = np.asarray(
        rich_hand_feature(left_hand)
        + rich_hand_feature(right_hand)
        + rich_palm_orientation_feature(left_hand, body, pose_points)
        + rich_palm_orientation_feature(right_hand, body, pose_points)
        + rich_arm_feature(body)
        + rich_face_feature(pose_points, left_hand, right_hand),
        dtype=np.float32,
    )
    if feature.shape != (RICH_FEATURE_DIM,):
        raise RuntimeError(
            f"丰富相对特征维度错误：得到 {feature.shape}，预期 {(RICH_FEATURE_DIM,)}"
        )
    return feature


def body_feature(body):
    features = list(body.flatten())
    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist = body

    features.append(angle(left_shoulder, left_elbow, left_wrist))
    features.append(angle(right_shoulder, right_elbow, right_wrist))

    shoulder_center = (left_shoulder + right_shoulder) / 2
    features.extend(left_elbow - shoulder_center)
    features.extend(right_elbow - shoulder_center)
    features.append(float(np.linalg.norm(left_elbow - shoulder_center)))
    features.append(float(np.linalg.norm(right_elbow - shoulder_center)))

    shoulder_axis = normalize(right_shoulder - left_shoulder)
    features.append(float(np.dot(normalize(left_elbow - left_shoulder), -shoulder_axis)))
    features.append(float(np.dot(normalize(right_elbow - right_shoulder), shoulder_axis)))

    return features


def _normalized_distance(first, second, scale):
    return float(np.linalg.norm(first - second) / max(scale, 1e-6))


def _hand_head_distances(hand, mouth_center, head_anchors, scale):
    if not np.any(np.abs(hand)):
        return [0.0] * 8

    wrist = hand[0]
    index_tip = hand[8]
    middle_tip = hand[12]
    nearest_head_distance = lambda point: min(
        _normalized_distance(point, anchor, scale)
        for anchor in head_anchors
    )
    return [
        _normalized_distance(wrist, mouth_center, scale),
        _normalized_distance(index_tip, mouth_center, scale),
        _normalized_distance(middle_tip, mouth_center, scale),
        nearest_head_distance(wrist),
        nearest_head_distance(index_tip),
        nearest_head_distance(middle_tip),
        _normalized_distance(index_tip, middle_tip, scale),
        _normalized_distance(hand[5], hand[9], scale),
    ]


def head_feature(pose_points, left_hand, right_hand):
    """提取嘴部、太阳穴和手到头部的相对几何特征。"""
    pose_points = np.asarray(pose_points, dtype=np.float32)
    left_hand = np.asarray(left_hand, dtype=np.float32)
    right_hand = np.asarray(right_hand, dtype=np.float32)
    if pose_points.shape != (33, 3):
        raise ValueError(f"Pose 点形状错误：{pose_points.shape}")
    if left_hand.shape != (21, 3) or right_hand.shape != (21, 3):
        raise ValueError(
            f"手部点形状错误：left={left_hand.shape}, right={right_hand.shape}"
        )
    if not np.any(np.abs(pose_points)):
        return [0.0] * HEAD_FEATURE_DIM

    shoulder_width = np.linalg.norm(pose_points[12] - pose_points[11])
    if shoulder_width <= 1e-6:
        return [0.0] * HEAD_FEATURE_DIM

    nose = pose_points[0]
    mouth_center = (pose_points[9] + pose_points[10]) / 2.0
    head_anchors = pose_points[[3, 6, 7, 8]]
    features = [
        _normalized_distance(pose_points[9], pose_points[10], shoulder_width),
        _normalized_distance(nose, mouth_center, shoulder_width),
        _normalized_distance(pose_points[3], pose_points[7], shoulder_width),
        _normalized_distance(pose_points[6], pose_points[8], shoulder_width),
        _normalized_distance(nose, pose_points[3], shoulder_width),
        _normalized_distance(nose, pose_points[6], shoulder_width),
    ]
    features.extend(
        _hand_head_distances(left_hand, mouth_center, head_anchors, shoulder_width)
    )
    features.extend(
        _hand_head_distances(right_hand, mouth_center, head_anchors, shoulder_width)
    )
    return features


def to_pixel(point, width, height):
    if hasattr(point, "x") and hasattr(point, "y"):
        normalized_x = point.x
        normalized_y = point.y
    else:
        normalized_x = point[0]
        normalized_y = point[1]
    x = int(np.clip(normalized_x * width, 0, width - 1))
    y = int(np.clip(normalized_y * height, 0, height - 1))
    return x, y


def draw_pose_skeleton(frame, pose_landmarks):
    height, width, _ = frame.shape
    ids = [0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    points = {
        index: to_pixel(pose_landmarks.landmark[index], width, height)
        for index in ids
    }

    connections = [
        (3, 7),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    for start, end in connections:
        color = (255, 0, 255) if start in [3, 6, 9] else (255, 0, 0)
        cv2.line(frame, points[start], points[end], color, 2)

    for index, point in points.items():
        color = (255, 0, 255) if index in [0, 3, 6, 7, 8, 9, 10] else (255, 0, 0)
        cv2.circle(frame, point, 4, color, -1)


def get_hand_point_color(point_index):
    """返回 BGR 颜色：拇指绿、食指橙、中指黄、其他手指红。"""
    if point_index in [1, 2, 3, 4]:
        return (0, 255, 0)
    if point_index in [5, 6, 7, 8]:
        return (0, 165, 255)
    if point_index in [9, 10, 11, 12]:
        return (0, 255, 255)
    return (0, 0, 255)


def extract_feature(frame, draw=False, label=""):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    height, width, _ = frame.shape

    if hand_result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness,
        ):
            points = np.array(
                [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                dtype=np.float32,
            )

            if handedness.classification[0].label == "Left":
                left = points
            else:
                right = points

            if draw:
                for point_index, point in enumerate(points):
                    cv2.circle(
                        frame,
                        to_pixel(point, width, height),
                        3,
                        get_hand_point_color(point_index),
                        -1,
                    )

    pose_points = np.zeros((33, 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        pose_points = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
                for landmark in pose_result.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        if draw:
            draw_pose_skeleton(frame, pose_result.pose_landmarks)

    body = pose_points[[11, 12, 13, 14, 15, 16]]

    feature = np.array(
        hand_feature(left)
        + hand_feature(right)
        + body_feature(body)
        + head_feature(pose_points, left, right),
        dtype=np.float32,
    )
    if feature.shape != (FEATURE_DIM,):
        raise RuntimeError(f"特征维度错误：得到 {feature.shape}，预期 {(FEATURE_DIM,)}")

    if draw and label:
        put_cn_text(frame, label, (20, 10), 36, (255, 255, 255))

    return feature


def extract_rich_feature(frame, draw=False, label=""):
    """从 MediaPipe 关键点提取 358 维相对几何特征。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    height, width, _ = frame.shape
    if hand_result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness,
        ):
            points = np.array(
                [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            if handedness.classification[0].label == "Left":
                left = points
            else:
                right = points
            if draw:
                for point_index, point in enumerate(points):
                    cv2.circle(
                        frame,
                        to_pixel(point, width, height),
                        3,
                        get_hand_point_color(point_index),
                        -1,
                    )

    pose_points = np.zeros((33, 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        pose_points = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in pose_result.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        if draw:
            draw_pose_skeleton(frame, pose_result.pose_landmarks)

    body = pose_points[[11, 12, 13, 14, 15, 16]]
    feature = build_rich_feature(left, right, body, pose_points)
    if draw and label:
        put_cn_text(frame, label, (20, 10), 36, (255, 255, 255))
    return feature


def hand_is_present(feature):
    """判断特征中是否包含至少一只被 MediaPipe 检测到的手。"""
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (FEATURE_DIM,):
        raise ValueError(f"特征形状错误：{feature.shape}")
    hand_features = feature[: 2 * HAND_FEATURE_DIM]
    return bool(np.any(np.abs(hand_features) > 1e-6))


def list_videos(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"数据文件夹不存在：{directory}")
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def static_nine_class_id(filename):
    """根据九个新静态视频的文件名返回类别编号。"""
    basename = Path(filename).name
    if basename not in STATIC_NINE_VIDEO_LABELS:
        raise ValueError(
            f"不是九类静态动作视频：{basename}，"
            f"应为 {sorted(STATIC_NINE_VIDEO_LABELS)}"
        )
    return STATIC_NINE_VIDEO_LABELS[basename]


def list_static_nine_videos():
    """只扫描静态动作根目录的九个视频，不读取旧的 0～3 子文件夹。"""
    if not STATIC_ROOT.is_dir():
        raise FileNotFoundError(f"静态动作文件夹不存在：{STATIC_ROOT}")

    videos = []
    for video_path in sorted(STATIC_ROOT.iterdir(), key=lambda path: path.name):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if video_path.name in STATIC_NINE_VIDEO_LABELS:
            videos.append((video_path, static_nine_class_id(video_path.name)))

    expected = set(STATIC_NINE_VIDEO_LABELS)
    actual = {path.name for path, _ in videos}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"九类静态动作缺少视频：{missing}")
    return sorted(videos, key=lambda item: item[1])


def prepare_video_frame(frame, is_static):
    """保留手机视频的原始竖直方向，与摄像头特征提取保持一致。"""
    return frame


def discover_videos():
    entries = []
    for class_id in STATIC_CLASS_IDS:
        class_dir = STATIC_ROOT / str(class_id)
        videos = list_videos(class_dir)
        if not videos:
            raise RuntimeError(f"静态类别 {class_id} 没有视频：{class_dir}")
        entries.extend((video, class_id, True) for video in videos)

    for class_id in DYNAMIC_CLASS_IDS:
        class_dir = DYNAMIC_ROOT / str(class_id)
        videos = list_videos(class_dir)
        if not videos:
            raise RuntimeError(f"动态类别 {class_id} 没有视频：{class_dir}")
        entries.extend((video, class_id, False) for video in videos)

    return entries


def debug_path(kind, class_id, video_path, frame_index):
    output_dir = DEBUG_DIR / kind / str(class_id) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"frame_{frame_index:06d}.jpg"


def rich_debug_path(class_id, video_path, frame_index):
    output_dir = DEBUG_DIR / "rich_static" / str(class_id) / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"frame_{frame_index:06d}.jpg"


def save_debug_image(path, frame):
    """用 Python 文件接口保存图片，兼容 Windows 中文目录。"""
    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise RuntimeError(f"debug 图片编码失败：{path}")
    buffer.tofile(str(path))


def read_video_features(video_path, class_id, is_static):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    kind = "static" if is_static else "dynamic"
    fps = capture.get(cv2.CAP_PROP_FPS)
    features = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = prepare_video_frame(frame, is_static)
            feature = extract_feature(frame, draw=True, label=CLASS_NAMES[class_id])
            save_debug_image(
                debug_path(kind, class_id, video_path, frame_index),
                frame,
            )
            features.append(feature)
            frame_index += 1
    finally:
        capture.release()

    print(f"读取 {video_path.name}: {len(features)} 帧，FPS={fps:.2f}")
    return features, fps


def static_to_sequence(feature):
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (FEATURE_DIM,):
        raise ValueError(f"静态特征形状错误：{feature.shape}")
    return np.repeat(feature[None, :], SEQ_LEN, axis=0).astype(np.float32)


def sample_uniform_sequence(features, timestamps, start_time):
    """在固定 2 秒时间窗内均匀抽取 30 帧，训练和摄像头共用。"""
    features = np.asarray(features, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"特征数组形状错误：{features.shape}")
    if timestamps.ndim != 1 or len(timestamps) != len(features):
        raise ValueError("特征和时间戳数量不一致")
    if len(features) == 0:
        return None

    end_time = float(start_time) + DYNAMIC_WINDOW_SECONDS
    if len(timestamps) > 1:
        frame_interval = float(np.median(np.diff(timestamps)))
    else:
        frame_interval = 0.0
    covered_until = float(timestamps[-1]) + max(frame_interval, 0.0)
    if covered_until < end_time - 1e-6:
        return None

    target_times = np.linspace(
        float(start_time),
        end_time,
        SEQ_LEN,
        endpoint=False,
        dtype=np.float32,
    )
    positions = np.searchsorted(timestamps, target_times, side="left")
    positions = np.clip(positions, 0, len(features) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = np.abs(timestamps[previous] - target_times) <= np.abs(
        timestamps[positions] - target_times
    )
    positions = np.where(use_previous, previous, positions)
    return features[positions].astype(np.float32)


def dynamic_sequences_from_features(features, fps):
    """按视频时间戳生成固定 2 秒、30 帧的动态序列。"""
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2 or feature_array.shape[1] != FEATURE_DIM:
        raise ValueError(f"动态特征形状错误：{feature_array.shape}")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 1.0 or fps > 120.0:
        fps = 30.0

    timestamps = np.arange(len(feature_array), dtype=np.float32) / fps
    total_duration = len(feature_array) / fps
    max_start = total_duration - DYNAMIC_WINDOW_SECONDS
    if max_start < 0:
        return []

    sequences = []
    starts = np.arange(
        0.0,
        max_start + 1e-6,
        DYNAMIC_STRIDE_SECONDS,
        dtype=np.float32,
    )
    for start_time in starts:
        sequence = sample_uniform_sequence(
            feature_array,
            timestamps,
            start_time=float(start_time),
        )
        if sequence is not None:
            sequences.append(sequence)
    return sequences


def select_dynamic_sequences(sequences, labels, groups):
    """从完整序列中筛选动态类，并重映射为动态模型的 0～4 类。"""
    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    if not (len(sequences) == len(labels) == len(groups)):
        raise ValueError("动态序列、标签和视频分组数量不一致")

    mask = np.isin(labels, DYNAMIC_CLASS_IDS)
    if not np.any(mask):
        raise ValueError("没有找到动态类别序列")

    source_to_dynamic = {
        source_id: dynamic_id
        for dynamic_id, source_id in enumerate(DYNAMIC_CLASS_IDS)
    }
    dynamic_labels = np.asarray(
        [source_to_dynamic[int(label)] for label in labels[mask]],
        dtype=np.int64,
    )
    return sequences[mask], dynamic_labels, groups[mask]


def build_dynamic_sequence_dataset():
    sequences, labels, groups = build_sequence_dataset()
    dynamic_sequences, dynamic_labels, dynamic_groups = select_dynamic_sequences(
        sequences, labels, groups
    )
    np.save(DYNAMIC_SEQUENCES_FILE, dynamic_sequences)
    np.save(DYNAMIC_LABELS_FILE, dynamic_labels)
    np.save(DYNAMIC_GROUPS_FILE, dynamic_groups)
    print(f"动态序列数据保存：{dynamic_sequences.shape}")
    print(
        "动态类别数量："
        f"{dict(sorted(Counter(dynamic_labels.tolist()).items()))}"
    )
    return dynamic_sequences, dynamic_labels, dynamic_groups


def build_static_nine_dataset():
    """读取全部固定动作视频，生成逐帧静态特征训练集。"""
    features = []
    labels = []
    groups = []

    for group_id, (video_path, class_id) in enumerate(list_static_nine_videos()):
        video_features, _ = read_video_features(
            video_path,
            class_id,
            is_static=True,
        )
        if not video_features:
            raise RuntimeError(f"静态视频没有读取到帧：{video_path}")

        for feature in video_features[::STATIC_FRAME_STRIDE]:
            features.append(feature)
            labels.append(class_id)
            groups.append(group_id)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"静态特征数据形状错误：{X.shape}")
    if not (len(X) == len(y) == len(group_array)):
        raise RuntimeError("静态特征、标签和视频分组数量不一致")
    if set(np.unique(y).tolist()) != set(STATIC_NINE_CLASS_NAMES):
        raise RuntimeError(f"静态类别不完整：{sorted(np.unique(y).tolist())}")

    np.save(STATIC_FEATURES_FILE, X)
    np.save(STATIC_LABELS_FILE, y)
    np.save(STATIC_GROUPS_FILE, group_array)
    print(f"静态特征保存：{X.shape}")
    print(f"静态类别数量：{dict(sorted(Counter(y.tolist()).items()))}")
    return X, y, group_array


def read_video_images(video_path):
    """从视频按固定间隔抽取 RGB 图像并统一到 640x480。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frames = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % IMAGE_FRAME_STRIDE == 0:
                frames.append(preprocess_image_frame_uint8(frame))
            frame_index += 1
    finally:
        capture.release()
    return frames


def build_image_dataset():
    """读取根目录 14 个动作视频，生成原始图像分类训练集。"""
    frames = []
    labels = []
    groups = []

    for group_id, (video_path, class_id) in enumerate(list_static_nine_videos()):
        video_frames = read_video_images(video_path)
        if not video_frames:
            raise RuntimeError(f"视频没有读取到帧：{video_path}")
        frames.extend(video_frames)
        labels.extend([class_id] * len(video_frames))
        groups.extend([group_id] * len(video_frames))
        print(f"抽帧 {video_path.name}: {len(video_frames)} 张")

    X = np.asarray(frames, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    if X.ndim != 4 or X.shape[1:] != (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(f"图像数据形状错误：{X.shape}")
    if not (len(X) == len(y) == len(group_array)):
        raise RuntimeError("图像、标签和视频分组数量不一致")
    if set(np.unique(y).tolist()) != set(STATIC_NINE_CLASS_NAMES):
        raise RuntimeError(f"图像类别不完整：{sorted(np.unique(y).tolist())}")

    np.save(IMAGE_FEATURES_FILE, X)
    np.save(IMAGE_LABELS_FILE, y)
    np.save(IMAGE_GROUPS_FILE, group_array)
    print(f"图像训练数据保存：{X.shape}")
    print(f"图像类别数量：{dict(sorted(Counter(y.tolist()).items()))}")
    return X, y, group_array


def build_rich_dataset():
    """读取同一批 14 个视频，提取相对手臂/手型/手脸关系特征。"""
    features = []
    labels = []
    groups = []
    for group_id, (video_path, class_id) in enumerate(list_static_nine_videos()):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")
        frame_index = 0
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % STATIC_FRAME_STRIDE == 0:
                    debug_frame = frame.copy()
                    features.append(
                        extract_rich_feature(
                            debug_frame,
                            draw=True,
                            label=CLASS_NAMES[class_id],
                        )
                    )
                    save_debug_image(
                        rich_debug_path(class_id, video_path, frame_index),
                        debug_frame,
                    )
                    labels.append(class_id)
                    groups.append(group_id)
                    count += 1
                frame_index += 1
        finally:
            capture.release()
        if count == 0:
            raise RuntimeError(f"视频没有读取到帧：{video_path}")
        print(f"提取相对特征 {video_path.name}: {count} 帧")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != RICH_FEATURE_DIM:
        raise RuntimeError(f"丰富特征数据形状错误：{X.shape}")
    if not (len(X) == len(y) == len(group_array)):
        raise RuntimeError("丰富特征、标签和视频分组数量不一致")
    if set(np.unique(y).tolist()) != set(STATIC_NINE_CLASS_NAMES):
        raise RuntimeError(f"丰富特征类别不完整：{sorted(np.unique(y).tolist())}")

    np.save(RICH_FEATURES_FILE, X)
    np.save(RICH_LABELS_FILE, y)
    np.save(RICH_GROUPS_FILE, group_array)
    print(f"丰富相对特征保存：{X.shape}")
    print(f"丰富特征类别数量：{dict(sorted(Counter(y.tolist()).items()))}")
    return X, y, group_array


def build_sequence_dataset():
    sequences = []
    labels = []
    groups = []
    video_entries = discover_videos()

    for group_id, (video_path, class_id, is_static) in enumerate(video_entries):
        features, fps = read_video_features(video_path, class_id, is_static)
        if is_static:
            for feature in features:
                sequences.append(static_to_sequence(feature))
                labels.append(class_id)
                groups.append(group_id)
            continue

        dynamic_sequences = dynamic_sequences_from_features(features, fps)
        if not dynamic_sequences:
            print(
                f"警告：动态视频 {video_path.name} 不足 "
                f"{DYNAMIC_WINDOW_SECONDS:.1f} 秒，跳过"
            )
            continue

        for sequence in dynamic_sequences:
            sequences.append(sequence)
            labels.append(class_id)
            groups.append(group_id)

    if not sequences:
        raise RuntimeError("没有生成任何序列，请检查视频目录和视频文件")

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)

    if X.ndim != 3 or X.shape[1:] != (SEQ_LEN, FEATURE_DIM):
        raise RuntimeError(f"序列数据形状错误：{X.shape}")
    if not (len(X) == len(y) == len(group_array)):
        raise RuntimeError("序列、标签、视频分组数量不一致")
    if set(np.unique(y).tolist()) != set(CLASS_NAMES):
        raise RuntimeError(f"类别不完整，当前类别：{sorted(np.unique(y).tolist())}")

    np.save(SEQUENCES_FILE, X)
    np.save(SEQUENCE_LABELS_FILE, y)
    np.save(SEQUENCE_GROUPS_FILE, group_array)

    print(f"序列数据保存：{X.shape}")
    print(f"类别数量：{dict(sorted(Counter(y.tolist()).items()))}")
    return X, y, group_array


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=len(CLASS_NAMES),
    ):
        super().__init__()
        dropout = 0.2 if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        output, _ = self.gru(inputs)
        return self.classifier(output[:, -1, :])


class StaticMLP(nn.Module):
    """静态动作的逐帧 MLP。"""

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=256,
        num_classes=len(STATIC_NINE_CLASS_NAMES),
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)


def preprocess_image_frame(frame):
    """将 OpenCV BGR 帧转换为 3x480x640 的 RGB 浮点输入。"""
    return preprocess_image_frame_uint8(frame).astype(np.float32) / 255.0


def preprocess_image_frame_uint8(frame):
    """将 OpenCV BGR 帧转换为 3x480x640 的 RGB 字节输入。"""
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != IMAGE_CHANNELS:
        raise ValueError(f"图像帧形状错误：{frame.shape}")
    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb, (2, 0, 1))


class TinyImageCNN(nn.Module):
    """直接处理摄像头图像的 ESP32-S3 友好型静态动作分类器。"""

    def __init__(self, num_classes=len(STATIC_NINE_CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 8, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(30, 40)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def fit_sequence_scaler(sequences):
    scaler = StandardScaler()
    scaler.fit(sequences.reshape(-1, FEATURE_DIM))
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def transform_sequences(sequences, mean, scale):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, FEATURE_DIM)
    scale = np.asarray(scale, dtype=np.float32).reshape(1, 1, FEATURE_DIM)
    return ((sequences - mean) / scale).astype(np.float32)


def fit_frame_scaler(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"静态特征形状错误：{features.shape}")
    scaler = StandardScaler()
    scaler.fit(features)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def transform_frames(features, mean, scale):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"静态特征形状错误：{features.shape}")
    feature_dim = features.shape[1]
    mean = np.asarray(mean, dtype=np.float32).reshape(1, feature_dim)
    scale = np.asarray(scale, dtype=np.float32).reshape(1, feature_dim)
    return ((features - mean) / scale).astype(np.float32)


class HandMotionDetector:
    """用双手关键点的归一化位移判断静止/运动，过滤 MediaPipe 微抖动。"""

    POINT_IDS = (0, 5, 8, 9, 12, 13, 17)

    def __init__(
        self,
        start_threshold=0.018,
        stop_threshold=0.011,
        start_frames=3,
        stop_frames=7,
    ):
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.state = "静止"
        self.previous_points = None
        self.previous_present = None
        self.smoothed_score = 0.0
        self.high_count = 0
        self.low_count = 0

    def _motion_points(self, feature):
        feature = np.asarray(feature, dtype=np.float32)
        left = feature[:63].reshape(21, 3)
        right_start = HAND_FEATURE_DIM
        right = feature[right_start : right_start + 63].reshape(21, 3)
        body = feature[172:190].reshape(6, 3)

        shoulder_width = float(np.linalg.norm(body[1] - body[0]))
        if shoulder_width <= 1e-6:
            shoulder_width = 0.25

        points = np.zeros((2, len(self.POINT_IDS), 3), dtype=np.float32)
        present = np.zeros(2, dtype=bool)
        for hand_index, hand in enumerate((left, right)):
            if np.any(np.abs(hand)):
                points[hand_index] = hand[list(self.POINT_IDS)]
                present[hand_index] = True
        return points, present, shoulder_width

    def update(self, feature):
        points, present, shoulder_width = self._motion_points(feature)
        if not np.any(present):
            self.previous_points = None
            self.previous_present = None
            self.state = "静止"
            self.smoothed_score = 0.0
            self.high_count = 0
            self.low_count = 0
            return self.state, 0.0

        if self.previous_points is None:
            self.previous_points = points
            self.previous_present = present
            return self.state, 0.0

        common_hands = present & self.previous_present
        if np.any(common_hands):
            displacement = np.linalg.norm(
                points[common_hands] - self.previous_points[common_hands],
                axis=2,
            )
            # 手指动作往往比手掌移动小。保留手掌的整体运动，同时提高
            # 食指尖(点 8)和中指尖(点 12)的权重，避免小幅手指动作
            # 被其它不动关键点的中位数抵消。
            palm_displacement = np.median(
                displacement[:, [0, 1, 3, 5, 6]], axis=1
            )
            fingertip_displacement = np.max(displacement[:, [2, 4]], axis=1)
            hand_motion = 0.35 * palm_displacement + 0.65 * fingertip_displacement
            raw_score = float(np.max(hand_motion) / shoulder_width)
        else:
            raw_score = 0.0

        self.smoothed_score = 0.6 * self.smoothed_score + 0.4 * raw_score
        if self.state == "静止":
            self.low_count = 0
            if self.smoothed_score >= self.start_threshold:
                self.high_count += 1
                if self.high_count >= self.start_frames:
                    self.state = "运动"
                    self.high_count = 0
            else:
                self.high_count = 0
        else:
            self.high_count = 0
            if self.smoothed_score <= self.stop_threshold:
                self.low_count += 1
                if self.low_count >= self.stop_frames:
                    self.state = "静止"
                    self.low_count = 0
            else:
                self.low_count = 0

        self.previous_points = points
        self.previous_present = present
        return self.state, self.smoothed_score


class MotionEventWindow:
    """把一次运动开始到结束整理成一个只提交一次的识别窗口。"""

    def __init__(self, window_seconds=DYNAMIC_WINDOW_SECONDS, pre_event_seconds=0.2):
        self.window_seconds = float(window_seconds)
        self.pre_event_seconds = float(pre_event_seconds)
        self.history = deque()
        self.previous_state = "静止"
        self.active = False
        self.motion_start_time = None
        self.event_start_time = None
        self.event_features = []
        self.event_timestamps = []

    def _trim_history(self, timestamp):
        keep_after = float(timestamp) - max(self.pre_event_seconds + 0.5, 1.0)
        while self.history and self.history[0][0] < keep_after:
            self.history.popleft()

    def _reset_event(self):
        self.active = False
        self.motion_start_time = None
        self.event_start_time = None
        self.event_features = []
        self.event_timestamps = []

    def update(self, feature, timestamp, motion_state):
        feature = np.asarray(feature, dtype=np.float32)
        timestamp = float(timestamp)
        self.history.append((timestamp, feature.copy()))
        self._trim_history(timestamp)

        result = None
        starts_event = (
            not self.active
            and self.previous_state != "运动"
            and motion_state == "运动"
        )
        if starts_event:
            self.active = True
            self.motion_start_time = timestamp
            target_start = timestamp - self.pre_event_seconds
            selected = [item for item in self.history if item[0] >= target_start]
            if not selected:
                selected = [(timestamp, feature.copy())]
            self.event_start_time = selected[0][0]
            self.event_timestamps = [item[0] for item in selected]
            self.event_features = [item[1] for item in selected]
        elif self.active:
            self.event_timestamps.append(timestamp)
            self.event_features.append(feature.copy())

        if (
            self.active
            and motion_state == "静止"
            and timestamp - self.motion_start_time >= self.window_seconds
        ):
            result = sample_uniform_sequence(
                np.asarray(self.event_features, dtype=np.float32),
                np.asarray(self.event_timestamps, dtype=np.float32),
                start_time=self.event_start_time,
            )
            self._reset_event()

        self.previous_state = motion_state
        return result


def balanced_sample_weights(labels, num_classes=None):
    labels = np.asarray(labels, dtype=np.int64)
    if num_classes is None:
        num_classes = len(CLASS_NAMES)
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"训练数据缺少类别，无法均衡采样：{counts.tolist()}")
    return (1.0 / counts[labels]).astype(np.float64)


def train_model(sequences, labels, epochs=EPOCHS, num_classes=None):
    if num_classes is None:
        num_classes = len(CLASS_NAMES)
    model = SequenceGRU(num_classes=num_classes).to(device)
    dataset = TensorDataset(
        torch.from_numpy(sequences),
        torch.from_numpy(labels),
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            balanced_sample_weights(labels, num_classes=num_classes),
            dtype=torch.double,
        ),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        sampler=sampler,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(batch_y)
            correct += int((output.argmax(1) == batch_y).sum())
            total += len(batch_y)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}"
            )

    return model


def train_static_model(
    features,
    labels,
    epochs=EPOCHS,
    num_classes=None,
    input_dim=None,
    hidden_dim=256,
):
    if num_classes is None:
        num_classes = len(STATIC_NINE_CLASS_NAMES)
    if input_dim is None:
        input_dim = int(np.asarray(features).shape[1])
    model = StaticMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            balanced_sample_weights(labels, num_classes=num_classes),
            dtype=torch.double,
        ),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        sampler=sampler,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(batch_y)
            correct += int((output.argmax(1) == batch_y).sum())
            total += len(batch_y)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Static Epoch {epoch + 1}/{epochs} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}"
            )

    return model


def image_batch_to_device(batch):
    """把 uint8 图像批次转换成模型使用的 [0, 1] 浮点数。"""
    if batch.dtype == torch.uint8:
        batch = batch.to(device=device, dtype=torch.float32)
        return batch.div_(255.0)
    batch = batch.to(device=device, dtype=torch.float32)
    if batch.numel() and float(batch.detach().max()) > 1.0:
        batch = batch / 255.0
    return batch


def train_image_model(images, labels, epochs=EPOCHS, num_classes=None):
    """训练直接读取 640x480 RGB 图像的轻量 CNN。"""
    if num_classes is None:
        num_classes = len(STATIC_NINE_CLASS_NAMES)
    model = TinyImageCNN(num_classes=num_classes).to(device)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(images)),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)),
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            balanced_sample_weights(np.asarray(labels), num_classes=num_classes),
            dtype=torch.double,
        ),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(32, len(dataset)),
        sampler=sampler,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = image_batch_to_device(batch_x)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(batch_y)
            correct += int((output.argmax(1) == batch_y).sum())
            total += len(batch_y)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Image Epoch {epoch + 1}/{epochs} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.4f}"
            )

    return model


def predict(model, sequences):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sequences).to(device))
        return logits.argmax(1).cpu().numpy()


def predict_static(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        return logits.argmax(1).cpu().numpy()


def predict_image(model, images):
    """分批预测图像，避免一次性把整个数据集搬进显存。"""
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(images)))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(image_batch_to_device(batch_x))
            predictions.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def validate_leave_one_video_out(X, y, groups, class_names=CLASS_NAMES):
    unique_groups = sorted(np.unique(groups).tolist())
    scores = []
    video_entries = discover_videos()

    for held_out in unique_groups:
        train_mask = groups != held_out
        test_mask = groups == held_out
        train_labels = y[train_mask]
        if set(np.unique(train_labels).tolist()) != set(class_names):
            raise RuntimeError(f"LOOV 折 {held_out} 的训练集缺少类别")

        mean, scale = fit_sequence_scaler(X[train_mask])
        train_X = transform_sequences(X[train_mask], mean, scale)
        test_X = transform_sequences(X[test_mask], mean, scale)
        model = train_model(
            train_X,
            train_labels,
            epochs=VALIDATION_EPOCHS,
            num_classes=len(class_names),
        )
        predictions = predict(model, test_X)
        score = accuracy_score(y[test_mask], predictions)
        scores.append(score)

        video_name = video_entries[held_out][0].name
        print(f"LOOV 测试 {video_name}: {score:.4f}")

    mean_score = float(np.mean(scores))
    print(f"LOOV 平均准确率: {mean_score:.4f}")
    return scores


def train_final_model(X, y, model_file=MODEL_FILE, class_names=CLASS_NAMES):
    mean, scale = fit_sequence_scaler(X)
    normalized_X = transform_sequences(X, mean, scale)
    model = train_model(
        normalized_X,
        y,
        epochs=EPOCHS,
        num_classes=len(class_names),
    )
    checkpoint = {
        "model": model.state_dict(),
        "class_names": class_names,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "seq_len": SEQ_LEN,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dynamic_window_seconds": DYNAMIC_WINDOW_SECONDS,
    }
    torch.save(checkpoint, model_file)
    print(f"最终模型保存：{model_file}")


def validate_static_model(X, y, groups=None):
    """用每个视频后 20% 的帧做时间尾部验证。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    test_indices = []
    train_indices = []
    for class_id in sorted(np.unique(y).tolist()):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < 5:
            raise RuntimeError(f"类别 {class_id} 的静态帧太少，无法验证")
        split = max(1, int(len(class_indices) * 0.8))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    mean, scale = fit_frame_scaler(X[train_indices])
    train_X = transform_frames(X[train_indices], mean, scale)
    test_X = transform_frames(X[test_indices], mean, scale)
    model = train_static_model(
        train_X,
        y[train_indices],
        epochs=VALIDATION_EPOCHS,
        num_classes=len(STATIC_NINE_CLASS_NAMES),
    )
    predictions = predict_static(model, test_X)
    score = accuracy_score(y[test_indices], predictions)
    print(f"九类静态时间尾部验证准确率: {score:.4f}")
    print(
        "说明：每个类别目前只有一个新视频，因此这里按每个视频前 80% 训练、后 20% 验证，"
        "不是跨视频 LOOV。"
    )
    return score


def train_static_final_model(
    X,
    y,
    model_file=STATIC_MODEL_FILE,
    class_names=STATIC_NINE_CLASS_NAMES,
):
    mean, scale = fit_frame_scaler(X)
    normalized_X = transform_frames(X, mean, scale)
    model = train_static_model(
        normalized_X,
        y,
        epochs=EPOCHS,
        num_classes=len(class_names),
    )
    checkpoint = {
        "model_type": "static_mlp",
        "model": model.state_dict(),
        "class_names": class_names,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": 256,
        "num_classes": len(class_names),
    }
    torch.save(checkpoint, model_file)
    print(f"静态最终模型保存：{model_file}")


def validate_rich_model(X, y, groups=None):
    """验证 358 维相对特征模型，按每个视频的时间尾部切分。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    train_indices = []
    test_indices = []
    for class_id in sorted(np.unique(y).tolist()):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < 5:
            raise RuntimeError(f"类别 {class_id} 的相对特征太少，无法验证")
        split = max(1, int(len(class_indices) * 0.8))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    mean, scale = fit_frame_scaler(X[train_indices])
    train_X = transform_frames(X[train_indices], mean, scale)
    test_X = transform_frames(X[test_indices], mean, scale)
    model = train_static_model(
        train_X,
        y[train_indices],
        epochs=VALIDATION_EPOCHS,
        num_classes=len(STATIC_NINE_CLASS_NAMES),
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=384,
    )
    predictions = predict_static(model, test_X)
    score = accuracy_score(y[test_indices], predictions)
    print(f"相对特征模型时间尾部验证准确率: {score:.4f}")
    return score


def train_rich_final_model(
    X,
    y,
    model_file=RICH_MODEL_FILE,
    class_names=STATIC_NINE_CLASS_NAMES,
):
    X = np.asarray(X, dtype=np.float32)
    mean, scale = fit_frame_scaler(X)
    normalized_X = transform_frames(X, mean, scale)
    model = train_static_model(
        normalized_X,
        y,
        epochs=EPOCHS,
        num_classes=len(class_names),
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=384,
    )
    checkpoint = {
        "model_type": "rich_relative_mlp",
        "model": model.state_dict(),
        "class_names": class_names,
        "scaler_mean": mean,
        "scaler_scale": scale,
        "feature_dim": RICH_FEATURE_DIM,
        "hidden_dim": 384,
        "num_classes": len(class_names),
        "frame_stride": STATIC_FRAME_STRIDE,
    }
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_file)
    print(f"相对特征最终模型保存：{model_file}")
    return model


def validate_rich_pair_model(X, y, groups=None):
    """验证“要/有”专用二分类器，使用同一批视频的时间尾部。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mask = np.isin(y, RICH_PAIR_CLASSES)
    pair_X = X[mask]
    pair_y_source = y[mask]
    pair_y = np.asarray(
        [0 if label == RICH_PAIR_CLASSES[0] else 1 for label in pair_y_source],
        dtype=np.int64,
    )
    train_indices = []
    test_indices = []
    for local_class in (0, 1):
        class_indices = np.flatnonzero(pair_y == local_class)
        split = max(1, int(len(class_indices) * 0.8))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])
    mean, scale = fit_frame_scaler(pair_X[train_indices])
    train_X = transform_frames(pair_X[train_indices], mean, scale)
    test_X = transform_frames(pair_X[test_indices], mean, scale)
    model = train_static_model(
        train_X,
        pair_y[train_indices],
        epochs=VALIDATION_EPOCHS,
        num_classes=2,
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=256,
    )
    predictions = predict_static(model, test_X)
    score = accuracy_score(pair_y[test_indices], predictions)
    print(f"要/有二分类时间尾部验证准确率: {score:.4f}")
    return score


def train_rich_pair_final_model(
    X,
    y,
    model_file=RICH_PAIR_MODEL_FILE,
):
    """训练只区分“要”和“有”的专用模型。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mask = np.isin(y, RICH_PAIR_CLASSES)
    pair_X = X[mask]
    pair_y = np.asarray(
        [0 if label == RICH_PAIR_CLASSES[0] else 1 for label in y[mask]],
        dtype=np.int64,
    )
    mean, scale = fit_frame_scaler(pair_X)
    normalized_X = transform_frames(pair_X, mean, scale)
    model = train_static_model(
        normalized_X,
        pair_y,
        epochs=EPOCHS,
        num_classes=2,
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=256,
    )
    checkpoint = {
        "model_type": "rich_pair_mlp",
        "model": model.state_dict(),
        "class_ids": RICH_PAIR_CLASSES,
        "class_names": {
            0: STATIC_NINE_CLASS_NAMES[RICH_PAIR_CLASSES[0]],
            1: STATIC_NINE_CLASS_NAMES[RICH_PAIR_CLASSES[1]],
        },
        "scaler_mean": mean,
        "scaler_scale": scale,
        "feature_dim": RICH_FEATURE_DIM,
        "hidden_dim": 256,
        "num_classes": 2,
    }
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_file)
    print(f"要/有二分类模型保存：{model_file}")
    return model


def validate_rich_confusion_model(X, y, groups=None):
    """验证帮助/想/要/有四类混淆专用模型。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mask = np.isin(y, RICH_CONFUSION_CLASSES)
    confusion_X = X[mask]
    confusion_y_source = y[mask]
    confusion_y = np.asarray(
        [RICH_CONFUSION_CLASSES.index(int(label)) for label in confusion_y_source],
        dtype=np.int64,
    )
    train_indices = []
    test_indices = []
    for local_class in range(len(RICH_CONFUSION_CLASSES)):
        class_indices = np.flatnonzero(confusion_y == local_class)
        split = max(1, int(len(class_indices) * 0.8))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])
    mean, scale = fit_frame_scaler(confusion_X[train_indices])
    train_X = transform_frames(confusion_X[train_indices], mean, scale)
    test_X = transform_frames(confusion_X[test_indices], mean, scale)
    model = train_static_model(
        train_X,
        confusion_y[train_indices],
        epochs=VALIDATION_EPOCHS,
        num_classes=len(RICH_CONFUSION_CLASSES),
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=256,
    )
    predictions = predict_static(model, test_X)
    score = accuracy_score(confusion_y[test_indices], predictions)
    print(f"帮助/想/要/有四分类时间尾部验证准确率: {score:.4f}")
    return score


def train_rich_confusion_final_model(
    X,
    y,
    model_file=RICH_CONFUSION_MODEL_FILE,
):
    """训练帮助/想/要/有四类混淆专用模型。"""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    mask = np.isin(y, RICH_CONFUSION_CLASSES)
    confusion_X = X[mask]
    confusion_y = np.asarray(
        [RICH_CONFUSION_CLASSES.index(int(label)) for label in y[mask]],
        dtype=np.int64,
    )
    mean, scale = fit_frame_scaler(confusion_X)
    normalized_X = transform_frames(confusion_X, mean, scale)
    model = train_static_model(
        normalized_X,
        confusion_y,
        epochs=EPOCHS,
        num_classes=len(RICH_CONFUSION_CLASSES),
        input_dim=RICH_FEATURE_DIM,
        hidden_dim=256,
    )
    checkpoint = {
        "model_type": "rich_confusion4_mlp",
        "model": model.state_dict(),
        "class_ids": RICH_CONFUSION_CLASSES,
        "class_names": {
            local_id: STATIC_NINE_CLASS_NAMES[global_id]
            for local_id, global_id in enumerate(RICH_CONFUSION_CLASSES)
        },
        "scaler_mean": mean,
        "scaler_scale": scale,
        "feature_dim": RICH_FEATURE_DIM,
        "hidden_dim": 256,
        "num_classes": len(RICH_CONFUSION_CLASSES),
    }
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_file)
    print(f"四类混淆模型保存：{model_file}")
    return model


def validate_image_model(X, y, groups=None):
    """按每个视频的时间尾部 20% 验证原始图像模型。"""
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    test_indices = []
    train_indices = []
    for class_id in sorted(np.unique(y).tolist()):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < 5:
            raise RuntimeError(f"类别 {class_id} 的图像帧太少，无法验证")
        split = max(1, int(len(class_indices) * 0.8))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    model = train_image_model(
        X[train_indices],
        y[train_indices],
        epochs=VALIDATION_EPOCHS,
        num_classes=len(STATIC_NINE_CLASS_NAMES),
    )
    predictions = predict_image(model, X[test_indices])
    score = accuracy_score(y[test_indices], predictions)
    print(f"图像模型时间尾部验证准确率: {score:.4f}")
    print(
        "说明：每个类别目前只有一个视频，因此按每个视频前 80% 训练、后 20% 验证，"
        "不是跨视频验证。"
    )
    return score


def export_image_model_onnx(model, output_path=IMAGE_ONNX_FILE):
    """导出固定 1x3x480x640 输入的 ONNX，供 ESP-DL/ESP-PPQ 转换。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(model.classifier.out_features)
    export_model = TinyImageCNN(num_classes=num_classes).cpu()
    export_model.load_state_dict(model.state_dict())
    export_model.eval()
    dummy_input = torch.zeros(
        (1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32
    )
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    try:
        import onnx

        onnx.checker.check_model(str(output_path))
    except ImportError:
        print("警告：未安装 onnx，已跳过 ONNX 完整性检查")
    print(f"ONNX 模型保存：{output_path}")
    return output_path


def train_image_final_model(
    X,
    y,
    model_file=IMAGE_MODEL_FILE,
    onnx_file=IMAGE_ONNX_FILE,
    class_names=STATIC_NINE_CLASS_NAMES,
):
    model = train_image_model(
        X,
        y,
        epochs=EPOCHS,
        num_classes=len(class_names),
    )
    checkpoint = {
        "model_type": "tiny_image_cnn",
        "model": model.state_dict(),
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_width": IMAGE_WIDTH,
        "input_height": IMAGE_HEIGHT,
        "input_channels": IMAGE_CHANNELS,
        "frame_stride": IMAGE_FRAME_STRIDE,
    }
    model_file = Path(model_file)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_file)
    print(f"图像最终模型保存：{model_file}")
    export_image_model_onnx(model, onnx_file)
    return model


def camera_test(show_motion=False, model_file=MODEL_FILE):
    if not model_file.exists():
        raise FileNotFoundError(f"找不到模型：{model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    model = SequenceGRU(
        input_dim=checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(class_names),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头")

    motion_detector = HandMotionDetector()
    event_window = MotionEventWindow(
        window_seconds=DYNAMIC_WINDOW_SECONDS,
        pre_event_seconds=0.2,
    )
    last_prediction_id = None
    last_confidence = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("摄像头读取失败")
                break

            feature = extract_feature(frame, draw=True)
            now = time.monotonic()
            motion_state, motion_score = motion_detector.update(feature)
            event_sequence = event_window.update(feature, now, motion_state)
            if event_sequence is not None:
                normalized = transform_sequences(
                    event_sequence[None, ...], mean, scale
                )
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(torch.from_numpy(normalized).to(device)),
                        dim=1,
                    )[0]
                last_prediction_id = int(probabilities.argmax().item())
                last_confidence = float(probabilities[last_prediction_id].item())

            if show_motion:
                motion_color = (0, 255, 0) if motion_state == "静止" else (0, 0, 255)
                put_cn_text_right(
                    frame,
                    motion_state,
                    y=10,
                    font_size=32,
                    color=motion_color,
                )

            if event_window.active:
                put_cn_text(
                    frame,
                    "动作采集中...",
                    (20, 10),
                    32,
                    (255, 255, 255),
                )
            elif last_prediction_id is None:
                put_cn_text(
                    frame,
                    "等待动作",
                    (20, 10),
                    32,
                    (255, 255, 255),
                )
            else:
                put_cn_text(
                    frame,
                    f"{class_names[last_prediction_id]} 置信度 {last_confidence:.2f}",
                    (20, 10),
                    32,
                    (0, 255, 0),
                )

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def camera_static_test(model_file=STATIC_MODEL_FILE):
    """加载静态 MLP，逐帧识别摄像头画面。"""
    if not model_file.exists():
        raise FileNotFoundError(f"找不到九类静态模型：{model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    model = StaticMLP(
        input_dim=checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=len(class_names),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头")

    prediction_queue = deque(maxlen=7)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("摄像头读取失败")
                break

            feature = extract_feature(frame, draw=True)
            if hand_is_present(feature):
                normalized = transform_frames(feature[None, :], mean, scale)
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(torch.from_numpy(normalized).to(device)),
                        dim=1,
                    )[0]

                current_id = int(probabilities.argmax().item())
                prediction_queue.append(current_id)
                voted_id = Counter(prediction_queue).most_common(1)[0][0]
                confidence = float(probabilities[voted_id].item())
            else:
                prediction_queue.clear()
                voted_id = None

            put_cn_text_right(
                frame,
                "静态模式",
                y=10,
                font_size=32,
                color=(0, 255, 0),
            )
            if voted_id is None:
                put_cn_text(frame, "等待手势", (20, 10), 32, (255, 255, 255))
            else:
                put_cn_text(
                    frame,
                    f"{class_names[voted_id]} 置信度 {confidence:.2f}",
                    (20, 10),
                    32,
                    (0, 255, 0),
                )

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def camera_rich_test(model_file=RICH_MODEL_FILE):
    """加载 380 维相对特征模型，并显示骨架、手部关键点。"""
    if not model_file.exists():
        raise FileNotFoundError(
            f"找不到相对特征模型：{model_file}\n"
            "请先运行 sign_system_v18_complete_run.py --rich-static 完成训练。"
        )

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    model = StaticMLP(
        input_dim=checkpoint["feature_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=len(class_names),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    special_model = None
    special_mean = None
    special_scale = None
    special_class_ids = RICH_CONFUSION_CLASSES
    special_model_file = (
        RICH_CONFUSION_MODEL_FILE
        if RICH_CONFUSION_MODEL_FILE.exists()
        else RICH_PAIR_MODEL_FILE
    )
    if special_model_file.exists():
        special_checkpoint = torch.load(
            special_model_file,
            map_location=device,
            weights_only=False,
        )
        special_class_ids = tuple(special_checkpoint["class_ids"])
        special_mean = special_checkpoint["scaler_mean"]
        special_scale = special_checkpoint["scaler_scale"]
        special_model = StaticMLP(
            input_dim=special_checkpoint["feature_dim"],
            hidden_dim=special_checkpoint["hidden_dim"],
            num_classes=special_checkpoint["num_classes"],
        ).to(device)
        special_model.load_state_dict(special_checkpoint["model"])
        special_model.eval()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    prediction_queue = deque(maxlen=7)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("摄像头读取失败")
                break

            feature = extract_rich_feature(frame, draw=True)
            hand_present = bool(np.any(np.abs(feature[:260]) > 1e-6))
            if hand_present:
                normalized = transform_frames(feature[None, :], mean, scale)
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(torch.from_numpy(normalized).to(device)),
                        dim=1,
                    )[0]
                prediction_queue.append(int(probabilities.argmax().item()))
                voted_id = Counter(prediction_queue).most_common(1)[0][0]
                confidence = float(probabilities[voted_id].item())
                if special_model is not None and voted_id in special_class_ids:
                    special_normalized = transform_frames(
                        feature[None, :], special_mean, special_scale
                    )
                    with torch.no_grad():
                        special_probabilities = torch.softmax(
                            special_model(
                                torch.from_numpy(special_normalized).to(device)
                            ),
                            dim=1,
                        )[0]
                    special_id = int(special_probabilities.argmax().item())
                    voted_id = int(special_class_ids[special_id])
                    confidence = float(special_probabilities[special_id].item())
            else:
                prediction_queue.clear()
                voted_id = None

            put_cn_text_right(
                frame,
                "相对特征模式",
                y=10,
                font_size=32,
                color=(0, 255, 0),
            )
            if voted_id is None:
                put_cn_text(frame, "等待手势", (20, 10), 32, (255, 255, 255))
            else:
                put_cn_text(
                    frame,
                    f"{class_names[voted_id]} 置信度 {confidence:.2f}",
                    (20, 10),
                    32,
                    (0, 255, 0),
                )
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def camera_image_test(model_file=IMAGE_MODEL_FILE, show_landmarks=True):
    """加载原始图像 CNN，直接识别 640x480 摄像头画面。"""
    if not model_file.exists():
        raise FileNotFoundError(f"找不到图像模型：{model_file}")

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = TinyImageCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    prediction_queue = deque(maxlen=7)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("摄像头读取失败")
                break

            image = preprocess_image_frame(frame)
            with torch.no_grad():
                probabilities = torch.softmax(
                    model(image_batch_to_device(torch.from_numpy(image[None, ...]))),
                    dim=1,
                )[0]
            current_id = int(probabilities.argmax().item())
            prediction_queue.append(current_id)
            voted_id = Counter(prediction_queue).most_common(1)[0][0]
            confidence = float(probabilities[voted_id].item())

            if show_landmarks:
                extract_rich_feature(frame, draw=True)

            put_cn_text_right(
                frame,
                "原始图像模式",
                y=10,
                font_size=32,
                color=(0, 255, 0),
            )
            put_cn_text(
                frame,
                f"{class_names[voted_id]} 置信度 {confidence:.2f}",
                (20, 10),
                32,
                (0, 255, 0),
            )

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    set_seed()
    if "--image-14" in sys.argv:
        X, y, groups = build_image_dataset()
        validate_image_model(X, y, groups)
        train_image_final_model(X, y)
        return

    if "--rich-static" in sys.argv:
        X, y, groups = build_rich_dataset()
        validate_rich_model(X, y, groups)
        train_rich_final_model(X, y)
        validate_rich_pair_model(X, y, groups)
        train_rich_pair_final_model(X, y)
        validate_rich_confusion_model(X, y, groups)
        train_rich_confusion_final_model(X, y)
        return

    if "--dynamic-only" in sys.argv:
        X, y, groups = build_dynamic_sequence_dataset()
        validate_leave_one_video_out(
            X,
            y,
            groups,
            class_names=DYNAMIC_CLASS_NAMES,
        )
        train_final_model(
            X,
            y,
            model_file=DYNAMIC_MODEL_FILE,
            class_names=DYNAMIC_CLASS_NAMES,
        )
        camera_test(show_motion=True, model_file=DYNAMIC_MODEL_FILE)
        return

    if "--static-nine" in sys.argv or "--full-sequence" not in sys.argv:
        X, y, groups = build_static_nine_dataset()
        validate_static_model(X, y, groups)
        train_static_final_model(X, y)
        camera_static_test()
        return

    X, y, groups = build_sequence_dataset()
    validate_leave_one_video_out(X, y, groups)
    train_final_model(X, y)
    camera_test()


if __name__ == "__main__":
    main()
