import socket
import time

from sign_edge.app import BoardEventReceiver, LatestBoardTextSender


def test_sender_delivers_chinese_label_over_udp():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    port = receiver.getsockname()[1]

    sender = LatestBoardTextSender(f"udp://127.0.0.1:{port}/text")
    expected_labels = {
        "不": "不",
        "不是": "不是",
        "不知道": "不知道",
        "不行": "不行",
        "什么or哪里": "哪里",
        "他": "他",
        "你": "你",
        "去": "去",
        "可以": "可以",
        "有": "有",
        "要": "要",
        "吃饭": "吃饭",
        "喝水": "喝水",
        "帮助": "帮助",
        "帮我": "帮我",
        "回": "回",
        "家": "家",
        "想": "想",
        "我": "我",
        "是": "是",
        "来": "来",
        "没有": "没有",
        "请": "请",
        "谢谢": "谢谢",
    }
    assert "好" not in sender.LABELS
    for label, expected in expected_labels.items():
        sender.submit(label, "model_ready")
        deadline = time.monotonic() + 1.0
        payload = None
        while time.monotonic() < deadline:
            try:
                payload, _ = receiver.recvfrom(64)
                break
            except socket.timeout:
                pass
        assert payload == expected.encode("utf-8")

    receiver.close()


def test_sender_repeats_same_label_after_retry_interval():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    port = receiver.getsockname()[1]

    sender = LatestBoardTextSender(f"udp://127.0.0.1:{port}/text")
    sender.submit("吃饭", "model_ready")
    first, _ = receiver.recvfrom(64)
    assert first == "吃饭".encode("utf-8")

    time.sleep(1.05)
    sender.submit("吃饭", "model_ready")
    second, _ = receiver.recvfrom(64)
    receiver.close()
    assert second == "吃饭".encode("utf-8")


def test_board_event_receiver_keeps_latest_confirmation_event():
    event_receiver = BoardEventReceiver(host="127.0.0.1", port=0)
    event_receiver.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"help_confirmed", ("127.0.0.1", event_receiver.port))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if event_receiver.latest_event() == "help_confirmed":
                break
            time.sleep(0.01)
        assert event_receiver.latest_event() == "help_confirmed"
    finally:
        sender.close()
        event_receiver.close()


def test_board_event_receiver_expires_confirmation_event_after_ttl():
    event_receiver = BoardEventReceiver(
        host="127.0.0.1", port=0, ttl_seconds=0.05
    )
    event_receiver.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b"help_confirmed", ("127.0.0.1", event_receiver.port))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if event_receiver.latest_event() == "help_confirmed":
                break
            time.sleep(0.01)
        assert event_receiver.latest_event() == "help_confirmed"
        time.sleep(0.08)
        assert event_receiver.latest_event() is None
    finally:
        sender.close()
        event_receiver.close()
