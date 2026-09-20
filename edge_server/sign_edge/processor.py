from dataclasses import asdict, dataclass
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProcessingResult:
    status: str
    label: str
    confidence: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


class WaitingModelProcessor:
    """Protocol-compatible placeholder until the trained model is supplied."""

    def process(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        del frame, frame_id
        return ProcessingResult(
            status="waiting_model",
            label="",
            confidence=0.0,
        )


@dataclass
class PairModel:
    class_ids: tuple[int, int]
    model: object
    mean: np.ndarray
    scale: np.ndarray


class RichModelProcessor:
    """Run the trained 25-class rich-feature MLP on incoming camera frames."""

    DEFAULT_FEATURE_SOURCE = PACKAGE_ROOT / "model_runtime"
    BASE_MODULE = "sign_system_v18_complete_run"
    DEFAULT_PAIR_MODEL_NAMES = {
        "have_want": "sign_rich_have_want_pair.pth",
        "good_thanks": "sign_rich_good_thanks_pair.pth",
    }

    def __init__(
        self,
        model_path: str | Path,
        feature_source: str | Path | None = None,
    ) -> None:
        import torch

        self._torch = torch
        source = Path(feature_source or self.DEFAULT_FEATURE_SOURCE)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        self._base = importlib.import_module(self.BASE_MODULE)
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.model_path}")

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False,
        )
        self.feature_dim = int(checkpoint["feature_dim"])
        self.num_classes = int(checkpoint["num_classes"])
        self.class_names = {
            int(class_id): str(name)
            for class_id, name in checkpoint["class_names"].items()
        }
        self.rotation_degrees = int(checkpoint.get("rotation_degrees", 180))
        if self.rotation_degrees != 180:
            raise ValueError(
                f"unsupported model rotation: {self.rotation_degrees} degrees"
            )

        self._mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self._scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self._model = self._base.StaticMLP(
            input_dim=self.feature_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_classes=self.num_classes,
        ).cpu()
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        self.pair_models = {}
        for pair_name, pair_name_file in self.DEFAULT_PAIR_MODEL_NAMES.items():
            pair_path = self.model_path.parent / pair_name_file
            if pair_path.is_file():
                self.pair_models[pair_name] = self._load_pair_model(pair_path)

    def _load_pair_model(self, model_path: Path) -> PairModel:
        checkpoint = self._torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        class_ids = tuple(int(value) for value in checkpoint["class_ids"])
        if len(class_ids) != 2 or int(checkpoint["num_classes"]) != 2:
            raise ValueError(f"invalid binary pair model: {model_path}")
        model = self._base.StaticMLP(
            input_dim=int(checkpoint["feature_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_classes=2,
        ).cpu()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return PairModel(
            class_ids=class_ids,
            model=model,
            mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
            scale=np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
        )

    def process(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        del frame_id
        try:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
            feature = np.asarray(
                self._base.extract_rich_feature(rotated, draw=False),
                dtype=np.float32,
            )
            if feature.shape != (self.feature_dim,):
                raise ValueError(
                    f"unexpected feature shape: {feature.shape}, "
                    f"expected {(self.feature_dim,)}"
                )
            if not np.any(np.abs(feature[:260]) > 1e-6):
                return ProcessingResult("no_hand", "", 0.0)

            normalized = self._base.transform_frames(
                feature[None, :], self._mean, self._scale
            )
            with self._torch.no_grad():
                logits = self._model(self._torch.from_numpy(normalized))
                probabilities = self._torch.softmax(logits, dim=1)[0]
            class_id = int(probabilities.argmax().item())
            confidence = float(probabilities[class_id].item())
            for pair_model in self.pair_models.values():
                if class_id not in pair_model.class_ids:
                    continue
                pair_normalized = self._base.transform_frames(
                    feature[None, :], pair_model.mean, pair_model.scale
                )
                with self._torch.no_grad():
                    pair_logits = pair_model.model(
                        self._torch.from_numpy(pair_normalized)
                    )
                    pair_probabilities = self._torch.softmax(pair_logits, dim=1)[0]
                pair_class_id = int(pair_probabilities.argmax().item())
                class_id = pair_model.class_ids[pair_class_id]
                confidence = float(pair_probabilities[pair_class_id].item())
                break
            return ProcessingResult(
                status="model_ready",
                label=self.class_names.get(class_id, str(class_id)),
                confidence=confidence,
            )
        except Exception as exc:
            print(f"rich model inference failed: {exc}", flush=True)
            return ProcessingResult("model_error", "", 0.0)
