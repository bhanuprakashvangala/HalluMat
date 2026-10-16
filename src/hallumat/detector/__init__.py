"""HalluMatDetector: intrinsic self-checks, retrieval-backed NLI verification, a reliability score and a fact
graph. Models load only in `HalluMatDetector.from_pretrained`; tests pass their own."""

from hallumat.detector.config import DetectorConfig
from hallumat.detector.pipeline import DetectionReport, HalluMatDetector, format_report, save_plots

__all__ = ["DetectionReport", "DetectorConfig", "HalluMatDetector", "format_report", "save_plots"]
