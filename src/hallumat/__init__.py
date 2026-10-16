"""HalluMat: a materials-science hallucination benchmark and the HalluMatDetector pipeline.

The base install analyses the released dataset (`hallumat.analysis`). Generating and labelling responses
(`hallumat.generation`, `hallumat.labeling`) needs the `models` extra, and the detector (`hallumat.detector`)
needs the `detector` extra. Heavy dependencies are imported only when they are used.
"""

__version__ = "2.0.0"
