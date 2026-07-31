"""Exception types for the ONNX inference service."""


class OnnxInferError(Exception):
    """Base class for all onnx_infer errors."""


class MetaSchemaError(OnnxInferError):
    """A ``meta.json`` is missing required fields or has an unsupported version."""


class UnknownArchError(OnnxInferError):
    """A ``meta.json`` names an architecture with no registered handler."""
