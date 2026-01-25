# src/reconstructions/fast_embedder.py
"""
Fast text embedding using ONNX Runtime.

Auto-selects the best available backend:
- CUDAExecutionProvider (NVIDIA GPU)
- CoreMLExecutionProvider (Apple Silicon)
- CPUExecutionProvider (fallback)
"""

import numpy as np
from pathlib import Path

# Lazy-loaded globals
_onnx_session = None
_tokenizer = None
_backend = None


def _get_model_path() -> Path:
    """Get path to ONNX model, downloading if needed."""
    model_dir = Path(__file__).parent.parent.parent / "models"
    model_path = model_dir / "all-MiniLM-L6-v2.onnx"

    if not model_path.exists():
        # Download and convert model
        _download_and_convert_model(model_dir)

    return model_path


def _download_and_convert_model(model_dir: Path) -> None:
    """Download model and convert to ONNX format."""
    model_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    dummy_input = tokenizer(
        "Hello world",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    # Export to ONNX
    onnx_path = model_dir / "all-MiniLM-L6-v2.onnx"

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        },
        opset_version=14
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(model_dir))


def _init_session():
    """Initialize ONNX session with best available provider."""
    global _onnx_session, _tokenizer, _backend

    if _onnx_session is not None:
        return

    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Select best provider
    providers = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        _backend = "CUDAExecutionProvider"
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        if _backend is None:
            _backend = "CoreMLExecutionProvider"
    providers.append("CPUExecutionProvider")
    if _backend is None:
        _backend = "CPUExecutionProvider"

    # Load model
    model_path = _get_model_path()
    model_dir = model_path.parent

    _onnx_session = ort.InferenceSession(str(model_path), providers=providers)
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))


def _mean_pooling(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling to get sentence embedding."""
    # Expand attention mask
    mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    mask_expanded = np.broadcast_to(mask_expanded, hidden_state.shape)

    # Sum embeddings
    sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

    return sum_embeddings / sum_mask


class FastEmbedder:
    """
    Fast text embedder using ONNX Runtime.

    Auto-selects best available hardware backend.
    Returns 384-dimensional normalized vectors.
    """

    def __init__(self):
        """Initialize embedder, loading model if needed."""
        _init_session()

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            384-dimensional normalized embedding
        """
        # Tokenize
        inputs = _tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Run inference
        outputs = _onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )

        # Mean pooling
        embedding = _mean_pooling(outputs[0], inputs["attention_mask"])

        # Normalize
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (n, 384) with normalized embeddings
        """
        if not texts:
            return np.array([]).reshape(0, 384)

        # Tokenize batch
        inputs = _tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Run inference
        outputs = _onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )

        # Mean pooling
        embeddings = _mean_pooling(outputs[0], inputs["attention_mask"])

        # Normalize each
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def get_backend(self) -> str:
        """Get the active backend provider."""
        return _backend
