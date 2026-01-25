# src/reconstructions/vector_index.py
"""
Vector similarity index using USearch HNSW.

Provides fast approximate nearest neighbor search for 384-dimensional
vectors using Hierarchical Navigable Small World graphs.
"""

import json
import numpy as np
from pathlib import Path

from usearch.index import Index


class VectorIndex:
    """
    Fast vector similarity index using USearch HNSW.

    Maps string IDs to 384-dimensional vectors and supports
    efficient similarity search at scale (<5ms at 10k vectors).
    """

    NDIM = 384

    def __init__(self):
        """Initialize an empty vector index."""
        self._index = Index(ndim=self.NDIM, metric="cos", dtype="f32")
        self._id_to_key: dict[str, int] = {}  # string ID -> int key
        self._key_to_id: dict[int, str] = {}  # int key -> string ID
        self._next_key: int = 0

    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add or update a vector in the index.

        Args:
            id: String identifier for the vector
            vector: 384-dimensional numpy array
        """
        if id in self._id_to_key:
            # Update: remove old entry first
            self.remove(id)

        key = self._next_key
        self._next_key += 1

        self._index.add(key, vector.astype(np.float32))
        self._id_to_key[id] = key
        self._key_to_id[key] = id

    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.

        Args:
            id: String identifier to remove

        Returns:
            True if removed, False if not found
        """
        if id not in self._id_to_key:
            return False

        key = self._id_to_key[id]

        # USearch supports removal
        self._index.remove(key)

        del self._id_to_key[id]
        del self._key_to_id[key]

        return True

    def search(self, vector: np.ndarray, limit: int = 10) -> list[tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector (384-dim)
            limit: Maximum number of results

        Returns:
            List of (id, similarity) tuples, sorted by similarity descending.
            Similarity is in range [0, 1] where 1 is identical.
        """
        if self.count() == 0 or limit < 1:
            return []

        # Ensure we don't request more than available
        actual_limit = min(limit, self.count())

        matches = self._index.search(vector.astype(np.float32), actual_limit)

        results = []
        for key, distance in zip(matches.keys, matches.distances):
            if key in self._key_to_id:
                # Convert cosine distance to similarity: similarity = 1 - distance
                similarity = 1.0 - float(distance)
                results.append((self._key_to_id[key], similarity))

        return results

    def count(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._id_to_key)

    def contains(self, id: str) -> bool:
        """Check if an ID exists in the index."""
        return id in self._id_to_key

    def save(self, path: Path) -> None:
        """
        Save the index to disk.

        Saves both the USearch index and the ID mapping.

        Args:
            path: Base path (without extension). Will create:
                  - {path}.usearch - the HNSW index
                  - {path}.json - the ID mapping
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save USearch index
        index_path = path.with_suffix(".usearch")
        self._index.save(str(index_path))

        # Save ID mapping
        mapping_path = path.with_suffix(".json")
        mapping = {
            "id_to_key": self._id_to_key,
            "key_to_id": {str(k): v for k, v in self._key_to_id.items()},
            "next_key": self._next_key,
        }
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)

    def load(self, path: Path) -> None:
        """
        Load the index from disk.

        Args:
            path: Base path (without extension). Expects:
                  - {path}.usearch - the HNSW index
                  - {path}.json - the ID mapping
        """
        path = Path(path)

        # Load USearch index
        index_path = path.with_suffix(".usearch")
        self._index = Index(ndim=self.NDIM, metric="cos", dtype="f32")
        self._index.load(str(index_path))

        # Load ID mapping
        mapping_path = path.with_suffix(".json")
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        self._id_to_key = mapping["id_to_key"]
        self._key_to_id = {int(k): v for k, v in mapping["key_to_id"].items()}
        self._next_key = mapping["next_key"]
