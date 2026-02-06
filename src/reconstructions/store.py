"""
Fragment storage using SQLite and vector embeddings.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from .core import Fragment


class FragmentStore:
    """
    Persistent storage for fragments.

    Uses SQLite for structured data and VectorIndex for fast
    semantic similarity search (falls back to in-memory dict if unavailable).
    """

    def __init__(self, db_path: str):
        """
        Initialize fragment store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Try to use VectorIndex for fast similarity search
        self._use_vector_index = False
        self._vector_index = None
        self._vector_index_path = self.db_path.with_suffix(".vectors")

        try:
            from .vector_index import VectorIndex
            self._vector_index = VectorIndex()
            self._use_vector_index = True

            # Load existing index if available
            if self._vector_index_path.with_suffix(".usearch").exists():
                self._vector_index.load(self._vector_index_path)
        except ImportError:
            # Fall back to in-memory dict
            pass

        # In-memory vector storage (fallback when VectorIndex unavailable)
        self.embeddings: dict[str, np.ndarray] = {}

        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Main fragments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                content TEXT NOT NULL,
                bindings TEXT NOT NULL,
                initial_salience REAL NOT NULL,
                access_log TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT NOT NULL
            )
        """)
        
        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON fragments(created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_salience 
            ON fragments(initial_salience DESC)
        """)
        
        self.conn.commit()
    
    def save(self, fragment: Fragment) -> None:
        """
        Save a fragment to the store.
        
        Args:
            fragment: Fragment to save
        """
        cursor = self.conn.cursor()
        
        # Serialize lists/dicts to JSON
        data = fragment.to_dict()
        
        cursor.execute("""
            INSERT OR REPLACE INTO fragments 
            (id, created_at, content, bindings, initial_salience, 
             access_log, source, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data["created_at"],
            json.dumps(data["content"]),
            json.dumps(data["bindings"]),
            data["initial_salience"],
            json.dumps(data["access_log"]),
            data["source"],
            json.dumps(data["tags"])
        ))
        
        self.conn.commit()

        # Extract and store embedding if present
        if "semantic" in fragment.content and isinstance(fragment.content["semantic"], list):
            embedding = np.array(fragment.content["semantic"], dtype=np.float32)

            if self._use_vector_index and self._vector_index is not None:
                # Use VectorIndex for fast search (requires 384-dim vectors)
                if len(embedding) == 384:
                    self._vector_index.add(fragment.id, embedding)
                    # Auto-save index to persist across process boundaries
                    self._vector_index.save(self._vector_index_path)
                else:
                    # Fall back to in-memory for non-standard dimensions
                    self.embeddings[fragment.id] = embedding
            else:
                self.embeddings[fragment.id] = embedding
    
    def get(self, fragment_id: str) -> Optional[Fragment]:
        """
        Retrieve a fragment by ID.

        Args:
            fragment_id: Fragment ID

        Returns:
            Fragment if found, None otherwise
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM fragments WHERE id = ?
        """, (fragment_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        # Deserialize JSON fields
        data = {
            "id": row["id"],
            "created_at": row["created_at"],
            "content": json.loads(row["content"]),
            "bindings": json.loads(row["bindings"]),
            "initial_salience": row["initial_salience"],
            "access_log": json.loads(row["access_log"]),
            "source": row["source"],
            "tags": json.loads(row["tags"])
        }

        return Fragment.from_dict(data)

    def get_many(self, fragment_ids: list[str]) -> dict[str, Fragment]:
        """
        Retrieve multiple fragments by ID in a single query (batch loading).

        This is much faster than calling get() in a loop, avoiding N+1 queries.

        Args:
            fragment_ids: List of fragment IDs to retrieve

        Returns:
            Dict mapping fragment_id -> Fragment for found fragments
        """
        if not fragment_ids:
            return {}

        cursor = self.conn.cursor()

        # Build query with placeholders for IN clause
        placeholders = ','.join('?' * len(fragment_ids))
        cursor.execute(f"""
            SELECT * FROM fragments WHERE id IN ({placeholders})
        """, fragment_ids)

        rows = cursor.fetchall()

        fragments = {}
        for row in rows:
            # Deserialize JSON fields
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }

            fragments[row["id"]] = Fragment.from_dict(data)

        return fragments
    
    def delete(self, fragment_id: str) -> bool:
        """
        Delete a fragment.
        
        Args:
            fragment_id: Fragment ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            DELETE FROM fragments WHERE id = ?
        """, (fragment_id,))
        
        self.conn.commit()

        # Remove embedding if exists
        if self._use_vector_index and self._vector_index is not None:
            self._vector_index.remove(fragment_id)

        if fragment_id in self.embeddings:
            del self.embeddings[fragment_id]

        return cursor.rowcount > 0
    
    def find_by_time_range(self, start: float, end: float) -> List[Fragment]:
        """
        Find fragments within a time range.
        
        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            
        Returns:
            List of fragments in time range
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM fragments 
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at ASC
        """, (start, end))
        
        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))
        
        return fragments
    
    def find_by_domain(self, domain: str) -> List[Fragment]:
        """
        Find fragments containing a specific domain.
        
        Args:
            domain: Domain name (e.g., "semantic", "emotional")
            
        Returns:
            List of fragments with that domain
        """
        cursor = self.conn.cursor()
        
        # Use JSON search for domain key
        cursor.execute("""
            SELECT * FROM fragments 
            WHERE json_extract(content, ?) IS NOT NULL
            ORDER BY created_at DESC
        """, (f"$.{domain}",))
        
        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))

        return fragments

    def get_all_fragments(self, limit: Optional[int] = None) -> List[Fragment]:
        """
        Get all fragments from the store.

        Args:
            limit: Optional limit on number of fragments to return

        Returns:
            List of all fragments (or up to limit)
        """
        cursor = self.conn.cursor()

        if limit is not None:
            cursor.execute("""
                SELECT * FROM fragments
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT * FROM fragments
                ORDER BY created_at DESC
            """)

        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))

        return fragments

    def find_similar_semantic(self, embedding: np.ndarray, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Find fragments with similar semantic embeddings.

        Uses VectorIndex for fast HNSW search when available,
        falls back to brute-force otherwise.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (fragment_id, similarity_score) tuples
        """
        # Try VectorIndex first (fast path for 384-dim vectors)
        if (
            self._use_vector_index
            and self._vector_index is not None
            and self._vector_index.count() > 0
            and len(embedding) == 384
        ):
            return self._vector_index.search(embedding, limit=top_k)

        # Fall back to brute-force search
        if len(self.embeddings) == 0:
            return []

        # Normalize query embedding
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Compute cosine similarity with all stored embeddings
        similarities = []
        for fid, stored_emb in self.embeddings.items():
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            similarity = np.dot(query_norm, stored_norm)
            similarities.append((fid, float(similarity)))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
    
    def get_recent_fragments(
        self,
        hours: float = 24.0,
        min_salience: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Fragment]:
        """
        Get recent fragments using SQL WHERE instead of loading all.

        Args:
            hours: How far back to look (default 24h)
            min_salience: Minimum salience threshold
            limit: Optional max number to return

        Returns:
            List of matching fragments, ordered by created_at DESC
        """
        import time
        cutoff = time.time() - (hours * 3600)

        cursor = self.conn.cursor()

        query = """
            SELECT * FROM fragments
            WHERE created_at >= ? AND initial_salience >= ?
            ORDER BY created_at DESC
        """
        params: list = [cutoff, min_salience]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)

        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))

        return fragments

    def record_access(self, fragment_id: str, timestamp: float) -> None:
        """
        Record an access to a fragment (for rehearsal tracking).

        Args:
            fragment_id: Fragment ID
            timestamp: Access timestamp
        """
        fragment = self.get(fragment_id)
        if fragment is None:
            return

        fragment.access_log.append(timestamp)
        self.save(fragment)

    def record_access_batch(self, fragment_ids: List[str], timestamp: float) -> int:
        """
        Record access for multiple fragments in a single SQL operation.

        Much faster than calling record_access() in a loop since it avoids
        N individual get+save round-trips.

        Args:
            fragment_ids: List of fragment IDs to update
            timestamp: Access timestamp

        Returns:
            Number of fragments updated
        """
        if not fragment_ids:
            return 0

        cursor = self.conn.cursor()
        updated = 0

        # Use a single transaction for all updates
        for frag_id in fragment_ids:
            # Append timestamp to the JSON access_log array in-place
            cursor.execute("""
                UPDATE fragments
                SET access_log = json_insert(access_log, '$[#]', ?)
                WHERE id = ?
            """, (timestamp, frag_id))
            updated += cursor.rowcount

        self.conn.commit()
        return updated
    
    def is_empty(self) -> bool:
        """
        Check if store is empty.
        
        Returns:
            True if no fragments stored
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM fragments")
        row = cursor.fetchone()
        return row["count"] == 0
    
    def close(self):
        """Close database connection and save VectorIndex."""
        # Save VectorIndex if used
        if self._use_vector_index and self._vector_index is not None:
            if self._vector_index.count() > 0:
                self._vector_index.save(self._vector_index_path)

        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
