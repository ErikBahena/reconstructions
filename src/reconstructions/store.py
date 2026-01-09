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
    
    Uses SQLite for structured data and in-memory numpy arrays
    for vector embeddings (semantic search).
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
        
        # In-memory vector storage (will scale this later)
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
            self.embeddings[fragment.id] = np.array(fragment.content["semantic"], dtype=np.float32)
    
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
    
    def find_similar_semantic(self, embedding: np.ndarray, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Find fragments with similar semantic embeddings.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (fragment_id, similarity_score) tuples
        """
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
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
