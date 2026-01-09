"""
Performance test for Phase 1.

Tests store/retrieve performance with 1000 fragments.
"""

from src.reconstructions.core import Fragment
from src.reconstructions.store import FragmentStore
import numpy as np
import time
import tempfile
from pathlib import Path

def performance_test():
    """Test performance with 1000 fragments."""
    
    print("=" * 60)
    print("Phase 1 Performance Test")
    print("=" * 60)
    
    num_fragments = 1000
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perf_test.db"
        
        with FragmentStore(str(db_path)) as store:
            
            # Test 1: Save performance
            print(f"\n1. Saving {num_fragments} fragments...")
            fragments = []
            
            start = time.time()
            for i in range(num_fragments):
                # Create diverse fragments
                content = {"semantic": f"Memory {i}"}
                
                # Add emotional data to some
                if i % 3 == 0:
                    content["emotional"] = {
                        "valence": np.random.random(),
                        "arousal": np.random.random()
                    }
                
                # Add vector embedding to some
                if i % 2 == 0:
                    content["embedding"] = np.random.random(128).tolist()
                
                fragment = Fragment(
                    content=content,
                    initial_salience=np.random.random(),
                    tags=[f"tag{i % 10}"]
                )
                fragments.append(fragment)
                store.save(fragment)
            
            save_time = time.time() - start
            print(f"   ✓ Saved {num_fragments} fragments in {save_time:.3f}s")
            print(f"   ✓ Average: {save_time/num_fragments*1000:.2f}ms per fragment")
            
            # Test 2: Retrieve performance
            print(f"\n2. Retrieving {num_fragments} fragments...")
            
            start = time.time()
            for fragment in fragments:
                retrieved = store.get(fragment.id)
                assert retrieved is not None
            
            retrieve_time = time.time() - start
            print(f"   ✓ Retrieved {num_fragments} fragments in {retrieve_time:.3f}s")
            print(f"   ✓ Average: {retrieve_time/num_fragments*1000:.2f}ms per fragment")
            
            # Test 3: Time range query performance
            print(f"\n3. Testing time range queries...")
            
            start = time.time()
            for _ in range(100):
                start_time = time.time() - 600  # Last 10 minutes
                end_time = time.time()
                results = store.find_by_time_range(start_time, end_time)
            
            query_time = time.time() - start
            print(f"   ✓ Ran 100 time range queries in {query_time:.3f}s")
            print(f"   ✓ Average: {query_time/100:.4f}s per query")
            
            # Test 4: Domain query performance
            print(f"\n4. Testing domain queries...")
            
            start = time.time()
            for _ in range(100):
                results = store.find_by_domain("emotional")
            
            domain_time = time.time() - start
            print(f"   ✓ Ran 100 domain queries in {domain_time:.3f}s")
            print(f"   ✓ Average: {domain_time/100:.4f}s per query")
            
            # Test 5: Vector similarity (if embeddings present)
            print(f"\n5. Testing vector similarity search...")
            
            # Create query embedding
            query_embedding = np.random.random(128).astype(np.float32)
            
            start = time.time()
            for _ in range(100):
                results = store.find_similar_semantic(query_embedding, top_k=10)
            
            vector_time = time.time() - start
            print(f"   ✓ Ran 100 similarity searches in {vector_time:.3f}s")
            print(f"   ✓ Average: {vector_time/100:.4f}s per search")
            
            # Test 6: Database size
            db_size = db_path.stat().st_size
            print(f"\n6. Database statistics:")
            print(f"   ✓ Database size: {db_size/1024:.2f} KB ({db_size/1024/1024:.2f} MB)")
            print(f"   ✓ Average per fragment: {db_size/num_fragments:.0f} bytes")
    
    print("\n" + "=" * 60)
    print("Performance test complete!")
    print("=" * 60)
    
    # Performance criteria (loose for now)
    assert save_time < 5.0, "Saving should take less than 5 seconds for 1000 fragments"
    assert retrieve_time < 2.0, "Retrieval should take less than 2 seconds for 1000 fragments"
    print("\n✓ All performance criteria met!")

if __name__ == "__main__":
    performance_test()
