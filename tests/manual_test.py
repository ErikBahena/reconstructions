"""
Manual test script for Phase 1.

Tests creating fragments via Python REPL-style interactions.
"""

from reconstructions.core import Fragment, Strand, Query
from reconstructions.store import FragmentStore
import numpy as np
import time

def manual_test():
    """Manual test of core functionality."""
    
    print("=" * 60)
    print("Phase 1 Manual Test")
    print("=" * 60)
    
    # Test 1: Create fragments with different content types
    print("\n1. Creating fragments with various content...")
    
    f1 = Fragment(
        content={
            "semantic": "The sky was blue and clear",
            "emotional": {"valence": 0.7, "arousal": 0.3}
        },
        initial_salience=0.8,
        tags=["weather", "observation"]
    )
    print(f"   ✓ Created fragment: {f1.id[:8]}... with semantic + emotional content")
    
    f2 = Fragment(
        content={
            "semantic": "I felt peaceful watching the clouds",
            "emotional": {"valence": 0.8, "arousal": 0.2}
        },
        initial_salience=0.9,
        tags=["feeling", "observation"]
    )
    print(f"   ✓ Created fragment: {f2.id[:8]}... with high salience")
    
    f3 = Fragment(
        content={"visual": [0.1, 0.9, 0.3, 0.5]},
        initial_salience=0.5
    )
    print(f"   ✓ Created fragment: {f3.id[:8]}... with only visual data")
    
    # Test 2: Serialization
    print("\n2. Testing serialization...")
    data = f1.to_dict()
    restored = Fragment.from_dict(data)
    assert restored.content == f1.content
    print("   ✓ Serialization roundtrip successful")
    
    # Test 3: Store operations
    print("\n3. Testing store operations...")
    with FragmentStore("test_manual.db") as store:
        store.save(f1)
        store.save(f2)
        store.save(f3)
        print(f"   ✓ Saved 3 fragments to database")
        
        retrieved = store.get(f1.id)
        assert retrieved.id == f1.id
        print(f"   ✓ Retrieved fragment by ID")
        
        # Test time range query
        now = time.time()
        recent = store.find_by_time_range(now - 60, now + 60)
        print(f"   ✓ Found {len(recent)} fragments in time range")
        
        # Test domain query
        emotional = store.find_by_domain("emotional")
        print(f"   ✓ Found {len(emotional)} fragments with 'emotional' domain")
        
        visual = store.find_by_domain("visual")
        print(f"   ✓ Found {len(visual)} fragments with 'visual' domain")
    
    # Test 4: Query hashing
    print("\n4. Testing query hashing...")
    q1 = Query(semantic="test query", min_salience=0.5)
    q2 = Query(semantic="test query", min_salience=0.5)
    q3 = Query(semantic="different query", min_salience=0.5)
    
    assert q1.to_hash() == q2.to_hash()
    assert q1.to_hash() != q3.to_hash()
    print("   ✓ Query hashing works correctly")
    
    # Test 5: Strand creation
    print("\n5. Testing strand creation...")
    strand = Strand(
        fragments=[f1.id, f2.id],
        assembly_context={"mode": "test"},
        coherence_score=0.95,
        variance=0.1
    )
    print(f"   ✓ Created strand: {strand.id[:8]}... with 2 fragments")
    
    strand_data = strand.to_dict()
    restored_strand = Strand.from_dict(strand_data)
    assert restored_strand.fragments == strand.fragments
    print("   ✓ Strand serialization roundtrip successful")
    
    print("\n" + "=" * 60)
    print("All manual tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    manual_test()
