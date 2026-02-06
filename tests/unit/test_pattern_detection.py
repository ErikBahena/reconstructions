"""
Unit tests for cross-session pattern detection.
"""

import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from src.reconstructions.patterns import (
    CrossSessionPatternDetector,
    TemporalPattern,
    WorkflowPattern,
    ProjectPattern
)
from src.reconstructions.core import Fragment
from src.reconstructions.store import FragmentStore


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_patterns():
    """Create temporary patterns file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def store(temp_db):
    """Create fragment store."""
    return FragmentStore(str(temp_db))


@pytest.fixture
def detector(store, temp_patterns):
    """Create pattern detector."""
    return CrossSessionPatternDetector(store, patterns_path=temp_patterns)


def test_detector_initialization(detector):
    """Test detector initializes correctly."""
    assert detector.store is not None
    assert len(detector.temporal_patterns) == 0
    assert len(detector.workflow_patterns) == 0
    assert len(detector.project_patterns) == 0


def test_temporal_pattern_serialization():
    """Test temporal pattern to/from dict."""
    pattern = TemporalPattern(
        pattern_type="weekly",
        description="Tuesday: streaming work",
        confidence=0.8,
        examples=["frag1", "frag2"],
        time_signature={"day_of_week": 2}
    )

    data = pattern.to_dict()
    restored = TemporalPattern.from_dict(data)

    assert restored.pattern_type == pattern.pattern_type
    assert restored.description == pattern.description
    assert restored.confidence == pattern.confidence
    assert restored.examples == pattern.examples


def test_workflow_pattern_serialization():
    """Test workflow pattern to/from dict."""
    pattern = WorkflowPattern(
        steps=["git", "status", "add", "commit"],
        frequency=10,
        avg_duration_minutes=5.5,
        examples=[["f1", "f2", "f3", "f4"]]
    )

    data = pattern.to_dict()
    restored = WorkflowPattern.from_dict(data)

    assert restored.steps == pattern.steps
    assert restored.frequency == pattern.frequency
    assert restored.avg_duration_minutes == pattern.avg_duration_minutes


def test_project_pattern_serialization():
    """Test project pattern to/from dict."""
    import numpy as np

    pattern = ProjectPattern(
        project_name="streaming",
        keywords=["rtmp", "video", "server"],
        fragment_ids=["f1", "f2", "f3"],
        centroid=np.array([0.1, 0.2, 0.3])
    )

    data = pattern.to_dict()
    restored = ProjectPattern.from_dict(data)

    assert restored.project_name == pattern.project_name
    assert restored.keywords == pattern.keywords
    assert restored.fragment_ids == pattern.fragment_ids
    assert restored.centroid.tolist() == pattern.centroid.tolist()


def test_detect_temporal_patterns_insufficient_data(detector, store):
    """Test temporal detection with insufficient data."""
    # Add only a few fragments
    for i in range(3):
        fragment = Fragment(
            content={"semantic": f"test {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    patterns = detector.detect_temporal_patterns()
    assert len(patterns) == 0  # Not enough data


def test_detect_temporal_patterns_weekly(detector, store):
    """Test detecting weekly patterns."""
    # Simulate fragments on Tuesdays (day 1)
    tuesday_time = datetime(2024, 1, 2, 14, 0).timestamp()  # Tuesday

    for week in range(4):  # 4 weeks of data
        for i in range(5):  # 5 fragments each Tuesday
            fragment = Fragment(
                content={"semantic": f"streaming work week {week}"},
                initial_salience=0.5
            )
            fragment.created_at = tuesday_time + (week * 7 * 86400) + (i * 60)  # Add weeks
            store.save(fragment)

    # Add some random fragments on other days
    for day in range(7):
        if day != 1:  # Not Tuesday
            for i in range(2):
                fragment = Fragment(
                    content={"semantic": f"random work day {day}"},
                    initial_salience=0.5
                )
                fragment.created_at = tuesday_time + (day * 86400) + (i * 3600)
                store.save(fragment)

    patterns = detector.detect_temporal_patterns(min_confidence=0.3)

    # Should detect Tuesday pattern
    assert len(patterns) > 0
    tuesday_pattern = [p for p in patterns if p.time_signature.get("day_of_week") == 1]
    assert len(tuesday_pattern) > 0


def test_detect_workflow_patterns_insufficient_data(detector, store):
    """Test workflow detection with insufficient data."""
    # Add only a few fragments
    for i in range(3):
        fragment = Fragment(
            content={"semantic": f"test {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    patterns = detector.detect_workflow_patterns()
    assert len(patterns) == 0


def test_detect_workflow_patterns_git_sequence(detector, store):
    """Test detecting git workflow pattern."""
    now = time.time()

    # Simulate git workflow 5 times
    for iteration in range(5):
        base_time = now + (iteration * 3600)  # 1 hour apart

        # git status -> add -> commit sequence
        for i, command in enumerate(["git status", "git add file", "git commit message"]):
            fragment = Fragment(
                content={"semantic": command},
                initial_salience=0.5
            )
            fragment.created_at = base_time + (i * 60)  # 1 minute between commands
            store.save(fragment)

    patterns = detector.detect_workflow_patterns(min_frequency=3)

    # Should detect git workflow
    assert len(patterns) > 0
    git_patterns = [p for p in patterns if "git" in p.steps[0]]
    assert len(git_patterns) > 0


def test_detect_project_switches_insufficient_data(detector, store):
    """Test project detection with insufficient data."""
    # Add only a few fragments
    for i in range(3):
        fragment = Fragment(
            content={"semantic": f"test {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    patterns = detector.detect_project_switches()
    assert len(patterns) == 0


def test_detect_project_switches_streaming_cluster(detector, store):
    """Test detecting project clusters."""
    # Add streaming-related fragments
    for i in range(10):
        fragment = Fragment(
            content={"semantic": f"rtmp streaming video server implementation {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    # Add database-related fragments
    for i in range(10):
        fragment = Fragment(
            content={"semantic": f"database query optimization postgresql {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    patterns = detector.detect_project_switches(similarity_threshold=0.5)

    # Should detect at least one cluster
    assert len(patterns) > 0


def test_save_and_load_patterns(detector, store):
    """Test pattern persistence."""
    # Add some fragments to generate patterns
    for i in range(15):
        fragment = Fragment(
            content={"semantic": f"test fragment {i}"},
            initial_salience=0.5
        )
        store.save(fragment)

    # Detect patterns
    detector.detect_temporal_patterns()
    detector.detect_workflow_patterns()
    detector.detect_project_switches()

    # Save
    detector.save_patterns()

    # Create new detector and load
    new_detector = CrossSessionPatternDetector(store, patterns_path=detector.patterns_path)

    # Should have loaded patterns
    assert len(new_detector.temporal_patterns) == len(detector.temporal_patterns)
    assert len(new_detector.workflow_patterns) == len(detector.workflow_patterns)
    assert len(new_detector.project_patterns) == len(detector.project_patterns)


def test_get_all_patterns(detector):
    """Test getting all patterns."""
    # Add some mock patterns
    detector.temporal_patterns = [
        TemporalPattern("weekly", "Test", 0.8, [], {"day_of_week": 1})
    ]
    detector.workflow_patterns = [
        WorkflowPattern(["git", "status"], 5, 2.0, [])
    ]
    detector.project_patterns = [
        ProjectPattern("test", ["keyword"], [])
    ]

    all_patterns = detector.get_all_patterns()

    assert len(all_patterns["temporal"]) == 1
    assert len(all_patterns["workflow"]) == 1
    assert len(all_patterns["project"]) == 1


def test_extract_keywords(detector):
    """Test keyword extraction."""
    text = "git status shows changes, need to git add and git commit"
    keywords = detector._extract_keywords(text)

    assert "git" in keywords
    assert "status" in keywords
    assert "add" in keywords
    assert "commit" in keywords


def test_extract_themes(detector):
    """Test theme extraction from fragments."""
    fragments = [
        Fragment(content={"semantic": "git status check"}, initial_salience=0.5),
        Fragment(content={"semantic": "git add files"}, initial_salience=0.5),
        Fragment(content={"semantic": "git commit changes"}, initial_salience=0.5),
    ]

    themes = detector._extract_themes(fragments)

    assert "git" in themes
    assert "status" in themes or "add" in themes or "commit" in themes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
