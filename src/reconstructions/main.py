"""
Main entry point for the Memory Reconstruction System.

This module provides the primary way to run the memory system,
handling configuration, initialization, and the main execution loop.
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".reconstructions"


@dataclass
class Config:
    """Configuration for the memory system."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    db_name: str = "memory.db"
    
    # Engine settings
    decay_rate: float = 0.5
    rehearsal_bonus: float = 0.1
    max_fragments: int = 10
    
    # CLI settings
    show_welcome: bool = True
    
    @property
    def db_path(self) -> str:
        return str(self.data_dir / self.db_name)
    
    def to_dict(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "db_name": self.db_name,
            "decay_rate": self.decay_rate,
            "rehearsal_bonus": self.rehearsal_bonus,
            "max_fragments": self.max_fragments,
            "show_welcome": self.show_welcome
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(
            data_dir=Path(data.get("data_dir", DEFAULT_DATA_DIR)),
            db_name=data.get("db_name", "memory.db"),
            decay_rate=data.get("decay_rate", 0.5),
            rehearsal_bonus=data.get("rehearsal_bonus", 0.1),
            max_fragments=data.get("max_fragments", 10),
            show_welcome=data.get("show_welcome", True)
        )
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.data_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls.from_dict(json.load(f))


def setup_data_directory(config: Config) -> None:
    """Ensure data directory exists."""
    config.data_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Memory Reconstruction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        Start interactive CLI
  %(prog)s consolidate            Run consolidation daemon
  %(prog)s consolidate --interval 30  Consolidate every 30 seconds
  %(prog)s --data-dir ./my        Use custom data directory
  %(prog)s --new                  Start with fresh database
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="cli",
        choices=["cli", "consolidate"],
        help="Command to run (default: cli)"
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=None,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})"
    )

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database filename"
    )

    parser.add_argument(
        "--new",
        action="store_true",
        help="Start with a new/empty database"
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config file"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Consolidation interval in seconds (default: 60)"
    )

    return parser.parse_args()


def consolidate_daemon(db_path: str, interval: int = 60) -> int:
    """
    Run consolidation as a long-running daemon.

    This solves the problem where each hook spawns a fresh process
    that resets the consolidation timer. Instead, a single persistent
    process runs consolidation on schedule.

    Args:
        db_path: Path to the SQLite database
        interval: Seconds between consolidation attempts

    Returns:
        Exit code (0 for normal exit)
    """
    import time
    import signal
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [consolidation] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    log = logging.getLogger("consolidation")

    from .store import FragmentStore
    from .consolidation import ConsolidationScheduler, ConsolidationConfig

    running = True

    def handle_signal(signum, frame):
        nonlocal running
        log.info(f"Received signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log.info(f"Starting consolidation daemon (db={db_path}, interval={interval}s)")

    store = FragmentStore(db_path)
    config = ConsolidationConfig()
    config.CONSOLIDATION_INTERVAL_SECONDS = float(interval)

    scheduler = ConsolidationScheduler(store, config=config)
    # Reset last_consolidation to epoch so first run happens immediately
    scheduler.state.last_consolidation = 0.0

    poll_interval = min(interval, 10)
    total_runs = 0

    try:
        while running:
            if scheduler.should_consolidate():
                try:
                    stats = scheduler.consolidate()
                    total_runs += 1
                    log.info(
                        f"Consolidation #{total_runs}: "
                        f"rehearsed={stats['rehearsed_count']}, "
                        f"bindings={stats['bindings_strengthened']}, "
                        f"patterns={stats['patterns_discovered']}, "
                        f"duration={stats['duration_ms']}ms"
                    )
                except Exception as e:
                    log.error(f"Consolidation failed: {e}")

            time.sleep(poll_interval)
    finally:
        store.close()
        log.info(f"Daemon stopped after {total_runs} consolidation runs")

    return 0


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success)
    """
    args = parse_args()

    # Load or create configuration
    if args.config and args.config.exists():
        config = Config.load(args.config)
    else:
        config = Config()

    # Override with command line args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.db:
        config.db_name = args.db

    # Setup data directory
    setup_data_directory(config)

    # Route to subcommand
    if args.command == "consolidate":
        return consolidate_daemon(config.db_path, interval=args.interval)

    # Handle --new flag
    if args.new:
        db_path = Path(config.db_path)
        if db_path.exists():
            db_path.unlink()
            print(f"Removed existing database: {db_path}")

    # Import CLI here to avoid circular imports
    from .cli import CLI

    # Run CLI
    cli = CLI(db_path=config.db_path)

    try:
        cli.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
