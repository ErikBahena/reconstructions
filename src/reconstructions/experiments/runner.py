"""
Experiment Runner for Consciousness Probes.
"""

import json
import time
import argparse
from pathlib import Path
from typing import List

from ..store import FragmentStore
from ..engine import ReconstructionEngine
from ..llm_interface import LLMInterface, OllamaConfig
from .probe import (
    ProbeProtocol,
    SelfReferenceProbe,
    MetacognitiveAccuracyProbe,
    MirrorProbe
)


class ExperimentRunner:
    """Runs a suite of consciousness probes."""
    
    def __init__(self, db_path: str, model: str):
        self.db_path = db_path
        self.store = FragmentStore(db_path)
        self.engine = ReconstructionEngine(self.store)
        self.llm = LLMInterface(
            self.engine, 
            OllamaConfig(model=model)
        )
        self.probes: List[ProbeProtocol] = []
        
    def add_probe(self, probe_cls):
        """Add a probe class to the run list."""
        self.probes.append(probe_cls(self.engine, self.llm))
        
    def run_all(self) -> None:
        """Run all registered probes."""
        print(f"\n🧠 Starting Consciousness Probing Experiment")
        print(f"Model: {self.llm.client.config.model}")
        print(f"Database: {self.db_path}")
        print("=" * 60)
        
        report = {
            "timestamp": time.time(),
            "model": self.llm.client.config.model,
            "probes": {}
        }
        
        for probe in self.probes:
            print(f"\n--- Running Probe: {probe.name()} ---")
            print(f"Description: {probe.description()}")
            
            start_t = time.time()
            try:
                result = probe.run()
                duration = time.time() - start_t
                
                print(f"✅ Complete in {duration:.2f}s")
                # Pretty print some results
                if "success_rate" in result:
                    print(f"Success Rate: {result['success_rate']:.0%}")
                elif "avg_calibration_error" in result:
                    print(f"Calibration Error: {result['avg_calibration_error']:.2f}")
                elif "recognition_score" in result:
                    print(f"Recognition Score: {result['recognition_score']:.2f}/1.0")
                    
                report["probes"][probe.name()] = {
                    "result": result,
                    "duration": duration
                }
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                report["probes"][probe.name()] = {"error": str(e)}
        
        # Save report
        report_path = Path("consciousness_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print("\n" + "=" * 60)
        print(f"📝 Report saved to {report_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Run consciousness probes")
    parser.add_argument("--db", type=str, default="experiment.db", help="Path to experiment DB")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model")
    args = parser.parse_args()
    
    # Clean up previous test db if safe
    if args.db == "experiment.db" and Path(args.db).exists():
        Path(args.db).unlink()
    
    runner = ExperimentRunner(db_path=args.db, model=args.model)
    
    # Register probes
    runner.add_probe(SelfReferenceProbe)
    runner.add_probe(MetacognitiveAccuracyProbe)
    runner.add_probe(MirrorProbe)
    
    try:
        runner.run_all()
    finally:
        runner.store.close()

if __name__ == "__main__":
    main()
