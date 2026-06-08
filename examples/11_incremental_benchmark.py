"""
Benchmark incremental de mejoras del enjambre.

Ejecuta 3 seeds con 5 drones y 100km de presupuesto sobre maigmo_dataset
y reporta métricas de cobertura y víctimas encontradas.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sarenv.core.loading import DatasetLoader
from sarenv.swarm.config import SwarmConfig, DroneConfig
from sarenv.swarm.simulator import SwarmSimulator

DATASET = "maigmo_dataset"
SEEDS = [42, 123, 7]
NUM_DRONES = 5
BUDGET = 100_000
MAX_STEPS = 15_000

def run_single(seed):
    loader = DatasetLoader(dataset_directory=DATASET)
    item = loader.load_environment("medium")
    
    cfg = SwarmConfig(
        num_drones=NUM_DRONES,
        num_dogs=0,
        budget_per_agent=BUDGET,
        max_steps=MAX_STEPS,
        drone_config=DroneConfig(budget=BUDGET),
    )
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)
    
    t0 = time.time()
    history = sim.run(max_steps=MAX_STEPS)
    elapsed = time.time() - t0
    
    # Métricas
    env = sim.env
    heatmap = env.probability_map
    nonzero_cells = set(zip(*np.where(heatmap > 0)))
    total_nonzero = len(nonzero_cells)
    
    # Cobertura union de todos los agentes
    all_explored = set()
    per_drone_cells = {}
    for agent in sim.agents:
        all_explored.update(agent.cells_ever_explored)
        per_drone_cells[agent.id] = len(agent.cells_ever_explored)
    
    covered_nonzero = all_explored & nonzero_cells
    coverage_pct = 100.0 * len(covered_nonzero) / total_nonzero if total_nonzero else 0
    
    # Probabilidad acumulada capturada (proxy de víctimas)
    total_prob = heatmap.sum()
    captured_prob = sum(heatmap[r, c] for r, c in covered_nonzero)
    victims_pct = 100.0 * captured_prob / total_prob if total_prob > 0 else 0
    
    # Celdas únicas totales
    total_unique = len(all_explored)
    
    # Ticks usados
    ticks = len(history)
    
    return {
        'seed': seed,
        'coverage_pct': coverage_pct,
        'victims_pct': victims_pct,
        'total_unique_cells': total_unique,
        'covered_nonzero': len(covered_nonzero),
        'total_nonzero': total_nonzero,
        'ticks': ticks,
        'elapsed': elapsed,
        'per_drone': per_drone_cells,
    }

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "BASELINE"
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"  {NUM_DRONES} drones, {BUDGET/1000:.0f}km budget, seeds={SEEDS}")
    print(f"{'='*60}\n")
    
    results = []
    for seed in SEEDS:
        r = run_single(seed)
        results.append(r)
        print(f"  Seed {seed}: coverage={r['coverage_pct']:.1f}%  victims={r['victims_pct']:.1f}%  "
              f"cells={r['total_unique_cells']}  ticks={r['ticks']}  time={r['elapsed']:.1f}s")
        for did, cnt in sorted(r['per_drone'].items()):
            print(f"    {did}: {cnt} cells")
    
    avg_cov = np.mean([r['coverage_pct'] for r in results])
    avg_vic = np.mean([r['victims_pct'] for r in results])
    avg_cells = np.mean([r['total_unique_cells'] for r in results])
    
    print(f"\n  PROMEDIO: coverage={avg_cov:.1f}%  victims={avg_vic:.1f}%  unique_cells={avg_cells:.0f}")
    print(f"{'='*60}\n")
    
    return avg_cov, avg_vic

if __name__ == "__main__":
    main()
