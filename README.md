# SAREnv: UAV Search and Rescue Dataset and Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/namurproject/SAREnv)
[![Tests](https://github.com/fgs36-ua/SAREnv/actions/workflows/test.yml/badge.svg)](https://github.com/fgs36-ua/SAREnv/actions/workflows/test.yml)

SAREnv is an open-access dataset and evaluation framework designed to support research in UAV-based search and rescue (SAR) algorithms. This toolkit addresses the critical need for standardized datasets and benchmarks in wilderness SAR operations, enabling systematic evaluation and comparison of algorithmic approaches including coverage path planning, probabilistic search, and information-theoretic exploration.

This repository extends the original framework with a **multi-agent swarm module** (`sarenv.swarm`): a decentralised, bio-inspired coordination system for heterogeneous teams of UAV drones and ground robot dogs, with optional real-time 3D visualisation via [GAMA Platform](https://gama-platform.org/).

If you have issues cloning/downloading the repository with GitHub LFS (Large File Storage), the full repo with large files included can be found here: https://nextcloud.sdu.dk/index.php/s/pap6MJao5iXHWfw

## 🎯 Project Goals

Unmanned Aerial Vehicles (UAVs) play an increasingly vital role in wilderness search and rescue operations by enhancing situational awareness and extending the reach of human teams. However, the absence of standardized datasets and benchmarks has hindered systematic evaluation and comparison of UAV-based SAR algorithms. SAREnv bridges this gap by providing:

- **Realistic geospatial scenarios** across diverse terrain types
- **Synthetic victim locations** derived from statistical models of lost person behavior
- **Comprehensive evaluation metrics** for search trajectory assessment
- **Baseline planners** for reproducible algorithm comparisons
- **Extensible framework** for custom algorithm development

## 🌟 Key Features

### 📊 Dataset Generation

- **Multi-scale environments**: Small, medium, large, and extra-large search areas
- **Diverse terrain types**: Flat and mountainous environments
- **Climate variations**: Temperate and dry climate conditions
- **Realistic geospatial features**: Roads, water bodies, vegetation, structures, and terrain features extracted from OpenStreetMap

### 🎯 Lost Person Modeling

- Statistical models based on established lost person behavior research
- Probability heatmaps incorporating environmental factors
- Configurable victim location generation with terrain-aware distributions

### 🚁 Path Planning Algorithms

- **Spiral Coverage**: Efficient outward spiral search patterns
- **Concentric Circles**: Systematic circular search patterns
- **Pizza Zigzag**: Sector-based zigzag coverage
- **Greedy Search**: Probability-driven adaptive search
- **Extensible framework** for custom algorithm integration

### 📈 Evaluation Metrics

- **Coverage metrics**: Area coverage and search efficiency
- **Likelihood scores**: Probability-weighted path evaluation
- **Time-discounted scoring**: Temporal effectiveness assessment
- **Victim detection rates**: Success probability and timeliness analysis
- **Multi-drone coordination**: Support for collaborative search strategies

### 🤖 Multi-agent Swarm Extension

> Full technical documentation: **[sarenv/swarm/README.md](sarenv/swarm/README.md)**

- **Heterogeneous teams**: UAV drones + ground robot dogs with distinct mobility profiles
- **Decentralised coordination**: Virtual pheromone fields and epidemic gossip protocol (no central controller)
- **Bio-inspired exploration**: Attraction, repulsion, and anti-revisit mechanisms (Parunak 2002, Reynolds 1987)
- **Terrain-aware agents**: OSM-derived detection modifiers and traversability costs per agent type
- **GAMA Platform integration**: Optional real-time 3D visualisation over the terrain map
- **Reproducible benchmarks**: 60-scenario evaluation suite with published CSVs in `results/published/`

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/sarenv.git
cd sarenv

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional: GAMA Platform visualisation support
pip install -e ".[gama]"
```

### Download Pre-generated Dataset

The repository includes pre-generated datasets stored using Git LFS (Large File Storage). To download the data files needed to run the examples:

```bash
# Install Git LFS if not already installed
# On Ubuntu/Debian:
sudo apt-get install git-lfs

# On macOS with Homebrew:
brew install git-lfs

# On Windows, download from: https://git-lfs.github.io/

# Initialize Git LFS in the repository
git lfs install

# Download the dataset files
git lfs pull
```

**Note**: The dataset files are stored in the `sarenv_dataset/` directory and are required to run the examples. If you prefer to generate your own dataset, you can skip this step and use the dataset generation functionality described below.

### Basic Usage

#### 1. Generate Dataset (Optional)

If you prefer to generate your own dataset instead of using the pre-generated data:

```python
import sarenv

# Initialize data generator
generator = sarenv.DataGenerator()

# Generate dataset for different locations and sizes
generator.export_dataset()
```

**Note**: This step is optional if you've already downloaded the pre-generated dataset using Git LFS as described above.

#### 2. Load and Visualize Data

```python
import sarenv

# Load a dataset
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("large")

# Visualize the environment
from examples.02_load_and_visualize import visualize_heatmap, visualize_features
visualize_heatmap(item)
visualize_features(item)
```

#### 3. Generate Lost Person Locations

```python
import sarenv

# Load environment
loader = sarenv.DatasetLoader("sarenv_dataset")
item = loader.load_environment("medium")

# Generate victim locations
victim_generator = sarenv.LostPersonLocationGenerator(item)
locations = victim_generator.generate_locations(num_locations=100)
```

#### 4. Evaluate Search Algorithms

```python
import sarenv

# Initialize comparative evaluator
evaluator = sarenv.ComparativeEvaluator(
    dataset_directory="sarenv_dataset",
    evaluation_sizes=["medium", "large"],
    num_drones=5,
    num_lost_persons=50
)

# Run baseline evaluations
results = evaluator.run_baseline_evaluations()

# Plot comparative results
evaluator.plot_results(results)
```

### Multi-agent Swarm Quick Start

```python
from sarenv.core.loading import DatasetLoader
from sarenv.swarm import SwarmConfig, DroneConfig, SwarmSimulator, SwarmMetrics

# Load a SAR scenario
loader = DatasetLoader("sarenv_dataset")
item = loader.load_environment("medium")

# Configure a heterogeneous team: 3 drones + 2 robot dogs
config = SwarmConfig(
    num_drones=3,
    num_dogs=2,
    budget_per_agent=100_000,
)

# Run the decentralised simulation
sim = SwarmSimulator.from_dataset_item(item, config, seed=42)
sim.run()

# Evaluate
metrics = SwarmMetrics(sim)
summary = metrics.coverage_summary()
print(f"Coverage:    {summary['coverage_ratio']:.1%}")
print(f"Probability: {summary['probability_coverage_ratio']:.1%}")
```

## 📁 Example Scripts

The `examples/` directory is organised into three groups:

### 📂 `examples/01_dataset/` — Framework basics

#### `01_generate_sar_data.py`

**Purpose**: Generate custom SAR datasets for specific geographic regions  
**Usage**: Demonstrates how to create datasets from custom polygon areas using real geospatial data

```bash
python examples/01_dataset/01_generate_sar_data.py
```

- Creates datasets for custom geographic polygons
- Downloads and processes OpenStreetMap features
- Generates probability heatmaps for lost person locations
- Exports multi-scale environments (small to extra-large)

#### `02_load_and_visualize.py`

**Purpose**: Load existing datasets and create visualizations  
**Usage**: Shows how to load pre-generated datasets and create publication-quality plots

```bash
python examples/01_dataset/02_load_and_visualize.py
```

- Loads datasets from the `sarenv_dataset/` directory
- Creates heatmap visualizations with probability distributions
- Generates feature maps showing terrain, roads, water bodies, and vegetation
- Supports both basemap overlays and standalone visualizations

#### `03_generate_survivors.py`

**Purpose**: Generate realistic lost person locations  
**Usage**: Demonstrates statistical modeling of victim locations based on terrain features

```bash
python examples/01_dataset/03_generate_survivors.py
```

- Uses research-based behavioral models for lost person distributions
- Generates probabilistic victim locations considering terrain types
- Creates visualizations showing victim locations overlaid on terrain features
- Supports configurable numbers of victims for different scenario sizes

#### `04_evaluate_coverage_paths.py`

**Purpose**: Evaluate and compare path planning algorithms  
**Usage**: Run comparative analysis of built-in search algorithms on a single dataset

```bash
python examples/01_dataset/04_evaluate_coverage_paths.py
```

- Compares Spiral, Concentric, Pizza, and Greedy search algorithms
- Calculates comprehensive performance metrics (coverage, likelihood, detection rates)
- Generates comparison plots and performance charts
- Supports multi-drone scenarios with configurable team sizes

#### `05_evaluate_all_datasets.py`

**Purpose**: Large-scale evaluation across multiple datasets  
**Usage**: Systematic evaluation framework for algorithm benchmarking across many scenarios

```bash
python examples/01_dataset/05_evaluate_all_datasets.py --budget 300000 --num_drones 5
```

- Evaluates algorithms across multiple geographic regions
- Generates comprehensive CSV results for further analysis
- Supports custom algorithm integration for research purposes
- Command-line arguments for budget and drone configuration

#### `06_generate_comparative_coverage_video.py`

**Purpose**: Create animated videos showing algorithm performance  
**Usage**: Generate dynamic visualizations comparing multiple algorithms in real-time

```bash
python examples/01_dataset/06_generate_comparative_coverage_video.py
```

- Creates MP4 videos showing 4 algorithms side-by-side in 2×2 grid layout
- Real-time visualization of drone movement and path building
- Dynamic metrics graphs showing performance evolution over time
- Configurable video quality and frame rates for different use cases
- Efficient parallel processing for faster video generation

> **Note**: Scripts 02–06 require an existing dataset. Either download via Git LFS or run `01_generate_sar_data.py` first.

### 📂 `examples/02_swarm/` — Multi-agent swarm demos

| Script | Description |
|---|---|
| `01_run_swarm_simulation.py` | Basic swarm simulation: configure team, run, print metrics |
| `02_compare_vs_baselines.py` | Compare swarm coverage against centralised planners |
| `03_gama_visualization.py` | Live 3D visualisation in GAMA Platform via TCP |
| `04_coverage_video.py` | Generate coverage video for a swarm run |
| `05_comparative_video.py` | Side-by-side comparison video: swarm vs Pizza vs Greedy |

```bash
python examples/02_swarm/01_run_swarm_simulation.py --num_drones 3 --num_dogs 2
python examples/02_swarm/03_gama_visualization.py --scenario 1 --num-drones 5 --num-dogs 2
```

> **GAMA visualisation** requires [GAMA Platform 2025+](https://gama-platform.org/) installed separately and `pip install -e ".[gama]"`.

### 📂 `examples/03_benchmarks/` — Reproducibility

| Script | Description |
|---|---|
| `01_evaluate_60_scenarios.py` | Full 60-scenario benchmark used in the experimental evaluation |

```bash
python examples/03_benchmarks/01_evaluate_60_scenarios.py --num-drones 5 --budget 500000
```

> Precomputed results are available in `results/published/`.

## 📁 Repository Structure

```text
SAREnv/
├── sarenv/                     # Main package
│   ├── analytics/              # Path planning and evaluation (original framework)
│   │   ├── paths.py            # Coverage path algorithms
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── evaluator.py        # Comparative evaluation framework
│   ├── core/                   # Dataset generation and loading
│   ├── utils/                  # Geospatial and plotting utilities
│   └── swarm/                  # Multi-agent swarm extension → see sarenv/swarm/README.md
│       ├── config.py           # SwarmConfig, DroneConfig, RobotDogConfig
│       ├── agents.py           # DroneAgent, RobotDogAgent
│       ├── simulator.py        # Tick-based simulation loop
│       ├── knowledge.py        # Pheromone maps + gossip
│       ├── communication.py    # Epidemic gossip protocol
│       ├── metrics.py          # Coverage metrics + Gini coefficient
│       ├── terrain.py          # OSM-derived detection/traversability maps
│       ├── comparative.py      # Swarm vs centralised planners
│       ├── export.py           # Export to GAMA (CSV/GeoJSON)
│       └── gama_network_server.py  # TCP server for live GAMA visualisation
├── gama_model/                 # GAMA Platform model (3D visualisation)
│   └── models/sar_network.gaml
├── examples/
│   ├── 01_dataset/             # Dataset generation and baseline evaluation
│   ├── 02_swarm/               # Swarm simulation and GAMA visualisation
│   └── 03_benchmarks/          # 60-scenario reproducibility benchmark
├── tests/                      # Unit tests (90 passing)
├── results/
│   └── published/              # CSVs from the experimental evaluation
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── sarenv_dataset/             # Generated datasets (via Git LFS)
```

## 🔬 Research Applications

### Supported Algorithm Types

- **Coverage Path Planning**: Systematic area coverage strategies
- **Probabilistic Search**: Bayesian and heuristic search methods
- **Information-Theoretic Exploration**: Entropy-based search optimization
- **Multi-Agent Coordination**: Collaborative UAV search strategies

### Evaluation Dimensions

- **Spatial Coverage**: Area covered vs. time efficiency
- **Probability Optimization**: Likelihood-weighted search performance
- **Temporal Dynamics**: Time-sensitive victim detection modeling
- **Resource Utilization**: Multi-drone coordination effectiveness


## 🛠️ Custom Algorithm Integration

Add your own path planning algorithm:

```python
def custom_search_algorithm(center_x, center_y, max_radius, num_drones, **kwargs):
    """
    Custom search algorithm implementation.
    
    Args:
        center_x, center_y: Search center coordinates
        max_radius: Maximum search radius in meters
        num_drones: Number of UAVs
        **kwargs: Additional parameters (fov_deg, altitude, etc.)
    
    Returns:
        list[LineString]: Path for each drone
    """
    # Your algorithm implementation
    paths = []
    # ... algorithm logic ...
    return paths

# Register with evaluator
evaluator = sarenv.ComparativeEvaluator()
evaluator.path_generators['custom'] = custom_search_algorithm
```

## 📃 Experimental Results

The `results/published/` directory contains precomputed CSV data from the experimental evaluation:

| File | Description |
|---|---|
| `sarenv60_evaluation.csv` | Benchmark across 60 scenarios (180 runs: 3 drone configurations × 60 scenarios) |
| `exp_60scen_e1.csv` | E1 — Swarm vs centralised planners (Greedy, Pizza, Spiral) |
| `exp_60scen_e2.csv` | E2 — Heterogeneous team composition (drones vs robot dogs) |
| `exp_60scen_e3.csv` | E3 — Communication range (`max_hops`) sensitivity |
| `exp_60scen_e5.csv` | E5 — Resilience to agent failures |
| `exp5_resilience.csv` | E5 — Maigmó scenario resilience (final model) |
| `exp10_load_balance_comparative.csv` | E10 — Load balance across heterogeneous agents |

See `results/published/` for the full set of 13 CSV files.

## 📃 Publications

* Grøntved, K. A. R., Jarabo-Peñas, A., Reid, S., Rolland, E. G. A., Watson, M., Richards, A., Bullock, S., & Christensen, A. L. (2025). SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue Using UAVs. Drones, 9(9), 628. https://doi.org/10.3390/drones9090628

## 📝 Citation

If you use SAREnv in your research, please cite:

```bibtex
@article{sarenv2025,
  title={SAREnv: An Open-Source Dataset and Benchmark Tool for Informed Wilderness Search and Rescue using UAVs},
  author={Kasper Andreas Rømer Grøntved, Alejandro Jarabo-Peñas, Sid Reid, Edouard George Alain Rolland, Matthew Watson, Arthur Richards, Steve Bullock, and Anders Lyhne Christensen},
  journal={Drones},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is supported by; the Innovation Fund Denmark for the DIREC project (9142-00001B), the Independent Research Fund Denmark under grant 10.46540/4264-00105B (the NAMUR project), and by the WildDrone MSCA Doctoral Network funded by EU Horizon Europe under grant agreement no. 101071224


