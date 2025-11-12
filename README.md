# RLID-NET Project

## Project Overview

### System Name
**RLID-NET (Reinforcement Learning - Low Impact Development Network)**

### Objective
- Optimized allocation of LID(Low Impact Development) for urban flood reduction using RL(Reinforcement Learning)
- SWMM(Storm Water Management Model) based runoff simulation
- Stratigic approach to find LCC(Life Cycle Cost)-runoff multio-bjective LID allocation
---

## System Architecture

### 1. Entire System Structure
```
RLID-NET/
├── src/
│   ├── core/                    # core simulation engine
│   │   ├── swmm_simulator.py   # SWMM simulation management
│   │   └── lid_manager.py      # LID allocation % parameter management
│   ├── rl/                     # RL module
│   │   ├── environment.py      # RL environment manegement
│   │   └── agent.py           # DQN agent implementaion
│   ├── utils/                  # Utility functions
│   │   ├── config.py          # System configuration
│   │   └── visualization.py   # Result visualization
│   └── __init__.py
├── inp_file/                   # SWMM input files
├── results/                    # results
├── main.py                     # main
├── run_batch_training.py       # Batch training
└── test_system.py             # System test
```

### 2. Core Component

#### A. SWMM Simulation (`src/core/swmm_simulator.py`)
- **Purpose**: Urban drainage system simulation and data management
- **Dependencies**: Based on PySWMM library
- **Key Features**: 
  - Baseline performance analysis and caching
  - Runoff calculation after LID application
  - Progress display supported simulation
  - Automatic temporary directory management
- **Core Classes**:
  - `SubcatchmentData`: Subcatchment information data structure
  - `SimulationResults`: Simulation results container
- **Key Methods**:
  - `get_baseline_performance()`: Baseline performance measurement (cached)
  - `run_simulation()`: LID-applied simulation (with progress support)
  - `find_highest_runoff_subcatchment()`: Automatic identification of highest runoff area
  - `get_total_runoff_reduction()`: Runoff reduction calculation

#### B. LID Management System (`src/core/lid_manager.py`)
- **Purpose**: LID facility placement, constraint validation, INP file modification management
- **Core Features**:
  - Real-time constraint validation (`can_apply_action`)
  - Automatic INP file generation and modification
  - LID placement/removal operation support
  - SWMM-compatible parameter management
  - Maximum area constraint: 95% of impervious area (with safety margin)
- **Key Classes**:
  - `LIDType`: Enumeration of 8 LID types (includes SWMM codes)
  - `LIDPlacement`: LID placement information data structure
  - `LIDState`: Current LID state summary
- **Supported LID Types** (all 8 types enabled):
  - **Rain Garden** (RG): 270,000 KRW/m²
  - **Green Roof** (GR): 45,000 KRW/m²
  - **Permeable Pavement** (PP): 110,000 KRW/m²
  - **Infiltration Trench** (IT): 200,000 KRW/m²
  - **Bio-Retention Cell** (BC): 120,000 KRW/m²
  - **Rain Barrel** (RB): 2,000 KRW/m²
  - **Vegetative Swale** (VS): 19,000 KRW/m²
  - **Rooftop Disconnection** (RD): 500 KRW/m²

#### C. Reinforcement Learning Environment (`src/rl/environment.py`)
- **State Space**: 10-dimensional (`EnvironmentState`)
  - LID area ratio (8 dimensions): Normalized LID area relative to impervious area (0-1 range)
  - Normalized total cost (1 dimension): 0-1 range relative to dynamic budget
  - Runoff reduction rate (1 dimension): Estimated 0-1 range (limited to maximum 80%)
- **Action Space**: 48 actions (8 LID types × 6 area ratios)
  - Area ratios: [-2%, -1%, 0.5%, 1%, 2%, 3%]
  - Real-time valid action filtering support
- **Reward Function**: 
  - `WF × 3.3737 × Runoff Reduction Rate + WC × Cost Saving Rate`
  - Reward values are clamped to [0, 1] range
  - Environment variable weights: `RLID_RUNOFF_WEIGHT`, `RLID_COST_WEIGHT`
  - **Dynamic Budget Calculation**: `Impervious Area × Max Unit Cost × 0.5`

#### D. DQN Agent (`src/rl/agent.py`)
- **Neural Network Architecture** (`DQN` class): 
  - Input layer: 10 nodes (state space)
  - Hidden layers: [64, 32, 16] nodes (3-layer structure)
  - Output layer: 48 nodes (action space)
  - **Activation Function**: ReLU
  - Dropout: 0.1
  - Weight initialization: Xavier Uniform (bias=0.01)
- **Training Stabilization Techniques**:
  - Experience Replay Memory (10,000 capacity)
  - Target Network (updated every 100 steps)
  - Gradient Clipping (max_norm=2.0)
  - Q-value Clamping (0.0-1.2 range) - applied to both current Q and target Q
- **Learning Parameters**:
  - Learning rate: 0.00005 (Adam optimizer)
  - Batch size: 64
  - Discount factor: 0.99
  - Epsilon decay: 1.0 → 0.02 (0.995 decay rate)

#### E. Visualization and Reporting (`src/utils/visualization.py`)
- **Purpose**: Comprehensive training analysis and professional report generation
- **Core Class**: `RLIDVisualizer`
- **Advanced Visualization Features**:
  - Moving average trend analysis
  - Multi-axis charts (simultaneous display of reward-runoff reduction)
  - Log-scale loss visualization
  - Pie chart cost analysis
- **Excel Reports** (based on openpyxl):
  - Multi-tab configuration (placement/summary/cost reference)
  - Professional styling (header formatting, automatic column width)
  - Automatic TOTAL row generation
  - Cost efficiency metric calculation
- **Generated Files**: 5 main reports
  - `reward_trend.png`: Episode-by-episode reward trend
  - `loss_trend.png`: Training loss trend (log scale)
  - `training_metrics.xlsx`: Complete training data
  - `lid_placement_summary.xlsx`: Final LID placement results
  - `baseline_comparison.png`: 4-panel comprehensive performance comparison

---

## ⚙️ System Configuration Status

### 1. Current Configuration (`src/utils/config.py`)
```python
# LID configuration (all 8 types enabled)
LID_COSTS = {
    'Rain Garden': 270000.0,
    'Green Roof': 45000.0,
    'Permeable Pavement': 110000.0,
    'Infiltration Trench': 200000.0,
    'Bio-Retention Cell': 120000.0,
    'Rain Barrel': 2000.0,
    'Vegetative Swale': 19000.0,
    'Rooftop Disconnection': 500.0
}

# State space (10 dimensions)
CURRENT_STATE_SPACE = {
    'current_lid_areas': 8,         # Area ratio for each of 8 LID types
    'total_cost_normalized': 1,     # 0-1 normalized total cost
    'runoff_reduction_rate': 1,     # 0-1 runoff reduction ratio
}
CURRENT_STATE_SIZE = 10             # Total state space size

# Neural network configuration
NEURAL_NETWORK_CONFIG = {
    'input_size': 10,
    'hidden_layers': [64, 32, 16],
    'output_size': 48,              # 8 LID types × 6 area ratios
    'activation': 'ReLU',
    'dropout_rate': 0.1
}

# Reinforcement learning configuration
class RLConfig:
    num_episodes: int = 150
    max_steps_per_episode: int = int(os.environ.get('RLID_MAX_STEPS', 50))
    learning_rate: float = 0.00005
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995
    # Reward function weights (configurable via environment variables)
    reward_runoff_weight: float = float(os.environ.get('RLID_RUNOFF_WEIGHT', 0.7))
    reward_cost_weight: float = float(os.environ.get('RLID_COST_WEIGHT', 0.3))
    eval_episodes: int = 5
    save_freq: int = 25
```

### 2. Experiment Configuration
```python
class ExperimentConfig:
    base_inp_file: str = "inp_file/Example1.inp"
    output_dir: str = "results"
    log_level: str = "INFO"
    target_subcatchments: int = 1  # Single subcatchment focus
    max_lid_area_ratio: float = 0.5  # Maximum 50% of impervious area
    # Normalization factors (dynamically calculated)
    max_expected_cost: float = 1000000.0  # Deprecated: now calculated dynamically (impervious_area × max_unit_cost × 0.5)
    max_expected_runoff: float = 1000.0   # Expected maximum runoff
```

### 3. Batch Training Configuration (`run_batch_training.py`)
```python
# Various experiment configuration combinations
configurations = [
    # (episodes, steps, runoff_weight, cost_weight, suffix)
    (1000, 25, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 40, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 50, 0.7, 0.3, "_20250809_1_8lid_w7030"),
    (1000, 40, 0.6, 0.4, "_20250809_1_8lid_w6040"),
    (1000, 40, 0.8, 0.2, "_20250809_1_8lid_w8020"),
    # ... experiments with various combinations
]
```

---

## Testing and Validation

### 1. System Validation Script (`test_system.py`)
- Module import tests
- Configuration system validation
- SWMM analysis tests
- Quick training tests
- Visualization system tests

### 2. Latest Test Results
```
Test Results Summary:
├── Module Imports: PASS
├── Configuration: PASS
├── SWMM Analysis: PASS
├── Quick Training: PASS
└── Visualization: PASS

Overall: 5/5 tests passed
```

### 3. Key Performance Metrics (as of August 2024)
**Baseline Performance:**
- Baseline runoff: 676.20 m³ (36-hour simulation)
- Peak runoff: 0.142 m³/s
- Target subcatchment: ID 5 (15 ha, maximum runoff contributing area)

**System Configuration:**
- State vector size: (10,) - 8 LID areas + 2 normalization metrics
- Action space: 48 actions (8 LID types × 6 area change ratios)
- Supported LID types: All 8 types enabled

**Representative Performance Results:**
- Maximum runoff reduction rate: 27.5% (185.9 m³ reduction)
- Cost efficiency: 0.0428 m³/M KRW
- Total LID placement area: 36,080 m² (93.2% of impervious area)
- Total installation cost: 4.86 M KRW

---

## Usage

### 1. Main Command Line Arguments

| Argument | Description | Default | Example |
| --- | --- | --- | --- |
| `--inp_file` | Specifies the path to the SWMM `.inp` file to use. | `inp_file/Example1.inp` | `--inp_file inp_file/inputfile.inp` |
| `--episodes` | Specifies the number of episodes to train. (`main.py` only) | `150` | `--episodes 500` |
| `--output-dir` | Specifies the directory where result files will be saved. | `./results` | `--output-dir ./my_results` |
| `--quick-test` | Runs a quick test with 5 episodes. (`main.py` only) | `False` | `--quick-test` |

### 2. Single Experiment Execution
```bash
# Run with default settings
python main.py

# Run with 500 episodes
python main.py --episodes 500

# Run quick test with specific INP file
python main.py --quick-test --inp_file inp_file/inputfile.inp
```

### 3. Batch Experiment Execution
```bash
# Run batch experiment with default INP file
python run_batch_training.py

# Run batch experiment with specific INP file
python run_batch_training.py --inp_file inp_file/inputfile.inp

# Note: Detailed batch experiment settings (weights, etc.) must be modified directly in the configurations list within run_batch_training.py file.
```

### 4. System Validation
```bash
python test_system.py
```

### 5. Configuration Validation
```bash
python -c "from src.utils.config import validate_action_space; validate_action_space()"
```
---

## Development History

### Version History
- **v2.4** (November 2025): Cloud deployment implementation
- **v2.3** (October 2025): Simulation results analysis
- **v2.2** (August 2025): Added clipping to prevent loss divergence
- **v2.1** (July 2025): SWMM constraint implementation for LID installation
- **v2.0** (May 2025): Initial system implementation
- **v1.0** (April 2025): SWMM integration and I/O control testing

## Development Environment and Dependencies

### 1. Dependency Installation
All Python libraries required for the project are specified in the `requirements.txt` file. You can install them all at once using the following command:

```bash
pip install -r requirements.txt
```

### 2. Required Libraries
```python
# Core dependencies
torch                    # Deep learning
numpy                    # Numerical computation
pyswmm                   # SWMM simulation
matplotlib               # Visualization
logging                  # Logging
pathlib                  # File system
dataclasses              # Data structures
```

### 3. System Requirements
- Python 3.7+
- Windows/Linux compatible
- SWMM 5.1+ installation required