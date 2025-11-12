#!/usr/bin/env python3
"""
RLID-NET Configuration Module
Contains all configuration constants and dataclasses for the RLID-NET system
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
import logging

# =============================================================================
# LID CONFIGURATION
# =============================================================================

# Updated LID lifecycle costs (KRW/m²)
# LID_COSTS = {
#     'Rain Garden': 15000.0,
#     'Green Roof': 173000.0,
#     'Permeable Pavement': 16000.0,
#     'Infiltration Trench': 30000.0,
#     'Bio-Retention Cell': 15000.0,
#     'Rain Barrel': 30000.0,
#     'Vegetative Swale': 9750.0,
#     'Rooftop Disconnection': 20000.0
# }
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
# LID types supported by SWMM
LID_TYPES = list(LID_COSTS.keys())

# Area percentages for LID placement (including negative for removal)
AREA_PERCENTAGES = [-2, -1, 0.5, 1, 2, 3]  # in percentage

# =============================================================================
# STATE SPACE DEFINITIONS
# =============================================================================

# Current implementation: Simplified state space (10 dimensions)
CURRENT_STATE_SPACE = {
    'current_lid_areas': 8,         # Each LID type's area in m² (8 LID types)
    'total_cost_normalized': 1,     # 0-1 normalized total cost 
    'runoff_reduction_rate': 1,     # 0-1 runoff reduction percentage
}
CURRENT_STATE_SIZE = sum(CURRENT_STATE_SPACE.values())

# Future reference: Detailed state space (14 dimensions)
DETAILED_STATE_SPACE = {
    'current_lid_areas': 8,         # Each LID type's area in m² (8 LID types)
    'total_cost': 1,                # Total lifecycle cost (normalized)
    'runoff_reduction': 1,          # Total runoff reduction in m³
    'runoff_reduction_rate': 1,     # Reduction percentage
    'subcatchment_area': 1,         # Total subcatchment area
    'impervious_area': 1,           # Impervious area available for LID
    'remaining_capacity': 1         # Available area for additional LID
}
DETAILED_STATE_SIZE = sum(DETAILED_STATE_SPACE.values())

# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================

NEURAL_NETWORK_CONFIG = {
    'input_size': CURRENT_STATE_SIZE,  # 10 dimensions
    'hidden_layers': [64, 32, 16],     # Standard DQN architecture
    'output_size': len(LID_TYPES) * len(AREA_PERCENTAGES),  # 8×6 = 48 actions
    'activation': 'ReLU',              # Standard ReLU activation
    'dropout_rate': 0.1
}

# =============================================================================
# LEARNING PARAMETERS
# =============================================================================

@dataclass
class RLConfig:
    """Reinforcement Learning Configuration"""
    
    # Training parameters
    num_episodes: int = 150
    max_steps_per_episode: int = int(os.environ.get('RLID_MAX_STEPS', 50))  # 환경변수에서 가져오거나 기본값 50
    learning_rate: float = 0.00005  # Reduced from 0.0001 for stability
    
    # Experience replay
    batch_size: int = 64  # Increased from 32 for stability
    memory_size: int = 10000 # Increased from 5000 for stability
    target_update_freq: int = 100  # Increased from 25 for stability
    
    # Exploration strategy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995  # Increased from 0.98 for stability
    # floor_at_epsilon: float = 0.85
    # T = int(num_episodes * floor_at_epsilon)
    # epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / T)  # 매 에피소드 후 ε ← max(eps_end, ε*decay)
    
    # Reward function weights
    # weight sum = 1.0
    reward_runoff_weight: float = float(os.environ.get('RLID_RUNOFF_WEIGHT', 0.7))
    reward_cost_weight: float = float(os.environ.get('RLID_COST_WEIGHT', 0.3))
    
    # Evaluation parameters
    eval_episodes: int = 5
    save_freq: int = 25


@dataclass
class ExperimentConfig:
    """Experiment Configuration"""
    
    # File paths
    base_inp_file: str = "inp_file/Example1.inp"
    output_dir: str = "results"
    log_level: str = "INFO"
    
    # Simulation parameters
    target_subcatchments: int = 1  # Single subcatchment focus
    
    # Area constraints
    max_lid_area_ratio: float = 0.5  # Maximum 50% of impervious area
    
    # Normalization factors (max_cost now calculated dynamically during runtime)
    max_expected_cost: float = 1000000.0  # Deprecated: now calculated as impervious_area × max(LID_costs) × 0.4
    max_expected_runoff: float = 1000.0   # 1000 m³ for normalization


@dataclass
class RLIDConfig:
    """Complete RLID-NET Configuration"""
    
    rl: RLConfig = field(default_factory=RLConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not os.path.exists(self.experiment.base_inp_file):
            raise FileNotFoundError(f"Base INP file not found: {self.experiment.base_inp_file}")
        
        # Create output directory if it doesn't exist
        Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)


def create_default_config(logger: logging.Logger) -> RLIDConfig:
    """Create default RLID-NET configuration"""
    logger.info("=" * 40)
    logger.info("Creating default configuration")
    logger.info("=" * 40)
    return RLIDConfig()


def validate_action_space():
    """Validate action space configuration"""
    total_actions = len(LID_TYPES) * len(AREA_PERCENTAGES)

    print("=" * 40)
    print("Action Space Configuration")
    print("=" * 40)
    print(f"  LID Types: {len(LID_TYPES)} ({LID_TYPES})")
    print(f"  Area Percentages: {len(AREA_PERCENTAGES)} ({AREA_PERCENTAGES})")
    print(f"  Total Actions: {total_actions}")
    print(f"  State Space Size: {CURRENT_STATE_SIZE} ({CURRENT_STATE_SPACE})")
    print(f"  Neural Network Output: {NEURAL_NETWORK_CONFIG['output_size']}")
    
    assert total_actions == NEURAL_NETWORK_CONFIG['output_size'], \
        f"Mismatch: Actions({total_actions}) != NN_Output({NEURAL_NETWORK_CONFIG['output_size']})"


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    validate_action_space()
    print("Configuration validation successful!") 