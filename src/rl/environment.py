#!/usr/bin/env python3
"""
RLID-NET Reinforcement Learning Environment
Handles state representation, action space, and reward calculation for LID optimization
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ..core.swmm_simulator import SWMMSimulator, SimulationResults, SubcatchmentData
from ..core.lid_manager import LIDManager, LIDPlacement
from ..utils.config import (
    LID_TYPES, AREA_PERCENTAGES, CURRENT_STATE_SIZE, LID_COSTS,
    RLConfig, ExperimentConfig
)


@dataclass
class EnvironmentState:
    """Current environment state representation"""
    lid_areas: List[float]  # 8 dimensions - area for each LID type (Rain Garden, Green Roof, Permeable Pavement, Infiltration Trench, Bio-Retention Cell, Rain Barrel, Vegetative Swale, Rooftop Disconnection)
    total_cost_normalized: float  # 0-1 normalized total cost
    runoff_reduction_rate: float  # 0-1 runoff reduction percentage
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numpy vector for RL agent"""
        return np.array(self.lid_areas + [self.total_cost_normalized, self.runoff_reduction_rate], 
                       dtype=np.float32)


@dataclass
class ActionResult:
    """Result of applying an action in the environment"""
    success: bool
    reward: float
    next_state: EnvironmentState
    simulation_results: Optional[SimulationResults]
    placement_info: Optional[Dict]
    message: str


class RLIDEnvironment:
    """
    RLID-NET Reinforcement Learning Environment
    
    Manages the RL interaction between agent and SWMM simulation environment
    """
    
    def __init__(self, 
                 base_inp_file: str,
                 rl_config: RLConfig,
                 exp_config: ExperimentConfig,
                 baseline_results: Optional[SimulationResults] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize RLID Environment
        
        Args:
            base_inp_file: Path to base SWMM INP file
            rl_config: RL configuration
            exp_config: Experiment configuration
            baseline_results: Pre-computed baseline results (optional)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.base_inp_file = base_inp_file
        self.rl_config = rl_config
        self.exp_config = exp_config
        
        # Initialize SWMM simulator
        self.simulator = SWMMSimulator(base_inp_file, self.logger)
        
        # Setup working directory
        self.work_dir = Path(exp_config.output_dir) / "temp_simulation"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.simulator.setup_working_directory()
        
        # Get baseline performance and target subcatchment
        if baseline_results is not None:
            # Use provided baseline results
            self.logger.info("[Using provided baseline results]")
            self.baseline_results = baseline_results
        else:
            # Run baseline simulation if not provided
            self.logger.info("[Analyzing baseline performance]")
            self.baseline_results = self.simulator.get_baseline_performance()
        
        if not self.baseline_results.simulation_successful:
            raise RuntimeError("Failed to establish baseline performance")
        
        self.target_subcatchment = self.simulator.find_highest_runoff_subcatchment(self.baseline_results)
        
        # Initialize LID manager
        self.lid_manager = LIDManager(base_inp_file, self.target_subcatchment, self.logger)
        self.lid_manager.setup_working_directory(self.work_dir)
        
        # Action space: 48 actions (8 LID types × 6 area percentages)
        self.action_space_size = len(LID_TYPES) * len(AREA_PERCENTAGES)
        
        # State space: 5 dimensions (simplified)
        self.state_space_size = CURRENT_STATE_SIZE
        
        # Normalization factors - Calculate dynamic max budget
        max_unit_cost = max(LID_COSTS.values())
        impervious_area = self.target_subcatchment.impervious_area_m2
        self.max_cost = impervious_area * max_unit_cost * 0.5  # Dynamic budget calculation
        self.baseline_runoff = self.baseline_results.total_runoff_m3
        
        self.logger.info(f"Dynamic max budget calculated: {self.max_cost/1000000:.2f}M KRW")
        self.logger.info(f"  - Impervious area: {impervious_area:.1f} m²")
        self.logger.info(f"  - Max unit cost: {max_unit_cost:.0f} KRW/m²")
        self.logger.info(f"  - Coverage factor: 0.5 (50%)")
        
        # Current state
        self.current_state = self._get_initial_state()
        
        self.logger.info(f"Environment initialized with {len(LID_TYPES)} LID types: {LID_TYPES}")
        self.logger.info(f"Action space: {self.action_space_size} actions")
        self.logger.info(f"State space: {self.state_space_size} dimensions")
        self.episode_step = 0
        self.max_steps_per_episode = rl_config.max_steps_per_episode
        
        self.logger.info("[RLID Environment initialized successfully]")
        self.logger.info(f"   Action space size: {self.action_space_size}")
        self.logger.info(f"   State space size: {self.state_space_size}")
        self.logger.info(f"   Target subcatchment: {self.target_subcatchment.id}")
        self.logger.info(f"   Baseline runoff: {self.baseline_runoff:.2f} m³")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial state vector
        """
        # Reset LID manager
        self.lid_manager.reset_all_lids()
        
        # Reset episode tracking
        self.episode_step = 0
        
        # Get initial state
        self.current_state = self._get_initial_state()
        
        self.logger.debug("Environment reset to initial state")
        
        return self.current_state.to_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step
        
        Args:
            action: Action index (0-47)
            
        Returns:
            (next_state, reward, done, info) tuple
        """
        self.episode_step += 1
        
        # Decode action
        lid_type, area_percentage = self._decode_action(action)
        
        # Check if episode is done before action
        done = self.episode_step >= self.max_steps_per_episode
        
        # Apply action
        action_result = self._apply_action(lid_type, area_percentage)
        
        # Update current state
        self.current_state = action_result.next_state
        
        # Prepare info dictionary
        info = {
            'action_success': action_result.success,
            'message': action_result.message,
            'placement_info': action_result.placement_info,
            'episode_step': self.episode_step,
            'lid_type': lid_type,
            'area_percentage': area_percentage,
            'total_cost': self.lid_manager.current_state.total_cost_krw,
            'runoff_reduction_m3': 0.0,  # Will be calculated if simulation successful
            'baseline_runoff': self.baseline_runoff
        }
        
        # Add simulation results to info if available
        if action_result.simulation_results and action_result.simulation_results.simulation_successful:
            current_runoff = action_result.simulation_results.total_runoff_m3
            runoff_reduction = self.baseline_runoff - current_runoff
            info['runoff_reduction_m3'] = runoff_reduction
            info['current_runoff_m3'] = current_runoff
        
        return (self.current_state.to_vector(), 
                action_result.reward, 
                done, 
                info)
    
    def _decode_action(self, action: int) -> Tuple[str, float]:
        """
        Decode action index to LID type and area percentage
        
        Args:
            action: Action index (0-47)
            
        Returns:
            (lid_type, area_percentage) tuple
        """
        if not 0 <= action < self.action_space_size:
            raise ValueError(f"Invalid action: {action}. Must be in range [0, {self.action_space_size})")
        
        lid_index = action // len(AREA_PERCENTAGES)
        area_index = action % len(AREA_PERCENTAGES)
        
        lid_type = LID_TYPES[lid_index]
        area_percentage = AREA_PERCENTAGES[area_index]
        
        return lid_type, area_percentage
    
    def _encode_action(self, lid_type: str, area_percentage: float) -> int:
        """
        Encode LID type and area percentage to action index
        
        Args:
            lid_type: LID type name
            area_percentage: Area percentage
            
        Returns:
            Action index
        """
        lid_index = LID_TYPES.index(lid_type)
        area_index = AREA_PERCENTAGES.index(area_percentage)
        
        return lid_index * len(AREA_PERCENTAGES) + area_index
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions based on current state constraints
        
        Returns:
            List of valid action indices
        """
        valid_actions = []
        
        for action in range(self.action_space_size):
            lid_type, area_percentage = self._decode_action(action)
            can_apply, _ = self.lid_manager.can_apply_action(lid_type, area_percentage)
            
            if can_apply:
                valid_actions.append(action)
        
        return valid_actions
    
    def _apply_action(self, lid_type: str, area_percentage: float) -> ActionResult:
        """
        Apply LID action and calculate reward
        
        Args:
            lid_type: LID type to apply
            area_percentage: Area percentage (negative for removal)
            
        Returns:
            ActionResult with reward and next state
        """
        # Apply LID action
        success, message, placement = self.lid_manager.apply_lid_action(lid_type, area_percentage)
        
        if not success:
            # Failed action - small negative reward
            return ActionResult(
                success=False,
                reward=-1.0,
                next_state=self._get_current_state(),
                simulation_results=None,
                placement_info=None,
                message=f"Action failed: {message}"
            )
        
        # Run simulation with new LID configuration
        inp_file = self.lid_manager.get_current_inp_file()
        if inp_file:
            self.simulator.update_inp_file_path(str(inp_file))
        
        sim_results = self.simulator.run_simulation(
            show_progress=False, 
            quiet_logger=True,
            description=f"Step {self.episode_step}"
        )
        
        # Calculate reward
        reward = self._calculate_reward(sim_results)
        
        # Get next state
        next_state = self._get_current_state()
        
        # Prepare placement info
        placement_info = None
        if placement:
            placement_info = {
                'lid_type': placement.lid_type,
                'area_m2': placement.area_m2,
                'area_percentage': placement.area_percentage,
                'cost_krw': placement.cost_krw
            }
        
        return ActionResult(
            success=True,
            reward=reward,
            next_state=next_state,
            simulation_results=sim_results,
            placement_info=placement_info,
            message=message
        )
    
    def _calculate_reward(self, sim_results: SimulationResults) -> float:
        """
        Calculate reward based on simulation results using new formula:
        Reward = WF × (Runoff Reduction Rate) + WC × (Cost Saving Rate)
        
        Args:
            sim_results: SWMM simulation results
            
        Returns:
            Calculated reward value in [0, 1] range
        """
        if not sim_results.simulation_successful:
            return 0.0  # Minimum reward for failed simulation
        
        # Calculate runoff reduction rate
        baseline = max(self.baseline_runoff, 1e-6)  # Safety against division by zero
        current_runoff = sim_results.total_runoff_m3
        runoff_reduction = max(0.0, baseline - current_runoff)
        runoff_reduction_rate = runoff_reduction / baseline
        
        # Calculate cost saving rate
        current_cost = self.lid_manager.current_state.total_cost_krw
        budget = max(self.max_cost, 1e-6)  # Safety against division by zero
        cost_saving_rate = max(0.0, (budget - current_cost) / budget)
        
        # Apply weights and calculate final reward
        WF = self.rl_config.reward_runoff_weight  # Weight for runoff reduction
        WC = self.rl_config.reward_cost_weight    # Weight for cost saving
        
        reward = WF * 3.3737 * runoff_reduction_rate + WC * cost_saving_rate
        
        # Ensure reward is in [0, 1] range
        reward = max(0.0, min(1.0, reward))
        
        return reward
    
    def _get_initial_state(self) -> EnvironmentState:
        """Get initial environment state (no LIDs)"""
        return EnvironmentState(
            lid_areas=[0.0] * len(LID_TYPES),  # 8 LID types: Rain Garden, Green Roof, Permeable Pavement, Infiltration Trench, Bio-Retention Cell, Rain Barrel, Vegetative Swale, Rooftop Disconnection
            total_cost_normalized=0.0,
            runoff_reduction_rate=0.0
        )
    
    def _get_current_state(self) -> EnvironmentState:
        """Get current environment state based on LID placements"""
        # Get LID areas (raw values in m²)
        lid_areas_raw = self.lid_manager.get_current_state_vector()
        
        # Normalize LID areas by impervious area (convert to ratios 0-1)
        impervious_area = max(self.target_subcatchment.impervious_area_m2, 1e-6)
        lid_areas = [area / impervious_area for area in lid_areas_raw]
        
        # Normalize cost
        current_cost = self.lid_manager.current_state.total_cost_krw
        cost_normalized = min(1.0, current_cost / self.max_cost)
        
        # Estimate runoff reduction rate (simplified - could run simulation for accuracy)
        # For now, use a simple heuristic based on total LID area
        total_lid_area = sum(lid_areas_raw)
        available_area = self.target_subcatchment.impervious_area_m2
        
        if available_area > 0:
            area_coverage = min(1.0, total_lid_area / available_area)
            # Simple linear approximation: more LID area = more runoff reduction
            estimated_reduction_rate = min(0.8, area_coverage * 0.6)  # Max 80% reduction
        else:
            estimated_reduction_rate = 0.0
        
        return EnvironmentState(
            lid_areas=lid_areas,  # Now normalized ratios instead of raw m² values
            total_cost_normalized=cost_normalized,
            runoff_reduction_rate=estimated_reduction_rate
        )
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed information about current state"""
        lid_summary = self.lid_manager.get_current_state_summary()
        
        return {
            'current_state_vector': self.current_state.to_vector().tolist(),
            'episode_step': self.episode_step,
            'max_steps': self.max_steps_per_episode,
            'lid_placements': lid_summary['placements'],
            'total_cost_krw': lid_summary['total_cost_krw'],
            'total_area_m2': lid_summary['total_area_m2'],
            'area_utilization': lid_summary['area_utilization_ratio'],
            'valid_actions': len(self.get_valid_actions()),
            'baseline_runoff_m3': self.baseline_runoff
        }
    
    def get_next_action_preview(self, action: int) -> Dict[str, Any]:
        """
        Get preview of what the next action would do (without executing it)
        
        Args:
            action: Action index to preview
            
        Returns:
            Dictionary with action preview information
        """
        lid_type, area_percentage = self._decode_action(action)
        can_apply, reason = self.lid_manager.can_apply_action(lid_type, area_percentage)
        
        preview_info = {
            'action_index': action,
            'lid_type': lid_type,
            'area_percentage': area_percentage,
            'can_apply': can_apply,
            'reason': reason
        }
        
        if can_apply:
            # Calculate what would happen
            area_m2 = abs(self.target_subcatchment.area_m2 * area_percentage / 100)
            cost_change = area_m2 * self.lid_manager.LID_COSTS.get(lid_type, 0.0)
            
            if area_percentage < 0:
                cost_change = -cost_change  # Cost reduction for removal
            
            preview_info.update({
                'area_change_m2': area_m2 if area_percentage > 0 else -area_m2,
                'cost_change_krw': cost_change,
                'action_type': 'add' if area_percentage > 0 else 'remove'
            })
        
        return preview_info
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        if hasattr(self, 'simulator'):
            self.simulator.cleanup()
        
        self.logger.info("Environment cleanup completed")

    def _calculate_lid_coverage(self, subcatchment_id: str = None, exclude_lid_type: str = None) -> float:
        """
        Calculate total LID coverage percentage for validation
        
        Args:
            subcatchment_id: Target subcatchment ID (defaults to current target)
            exclude_lid_type: LID type to exclude from calculation
            
        Returns:
            Total LID coverage as percentage of impervious area
        """
        if subcatchment_id is None:
            subcatchment_id = self.target_subcatchment.id
        
        # Calculate impervious area
        impervious_area_m2 = self.target_subcatchment.impervious_area_m2
        
        if impervious_area_m2 <= 0:
            self.logger.warning(f"Subcatchment {subcatchment_id} has no impervious area")
            return float('inf')  # Indicates impossible to place LID
        
        # Calculate total LID area
        total_lid_area_m2 = 0.0
        current_state = self.lid_manager.get_current_state()
        
        for lid_type, placements in current_state.placements.items():
            if exclude_lid_type and lid_type == exclude_lid_type:
                continue
            total_lid_area_m2 += placements.get('area_m2', 0.0)
        
        # Calculate coverage as percentage of impervious area
        coverage_percentage = (total_lid_area_m2 / impervious_area_m2) * 100
        
        self.logger.debug(f"LID coverage calculation: {total_lid_area_m2:.1f}m² / {impervious_area_m2:.1f}m² = {coverage_percentage:.1f}%")
        
        return coverage_percentage
    
    def _validate_lid_placement(self, lid_type: str, area_change_m2: float) -> Tuple[bool, str, float]:
        """
        Validate LID placement against area constraints
        
        Args:
            lid_type: LID type to be placed
            area_change_m2: Area change in square meters (positive for addition, negative for removal)
            
        Returns:
            (is_valid, message, adjusted_area_change)
        """
        # Get current coverage excluding the LID type being modified
        current_coverage = self._calculate_lid_coverage(exclude_lid_type=lid_type)
        
        # Calculate new area for this LID type
        current_state = self.lid_manager.get_current_state()
        current_lid_area = current_state.placements.get(lid_type, {}).get('area_m2', 0.0)
        new_lid_area = max(0.0, current_lid_area + area_change_m2)
        
        # Calculate new coverage percentage for this LID
        impervious_area_m2 = self.target_subcatchment.impervious_area_m2
        new_lid_coverage = (new_lid_area / impervious_area_m2) * 100
        
        # Calculate total new coverage
        new_total_coverage = current_coverage + new_lid_coverage
        
        # Apply safety margin (95% instead of 100%)
        max_coverage = 95.0
        
        if new_total_coverage > max_coverage:
            # Calculate maximum allowable area for this LID
            max_allowable_coverage = max_coverage - current_coverage
            max_allowable_area = (max_allowable_coverage / 100) * impervious_area_m2
            adjusted_area_change = max_allowable_area - current_lid_area
            
            if adjusted_area_change <= 0:
                return False, f"Cannot place {lid_type}: total coverage would exceed {max_coverage}%", 0.0
            
            exceeded_amount = new_total_coverage - max_coverage
            message = f"Area adjusted to prevent exceeding {max_coverage}% coverage (would exceed by {exceeded_amount:.1f}%)"
            return True, message, adjusted_area_change
        
        return True, "Placement validated successfully", area_change_m2


def test_environment():
    """Test the RLID Environment"""
    import sys
    sys.path.append('.')
    
    from src.utils.config import RLConfig, ExperimentConfig
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test configuration
    rl_config = RLConfig()
    exp_config = ExperimentConfig()
    
    try:
        # Initialize environment
        env = RLIDEnvironment(
            base_inp_file="inp_file/Example1.inp",
            rl_config=rl_config,
            exp_config=exp_config
        )
        
        print("Testing RLID Environment:")
        print(f"   Action space size: {env.action_space_size}")
        print(f"   State space size: {env.state_space_size}")
        
        # Test reset
        initial_state = env.reset()
        print(f"   Initial state shape: {initial_state.shape}")
        
        # Test valid actions
        valid_actions = env.get_valid_actions()
        print(f"   Valid actions: {len(valid_actions)}/{env.action_space_size}")
        
        # Test action preview
        if valid_actions:
            preview = env.get_next_action_preview(valid_actions[0])
            print(f"   First valid action: {preview['lid_type']} ({preview['area_percentage']}%)")
        
        # Test step
        if valid_actions:
            next_state, reward, done, info = env.step(valid_actions[0])
            print(f"   Step result: reward={reward:.2f}, done={done}")
            print(f"   Action: {info['lid_type']} ({info['area_percentage']}%)")
        
        print("Environment test completed!")
        
    except Exception as e:
        print(f"Environment test failed: {str(e)}")
        raise
    
    finally:
        if 'env' in locals():
            env.cleanup()


if __name__ == "__main__":
    test_environment() 