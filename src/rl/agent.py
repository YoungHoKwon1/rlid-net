#!/usr/bin/env python3
"""
RLID-NET DQN Agent Implementation
Deep Q-Network agent for LID placement optimization
"""

import os
import random
import logging
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .environment import RLIDEnvironment
from ..utils.config import RLConfig, NEURAL_NETWORK_CONFIG


@dataclass
class TrainingMetrics:
    """Container for training metrics and progress tracking"""
    episode_rewards: List[float]
    episode_losses: List[float]
    episode_costs: List[float]
    episode_runoff_reductions: List[float]
    epsilon_values: List[float]
    
    def __post_init__(self):
        """Initialize empty lists if not provided"""
        for attr in ['episode_rewards', 'episode_losses', 'episode_costs', 
                    'episode_runoff_reductions', 'epsilon_values']:
            if getattr(self, attr) is None:
                setattr(self, attr, [])


class DQN(nn.Module):
    """
    Deep Q-Network for RLID-NET
    
    Implements the proposed compact neural network architecture
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.1):
        """
        Initialize DQN
        
        Args:
            input_size: State space size (10 dimensions)
            hidden_sizes: List of hidden layer sizes [64, 32, 16]
            output_size: Action space size (48 actions)
            dropout_rate: Dropout rate for regularization
        """
        super(DQN, self).__init__()
        
        # Build network layers
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Output layer (no activation for Q-values)
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class ReplayMemory:
    """Experience replay memory for DQN training"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay memory
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch from memory"""
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Get current memory size"""
        return len(self.memory)


class RLIDAgent:
    """
    DQN Agent for RLID-NET LID Placement Optimization
    
    Implements Deep Q-Network with experience replay and target network
    """
    
    def __init__(self, 
                 environment: RLIDEnvironment,
                 config: RLConfig,
                 logger: Optional[logging.Logger] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize RLID Agent
        
        Args:
            environment: RLID environment instance
            config: RL configuration parameters
            logger: Logger instance
            device: PyTorch device (CPU/GPU)
        """
        self.env = environment
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network configuration
        self.state_size = environment.state_space_size
        self.action_size = environment.action_space_size
        
        # Initialize networks
        self.q_network = DQN(
            input_size=self.state_size,
            hidden_sizes=NEURAL_NETWORK_CONFIG['hidden_layers'],
            output_size=self.action_size,
            dropout_rate=NEURAL_NETWORK_CONFIG['dropout_rate']
        ).to(self.device)
        
        self.target_network = DQN(
            input_size=self.state_size,
            hidden_sizes=NEURAL_NETWORK_CONFIG['hidden_layers'],
            output_size=self.action_size,
            dropout_rate=NEURAL_NETWORK_CONFIG['dropout_rate']
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = ReplayMemory(config.memory_size)
        
        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # Training tracking
        self.training_metrics = TrainingMetrics([], [], [], [], [])
        self.step_count = 0
        self.target_update_count = 0
        
        self.logger.info("RLID Agent initialized successfully")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   State size: {self.state_size}")
        self.logger.info(f"   Action size: {self.action_size}")
        self.logger.info(f"   Network architecture: {NEURAL_NETWORK_CONFIG['hidden_layers']}")
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state vector
            valid_actions: List of valid actions (constraint enforcement)
            
        Returns:
            Selected action index
        """
        # Get valid actions if not provided
        if valid_actions is None:
            valid_actions = self.env.get_valid_actions()
        
        if len(valid_actions) == 0:
            # No valid actions - return random action (should not happen)
            self.logger.warning("No valid actions available! Returning random action.")
            return random.randint(0, self.action_size - 1)
        
        # Epsilon-greedy exploration
        if random.random() > self.epsilon:
            # Exploit: choose best action among valid actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Mask invalid actions with very negative values
                masked_q_values = q_values.clone()
                for i in range(self.action_size):
                    if i not in valid_actions:
                        masked_q_values[0, i] = float('-inf')
                
                action = masked_q_values.argmax().item()
        else:
            # Explore: choose random valid action
            action = random.choice(valid_actions)
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Training loss value
        """
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        current_q_values = torch.clamp(current_q_values, 0.0, 1.2) # Clip current_q_values(안정화용)
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)  # 0.99 is discount factor
            target_q_values = torch.clamp(target_q_values, 0.0, 1.2)

        
        # Compute loss using Huber Loss for stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability (increased from 1.0 to 2.0)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=2.0)
        
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_count += 1
            self.logger.debug(f"Target network updated (#{self.target_update_count})")
        
        return loss.item()
    
    def train(self, num_episodes: int, save_freq: int = 25) -> TrainingMetrics:
        """
        Train the agent
        
        Args:
            num_episodes: Number of training episodes
            save_freq: Frequency to save model checkpoints
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Starting DQN training for {num_episodes} episodes")
        self.logger.info("-" * 60)
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            episode_steps = 0
            
            # Episode loop
            while True:
                # Get valid actions and select action
                valid_actions = self.env.get_valid_actions()
                action = self.select_action(state, valid_actions)
                
                # Get action preview for logging
                action_preview = self.env.get_next_action_preview(action)
                
                # Take action first
                next_state, reward, done, info = self.env.step(action)
                
                # Log current state and next action with actual results
                self._log_step_details_with_result(episode, episode_steps, action_preview, info)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train if enough experiences
                if len(self.memory) >= self.config.batch_size:
                    loss = self.train_step()
                    episode_loss += loss
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Log step result
                self._log_step_result(info, reward, episode_loss / max(1, episode_steps))
                
                if done:
                    break
            
            # Update exploration
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store episode metrics
            self.training_metrics.episode_rewards.append(episode_reward)
            self.training_metrics.episode_losses.append(episode_loss / max(1, episode_steps))
            self.training_metrics.episode_costs.append(info.get('total_cost', 0.0))
            self.training_metrics.episode_runoff_reductions.append(info.get('runoff_reduction_m3', 0.0))
            self.training_metrics.epsilon_values.append(self.epsilon)
            
            # Show episode progress
            self._show_episode_progress(episode, num_episodes, episode_reward, info)
            
            # Save checkpoint
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(episode + 1)
        
        self.logger.info("\nTraining completed!")
        
        return self.training_metrics
    
    def _log_step_details(self, episode: int, step: int, action_preview: Dict[str, Any]):
        """Log detailed step information (deprecated - use _log_step_details_with_result)"""
        self._log_step_details_with_result(episode, step, action_preview, {})
    
    def _log_step_details_with_result(self, episode: int, step: int, action_preview: Dict[str, Any], result_info: Dict[str, Any]):
        """Log detailed step information with actual results after action execution"""
        state_info = self.env.get_state_info()
        
        self.logger.info(f"\n--- Episode {episode+1}, Step {step+1} ---")
        
        # Current state
        self.logger.info("Current State:")
        if state_info['lid_placements']:
            for placement in state_info['lid_placements']:
                self.logger.info(
                    f"  Subcatchment {self.env.target_subcatchment.id}: "
                    f"{placement['lid_type']} {placement['area_m2']:.1f}m² "
                    f"({placement['area_percentage']:.1f}%, {placement['cost_krw']/1000:.1f}k KRW)"
                )
        else:
            self.logger.info(f"  Subcatchment {self.env.target_subcatchment.id}: No LID placements")
        
        # Current totals with actual results
        actual_reduction = result_info.get('runoff_reduction_m3', state_info.get('runoff_reduction_m3', 0.0))
        actual_cost = result_info.get('total_cost', state_info['total_cost_krw'])
        actual_area_util = result_info.get('area_utilization', state_info['area_utilization'])
        
        self.logger.info(
            f"  Total Runoff Reduction: {actual_reduction:.1f}m³ "
            f"| Total Cost: {actual_cost/1000000:.2f}M KRW "
            f"| Area Utilization: {actual_area_util*100:.1f}%"
        )
        
        # Next action
        self.logger.info("Next Action:")
        if action_preview['can_apply']:
            action_type = "Adding" if action_preview['area_percentage'] > 0 else "Removing"
            area_change = abs(action_preview.get('area_change_m2', 0.0))
            cost_change = action_preview.get('cost_change_krw', 0.0)
            
            self.logger.info(
                f"  {action_type} {action_preview['lid_type']} "
                f"{area_change:.1f}m² ({action_preview['area_percentage']}%) "
                f"to Subcatchment {self.env.target_subcatchment.id} "
                f"(Cost change: {cost_change/1000:.1f}k KRW)"
            )
        else:
            self.logger.info(f"  Action blocked: {action_preview['reason']}")
    
    def _log_step_result(self, info: Dict[str, Any], reward: float, avg_loss: float):
        """Log step execution result"""
        if info['action_success']:
            runoff_reduction = info.get('runoff_reduction_m3', 0.0)
            runoff_percentage = (runoff_reduction / info['baseline_runoff']) * 100 if info['baseline_runoff'] > 0 else 0.0
            
            self.logger.info(
                f"  Result: Reward={reward:.2f} | Loss={avg_loss:.4f} | "
                f"Runoff Reduction: {runoff_reduction:.1f}m³ ({runoff_percentage:.1f}%)"
            )
        else:
            self.logger.info(f"  Result: Action failed - {info['message']}")
    
    def _show_episode_progress(self, episode: int, total_episodes: int, 
                              episode_reward: float, info: Dict[str, Any]):
        """Show episode progress with progress bar"""
        progress = (episode + 1) / total_episodes
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        
        total_cost = info.get('total_cost', 0.0) / 1000000  # Convert to millions
        runoff_reduction = info.get('runoff_reduction_m3', 0.0)
        
        print(f'\rEpisode [{episode+1:3d}/{total_episodes}] '
              f'[{bar}] {progress:.1%} | '
              f'Reward: {episode_reward:6.1f} | '
              f'Cost: {total_cost:.2f}M KRW | '
              f'Reduction: {runoff_reduction:.1f}m³ | '
              f'Epsilon: {self.epsilon:.3f}')
        
        if episode == total_episodes - 1:
            print()  # New line at the end
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.env.exp_config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        
        checkpoint_path = checkpoint_dir / f"rlid_agent_episode_{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_metrics = checkpoint['training_metrics']
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"   Resumed from episode: {checkpoint['episode']}")
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate trained agent
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        eval_rewards = []
        eval_costs = []
        eval_runoff_reductions = []
        
        # Save current epsilon and set to 0 (no exploration)
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        try:
            for episode in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0.0
                
                while True:
                    valid_actions = self.env.get_valid_actions()
                    action = self.select_action(state, valid_actions)
                    
                    next_state, reward, done, info = self.env.step(action)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                eval_rewards.append(episode_reward)
                eval_costs.append(info.get('total_cost', 0.0))
                eval_runoff_reductions.append(info.get('runoff_reduction_m3', 0.0))
                
                self.logger.info(
                    f"Eval Episode {episode+1}: Reward={episode_reward:.2f}, "
                    f"Cost={info.get('total_cost', 0.0)/1000:.1f}k KRW, "
                    f"Reduction={info.get('runoff_reduction_m3', 0.0):.1f}m³"
                )
        
        finally:
            # Restore original epsilon
            self.epsilon = original_epsilon
        
        evaluation_results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_cost': np.mean(eval_costs),
            'mean_runoff_reduction': np.mean(eval_runoff_reductions),
            'all_rewards': eval_rewards,
            'all_costs': eval_costs,
            'all_runoff_reductions': eval_runoff_reductions
        }
        
        self.logger.info("Evaluation Results:")
        self.logger.info(f"   Mean Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
        self.logger.info(f"   Mean Cost: {evaluation_results['mean_cost']/1000:.1f}k KRW")
        self.logger.info(f"   Mean Runoff Reduction: {evaluation_results['mean_runoff_reduction']:.1f}m³")
        
        return evaluation_results


if __name__ == "__main__":
    # Test the agent
    import sys
    sys.path.append('.')
    
    from src.rl.environment import RLIDEnvironment
    from src.utils.config import RLConfig, ExperimentConfig
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test configuration
    rl_config = RLConfig()
    rl_config.num_episodes = 5  # Short test
    exp_config = ExperimentConfig()
    
    try:
        # Initialize environment and agent
        env = RLIDEnvironment("inp_file/Example1.inp", rl_config, exp_config)
        agent = RLIDAgent(env, rl_config)
        
        print("Testing RLID Agent:")
        print(f"   Network architecture: {NEURAL_NETWORK_CONFIG['hidden_layers']}")
        
        # Test action selection
        state = env.reset()
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        print(f"   Selected action: {action} (from {len(valid_actions)} valid)")
        
        # Test short training
        print("   Running short training test...")
        metrics = agent.train(num_episodes=3)
        
        print(f"   Training completed! Final reward: {metrics.episode_rewards[-1]:.2f}")
        
        print("Agent test completed!")
        
    except Exception as e:
        print(f"Agent test failed: {str(e)}")
        raise
    
    finally:
        if 'env' in locals():
            env.cleanup() 