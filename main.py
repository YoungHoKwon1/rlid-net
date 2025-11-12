#!/usr/bin/env python3
"""
RLID-NET Main Execution Script
Enhanced urban flood reduction through optimal LID placement using reinforcement learning
"""

import os
import sys
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import create_default_config, validate_action_space
from src.core.swmm_simulator import SWMMSimulator, analyze_example_inp
from src.rl.environment import RLIDEnvironment
from src.rl.agent import RLIDAgent
from src.utils.visualization import RLIDVisualizer


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging system
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    # Create output directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rlid_net_{timestamp}.log"
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('RLID-NET')
    
    print("=" * 40)
    print("RLID-NET System Starting")
    print("=" * 40)

    logger.info("[RLID-NET System Starting]")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log level: {log_level}")
    print()
    
    return logger


def validate_system_requirements(logger: logging.Logger, inp_file: str) -> bool:
    """
    Validate system requirements and dependencies
    
    Args:
        logger: Logger instance
        inp_file: Path to the input SWMM file
        
    Returns:
        True if all requirements are met
    """
    logger.info("=" * 40)
    logger.info("Validating system requirements")
    logger.info("=" * 40)
    
    try:
        # Check required libraries
        required_imports = [
            ('torch', 'PyTorch'),
            ('pyswmm', 'PySWMM'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('matplotlib', 'Matplotlib'),
            ('openpyxl', 'OpenPyXL')
        ]
        
        for module, name in required_imports:
            try:
                __import__(module)
                logger.info(f"{name} available")
            except ImportError:
                logger.error(f"{name} not found - install with: pip install {module}")
                return False
        print()
        # Validate action space configuration
        validate_action_space()
        logger.info("Action space configuration valid")
        
        # Check if INP file exists
        if not os.path.exists(inp_file):
            logger.error(f"{inp_file} not found.")
            return False
        
        logger.info(f"INP file found at: {inp_file}")
        logger.info("All system requirements validated successfully")
        logger.info("=" * 40)
        print()
        return True
        
    except Exception as e:
        logger.error(f"System validation failed: {str(e)}")
        return False


def analyze_input_data(logger: logging.Logger, inp_file: str) -> dict:
    """
    Analyze input SWMM model and return baseline results
    
    Args:
        logger: Logger instance
        inp_file: Path to the input SWMM file
        
    Returns:
        Dictionary with analysis results including baseline data
    """
    logger.info("=" * 40)
    logger.info(f"Analyzing input SWMM model: {inp_file}")
    logger.info("=" * 40)
    
    try:
        analysis = analyze_example_inp(inp_file)
        
        if "error" in analysis:
            logger.error(f"INP analysis failed: {analysis['error']}")
            return {}
        
        logger.info("[SWMM Model Analysis Results]")
        logger.info(f"   Total Subcatchments: {analysis['total_subcatchments']}")
        logger.info(f"   Total Baseline Runoff: {analysis['total_runoff_m3']:.2f} m³")
        
        target = analysis['target_subcatchment']
        logger.info(f"Target Subcatchment Selected: {target['id']}")
        logger.info(f"   Total Area: {target['area_m2']:.1f} m²")
        logger.info(f"   Impervious Area: {target['impervious_area_m2']:.1f} m²")
        logger.info(f"   Impervious Percentage: {target['percent_impervious']:.1f}%")
        logger.info(f"   Baseline Runoff: {target['runoff_m3']:.2f} m³")
        
        logger.info("Input data analysis completed successfully")
        logger.info("=" * 40)
        print()
        return analysis
        
    except Exception as e:
        logger.error(f"Input data analysis failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {}


def run_training_session(config, analysis_results: dict, logger: logging.Logger, inp_file: str) -> tuple:
    """
    Execute complete RLID-NET training session
    
    Args:
        config: RLID configuration object
        analysis_results: Pre-computed analysis results including baseline data
        logger: Logger instance
        inp_file: Path to the input SWMM file
        
    Returns:
        (agent, training_metrics, evaluation_results, visualizer) tuple
    """
    logger.info("=" * 40)
    logger.info("Starting RLID-NET training session")
    logger.info("=" * 40)
    
    # Create baseline results from analysis
    baseline_results = None
    if analysis_results and "baseline_results" in analysis_results:
        baseline_results = analysis_results["baseline_results"]
        logger.info("[Using pre-computed baseline results]")
    
    # Initialize environment
    logger.info("[Initializing RL environment]")
    env = RLIDEnvironment(
        base_inp_file=inp_file,
        rl_config=config.rl,
        exp_config=config.experiment,
        baseline_results=baseline_results,
        logger=logger
    )
    print()
    # Initialize agent
    logger.info("[Initializing DQN agent]")
    agent = RLIDAgent(
        environment=env,
        config=config.rl,
        logger=logger
    )
    print()
    # Initialize visualizer
    logger.info("[Initializing visualization system]")
    visualizer = RLIDVisualizer(
        output_dir=config.experiment.output_dir,
        logger=logger
    )
    print()
    try:
        # Run training
        logger.info("=" * 40)
        logger.info("Starting training")
        logger.info("=" * 40)
        
        logger.info(f"[Starting training: {config.rl.num_episodes} episodes]")
        training_metrics = agent.train(
            num_episodes=config.rl.num_episodes,
            save_freq=config.rl.save_freq
        )
        
        logger.info("Training completed successfully!")
        
        # Run evaluation
        logger.info("[Running final evaluation]")
        evaluation_results = agent.evaluate(num_episodes=config.rl.eval_episodes)
        logger.info("Evaluation completed!")
        
        return agent, training_metrics, evaluation_results, visualizer, env
        
    except Exception as e:
        logger.error(f"Training session failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Cleanup environment
        env.cleanup()


def generate_final_reports(agent, training_metrics, evaluation_results, 
                         visualizer, env, logger: logging.Logger):
    """
    Generate comprehensive final reports and visualizations
    
    Args:
        agent: Trained DQN agent
        training_metrics: Training metrics
        evaluation_results: Evaluation results
        visualizer: Visualization system
        env: RL environment
        logger: Logger instance
    """
    logger.info("=" * 40)
    logger.info("Generating final reports and visualizations")
    logger.info("=" * 40)
    
    try:
        # Get final LID placements
        final_placements = []
        lid_summary = env.lid_manager.get_current_state_summary()
        final_placements = lid_summary['placements']
        
        # Get baseline runoff
        baseline_runoff = env.baseline_runoff
        
        # Generate all visualization reports
        generated_files = visualizer.generate_all_reports(
            training_metrics=training_metrics,
            baseline_runoff=baseline_runoff,
            final_lid_placements=final_placements,
            evaluation_results=evaluation_results
        )
        
        # Log results summary
        logger.info("Final Results Summary:")
        logger.info("=" * 40)
        
        # Training results
        final_reward = training_metrics.episode_rewards[-1]
        best_reward = max(training_metrics.episode_rewards)
        final_cost = training_metrics.episode_costs[-1] if training_metrics.episode_costs else 0
        final_reduction = training_metrics.episode_runoff_reductions[-1] if training_metrics.episode_runoff_reductions else 0
        
        logger.info(f"Training Performance:")
        logger.info(f"   Final Reward: {final_reward:.2f}")
        logger.info(f"   Best Reward: {best_reward:.2f}")
        logger.info(f"   Final Episode Cost: {final_cost/1000000:.2f} M KRW")
        logger.info(f"   Final Runoff Reduction: {final_reduction:.1f} m³")
        
        # LID placement results
        logger.info(f"LID Placement Results:")
        if final_placements:
            total_area = sum(p['area_m2'] for p in final_placements)
            total_cost = sum(p['cost_krw'] for p in final_placements)
            
            logger.info(f"   Total LID Types Used: {len(final_placements)}")
            logger.info(f"   Total LID Area: {total_area:.1f} m²")
            logger.info(f"   Total LID Cost: {total_cost/1000000:.2f} M KRW")
            
            for placement in final_placements:
                logger.info(f"   - {placement['lid_type']}: {placement['area_m2']:.1f}m² "
                          f"({placement['area_percentage']:.1f}%, {placement['cost_krw']/1000:.1f}k KRW)")
        else:
            logger.info("   No LID placements in final configuration")
        
        # Evaluation results
        if evaluation_results:
            logger.info(f"Evaluation Performance:")
            logger.info(f"   Mean Evaluation Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
            logger.info(f"   Mean Cost: {evaluation_results['mean_cost']/1000:.1f}k KRW")
            logger.info(f"   Mean Runoff Reduction: {evaluation_results['mean_runoff_reduction']:.1f} m³")
        
        # Generated files
        logger.info(f"Generated Report Files:")
        for file_type, file_path in generated_files.items():
            logger.info(f"   {file_type}: {Path(file_path).name}")
        
        logger.info("Final reports generated successfully!")
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='RLID-NET: Reinforcement Learning for LID Placement Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--episodes', type=int, default=150, 
                       help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate system requirements and analyze input data')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal episodes (5 episodes)')

    parser.add_argument('--inp-file', type=str, default='inp_file/Example1.inp',
                       help='Input SWMM INP file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    try:
        # Validate system requirements
        if not validate_system_requirements(logger, args.inp_file):
            logger.error("System validation failed. Please check requirements.")
            sys.exit(1)
        
        # Analyze input data
        analysis = analyze_input_data(logger, args.inp_file)
        if not analysis:
            logger.error("Input data analysis failed. Cannot proceed.")
            sys.exit(1)
        
        # If validation-only mode, exit here
        if args.validate_only:
            logger.info("Validation completed successfully. Exiting.")
            return
        
        # Load configuration
        config = create_default_config(logger)
        
        # Override configuration with command line arguments
        config.experiment.base_inp_file = args.inp_file  # Pass inp_file to config
        if args.episodes:
            config.rl.num_episodes = args.episodes
        if args.output_dir:
            config.experiment.output_dir = args.output_dir
        
        # Quick test mode
        if args.quick_test:
            logger.info("****Running in quick test mode****")
            config.rl.num_episodes = 5
            config.rl.max_steps_per_episode = 10
            config.rl.save_freq = 5
        
        # Log final configuration
        logger.info("=" * 40)
        logger.info("Final Configuration")
        logger.info("=" * 40)
        logger.info(f"   Input INP file: {config.experiment.base_inp_file}")
        logger.info(f"   Episodes: {config.rl.num_episodes}")
        logger.info(f"   Max steps per episode: {config.rl.max_steps_per_episode}")
        logger.info(f"   Learning rate: {config.rl.learning_rate}")
        logger.info(f"   Batch size: {config.rl.batch_size}")
        logger.info(f"   Output directory: {config.experiment.output_dir}")
        logger.info("=" * 40)
        
        # Run training session
        agent, training_metrics, evaluation_results, visualizer, env = run_training_session(config, analysis, logger, args.inp_file)
        
        # Generate final reports
        generate_final_reports(agent, training_metrics, evaluation_results, visualizer, env, logger)
        
        # Success message
        logger.info("=" * 40)
        logger.info("RLID-NET execution completed successfully!")
        logger.info(f"Results saved to: {config.experiment.output_dir}")
        logger.info("=" * 40)
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error("Fatal error during execution:")
        logger.error(f"   Error: {str(e)}")
        logger.error("   Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 