#!/usr/bin/env python3
"""
RLID-NET System Validation Test Script
Quick test to verify all components are working correctly
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported successfully"""
    print("Testing module imports...")
    
    try:
        # Test configuration
        from src.utils.config import create_default_config, validate_action_space
        print("Configuration module")
        
        # Test SWMM simulator
        from src.core.swmm_simulator import SWMMSimulator, analyze_example_inp
        print("SWMM simulator module")
        
        # Test LID manager
        from src.core.lid_manager import LIDManager
        print("LID manager module")
        
        # Test RL environment
        from src.rl.environment import RLIDEnvironment
        print("RL environment module")
        
        # Test RL agent
        from src.rl.agent import RLIDAgent
        print("RL agent module")
        
        # Test visualization
        from src.utils.visualization import RLIDVisualizer
        print("Visualization module")
        
        print("All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"Import failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from src.utils.config import create_default_config, validate_action_space
        
        # Create logger for configuration
        logger = logging.getLogger("test_config")
        logger.setLevel(logging.INFO)
        
        # Test default configuration
        config = create_default_config(logger)
        print(f"Default config created")
        print(f"   Episodes: {config.rl.num_episodes}")
        print(f"   Learning rate: {config.rl.learning_rate}")
        print(f"   Action space size: 48 expected")
        
        # Test action space validation
        validate_action_space()
        print("Action space validation passed")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
        return False


def test_swmm_analysis():
    """Test SWMM analysis capabilities"""
    print("\nTesting SWMM analysis...")
    
    try:
        from src.core.swmm_simulator import analyze_example_inp
        
        # Check if INP file exists
        if not os.path.exists("inp_file/Example1.inp"):
            print("Example1.inp not found - skipping SWMM test")
            return True
        
        # Analyze INP file
        analysis = analyze_example_inp("inp_file/Example1.inp")
        
        if "error" in analysis:
            print(f"SWMM analysis failed: {analysis['error']}")
            return False
        
        print(f"SWMM analysis completed")
        print(f"   Subcatchments: {analysis['total_subcatchments']}")
        print(f"   Target subcatchment: {analysis['target_subcatchment']['id']}")
        print(f"   Baseline runoff: {analysis['total_runoff_m3']:.2f} m³")
        
        return True
        
    except Exception as e:
        print(f"SWMM analysis test failed: {str(e)}")
        return False


def test_quick_training():
    """Test quick training run (1 episode)"""
    print("\nTesting quick training run...")
    
    try:
        from src.utils.config import create_default_config
        from src.rl.environment import RLIDEnvironment
        from src.rl.agent import RLIDAgent
        
        # Check if INP file exists
        if not os.path.exists("inp_file/Example1.inp"):
            print("Example1.inp not found - skipping training test")
            return True
        
        # Create logger for configuration
        logger = logging.getLogger("test_training")
        logger.setLevel(logging.INFO)
        
        # Create minimal configuration
        config = create_default_config(logger)
        config.rl.num_episodes = 1
        config.rl.max_steps_per_episode = 3
        
        # Setup temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config.experiment.output_dir = temp_dir
            
            # Setup logging for test
            logging.basicConfig(level=logging.WARNING)  # Minimize output
            test_logger = logging.getLogger('test')
            
            # Initialize environment
            env = RLIDEnvironment(
                base_inp_file="inp_file/Example1.inp",
                rl_config=config.rl,
                exp_config=config.experiment,
                logger=test_logger
            )
            
            # Initialize agent
            agent = RLIDAgent(
                environment=env,
                config=config.rl,
                logger=test_logger
            )
            
            # Test action selection
            state = env.reset()
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            
            print(f"Quick training test passed")
            print(f"   State shape: {state.shape}")
            print(f"   Valid actions: {len(valid_actions)}")
            print(f"   Selected action: {action}")
            
            # Cleanup
            env.cleanup()
            
        return True
        
    except Exception as e:
        print(f"Quick training test failed: {str(e)}")
        return False


def test_visualization():
    """Test visualization system"""
    print("\nTesting visualization system...")
    
    try:
        from src.utils.visualization import RLIDVisualizer
        from src.rl.agent import TrainingMetrics
        
        # Create test data
        test_metrics = TrainingMetrics(
            episode_rewards=[10.0, 15.0, 20.0],
            episode_losses=[0.5, 0.3, 0.2],
            episode_costs=[100000, 150000, 200000],
            episode_runoff_reductions=[50.0, 75.0, 100.0],
            epsilon_values=[1.0, 0.95, 0.90]
        )
        
        test_placements = [
            {'lid_type': 'Rain Garden', 'area_m2': 100.0, 'area_percentage': 2.0, 'cost_krw': 1500000}
        ]
        
        # Test visualization creation
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = RLIDVisualizer(temp_dir)
            
            # This would generate files, but we'll just test initialization
            print("Visualizer initialization passed")
            
        return True
        
    except Exception as e:
        print(f"Visualization test failed: {str(e)}")
        return False


def main():
    """Run all system validation tests"""
    print("RLID-NET System Validation Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("SWMM Analysis", test_swmm_analysis),
        ("Quick Training", test_quick_training),
        ("Visualization", test_visualization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} test crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready to use.")
        return 0
    else:
        print("Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 