#!/usr/bin/env python3
"""
SWMM Simulator Module for RLID-NET
Enhanced SWMM simulation manager with improved error handling and data management
"""

import os
import sys
import tempfile
import shutil
import logging
import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import pyswmm
    from pyswmm import Simulation, Subcatchments, SystemStats, Nodes, Links
except ImportError:
    print("PySWMM not installed. Install with: pip install pyswmm")
    raise


@dataclass
class SubcatchmentData:
    """Data structure for subcatchment information"""
    id: str
    area_hectares: float
    area_m2: float
    runoff_m3: float
    runoff_rate_lps: float = 0.0
    percent_impervious: float = 0.0
    impervious_area_m2: float = 0.0
    
    def __post_init__(self):
        """Calculate derived values"""
        if self.area_m2 == 0:
            self.area_m2 = self.area_hectares * 10000
        
        # Calculate impervious area
        if self.impervious_area_m2 == 0 and self.percent_impervious > 0:
            self.impervious_area_m2 = self.area_m2 * (self.percent_impervious / 100.0)


@dataclass
class SimulationResults:
    """Container for simulation results"""
    subcatchments: List[SubcatchmentData]
    total_runoff_m3: float
    simulation_time_seconds: float
    flow_units: str
    system_units: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    simulation_successful: bool = True
    error_message: Optional[str] = None


class SWMMSimulator:
    """
    Enhanced SWMM simulation manager with improved error handling and data management
    """
    
    def __init__(self, inp_file_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize SWMM simulator
        
        Args:
            inp_file_path: Path to SWMM input file
            logger: Optional logger instance
        """
        self.base_inp_file = Path(inp_file_path)
        if not self.base_inp_file.exists():
            raise FileNotFoundError(f"SWMM input file not found: {inp_file_path}")
            
        self.logger = logger or self._setup_logger()
        self.results: Optional[SimulationResults] = None
        
        # Working directory for temporary files
        self.temp_dir = None
        self.current_inp_file = None
        
        # Baseline performance (measured once)
        self._baseline_results: Optional[SimulationResults] = None
        
        self.logger.info(f"SWMMSimulator initialized for: {self.base_inp_file}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _show_progress_bar(self, progress: int, current_time, final: bool = False):
        """Show progress bar for simulation"""
        bar_length = 30
        filled_length = int(bar_length * progress // 100)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        
        if final:
            print(f"\rSimulation: [{bar}] {progress}% - Completed!", flush=True)
        else:
            time_str = str(current_time).split()[1] if len(str(current_time).split()) > 1 else str(current_time)
            print(f"\rSimulation: [{bar}] {progress}% - {time_str}", end='', flush=True)
    
    def setup_working_directory(self) -> Path:
        """
        Create temporary working directory for SWMM simulations
        
        Returns:
            Path to temporary directory
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="rlid_net_")
            self.logger.info(f"Created working directory: {self.temp_dir}")
        
        return Path(self.temp_dir)
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up working directory: {self.temp_dir}")
    
    def get_baseline_performance(self) -> SimulationResults:
        """
        Run baseline simulation (no LID) and store results
        
        Returns:
            Baseline simulation results
        """
        if self._baseline_results is None:
            self.logger.info("Running baseline simulation (no LID)...")
            self._baseline_results = self.run_simulation(show_progress=True, description="Baseline")
            
            if self._baseline_results.simulation_successful:
                baseline_runoff = self._baseline_results.total_runoff_m3
                self.logger.info(f"Baseline total runoff: {baseline_runoff:.2f} m³")
            else:
                self.logger.error("Baseline simulation failed")
        
        return self._baseline_results
    
    def find_highest_runoff_subcatchment(self, baseline_results: Optional[SimulationResults] = None) -> SubcatchmentData:
        """
        Identify the subcatchment with highest runoff for LID placement
        
        Args:
            baseline_results: Pre-computed baseline results (optional)
            
        Returns:
            SubcatchmentData for the highest runoff subcatchment
        """
        if baseline_results is not None:
            # Use provided baseline results
            baseline = baseline_results
        else:
            # Run baseline simulation if not provided
            baseline = self.get_baseline_performance()
        
        if not baseline.simulation_successful:
            raise RuntimeError("Cannot find highest runoff subcatchment: baseline simulation failed")
        
        # Check if we have subcatchment data
        if not baseline.subcatchments:
            raise RuntimeError("No subcatchment data found in baseline simulation")
        
        # Data is already sorted by runoff (descending) in run_simulation
        target_subcatchment = baseline.subcatchments[0]
        
        self.logger.info(f"Target subcatchment identified: {target_subcatchment.id}")
        self.logger.info(f"   - Area: {target_subcatchment.area_m2:.1f} m²")
        self.logger.info(f"   - Impervious: {target_subcatchment.percent_impervious:.1f}%")
        self.logger.info(f"   - Runoff: {target_subcatchment.runoff_m3:.2f} m³")
        
        return target_subcatchment
    
    def run_simulation(self, 
                      show_progress: bool = True, 
                      quiet_logger: bool = False,
                      description: str = "") -> SimulationResults:
        """
        Run SWMM simulation and extract results
        
        Args:
            show_progress: Whether to show progress bar
            quiet_logger: Whether to suppress detailed logging
            description: Description for logging purposes
            
        Returns:
            SimulationResults object containing all extracted data
        """
        if not self.current_inp_file:
            # Use base INP file if no modified version exists
            self.current_inp_file = self.base_inp_file
        
        start_time = datetime.datetime.now()
        
        try:
            with Simulation(str(self.current_inp_file)) as sim:
                if not quiet_logger:
                    self.logger.info(f"Starting SWMM simulation: {description}")
                
                # Extract simulation metadata
                flow_units = sim.flow_units
                system_units = sim.system_units
                sim_start_time = sim.start_time
                sim_end_time = sim.end_time
                
                if not quiet_logger:
                    self.logger.info(f"Flow Units: {flow_units}")
                    self.logger.info(f"System Units: {system_units}")
                    self.logger.info(f"Simulation Period: {sim_start_time} to {sim_end_time}")
                
                # Initialize data containers
                subcatchments_data = []
                
                # Get subcatchments, nodes, and links
                subcatchments = Subcatchments(sim)
                nodes = Nodes(sim)
                links = Links(sim)
                
                if not quiet_logger:
                    self.logger.info(f"Found {len(subcatchments)} subcatchments, "
                                   f"{len(nodes)} nodes, {len(links)} links")
                
                # Run simulation with optional progress tracking
                if show_progress:
                    step_count = 0
                    last_progress = -1
                    
                    for step in sim:
                        if step_count % 100 == 0:
                            progress = int(sim.percent_complete * 100)
                            
                            # Only update progress bar when progress changes significantly
                            if progress != last_progress and progress % 10 == 0:
                                self._show_progress_bar(progress, sim.current_time)
                                last_progress = progress
                        step_count += 1
                    
                    # Ensure 100% is shown at the end
                    self._show_progress_bar(100, sim.end_time, final=True)
                else:
                    # Run silently for faster execution during training
                    for step in sim:
                        pass
                
                # Extract subcatchment results
                total_runoff = 0.0
                for subcatchment in subcatchments:
                    area_hectares = subcatchment.area
                    area_m2 = area_hectares * 10000
                    runoff_m3 = subcatchment.statistics.get("runoff", 0.0)
                    # Convert from fraction (0-1) to percentage (0-100)
                    percent_impervious = subcatchment.percent_impervious * 100
                    total_runoff += runoff_m3
                    
                    subc_data = SubcatchmentData(
                        id=subcatchment.subcatchmentid,
                        area_hectares=area_hectares,
                        area_m2=area_m2,
                        runoff_m3=runoff_m3,
                        percent_impervious=percent_impervious
                    )
                    subcatchments_data.append(subc_data)
                    
                    if not quiet_logger:
                        self.logger.info(f" subcatchment: {subc_data.id} | runoff: {runoff_m3:.2f} m³ | area: {area_m2:.1f} m² | impervious: {percent_impervious:.1f}%")
                
                # Sort subcatchments by runoff (descending)
                subcatchments_data.sort(key=lambda x: x.runoff_m3, reverse=True)
                
                end_time = datetime.datetime.now()
                simulation_time = (end_time - start_time).total_seconds()
                
                # Create results object
                self.results = SimulationResults(
                    subcatchments=subcatchments_data,
                    total_runoff_m3=total_runoff,
                    simulation_time_seconds=simulation_time,
                    flow_units=flow_units,
                    system_units=system_units,
                    start_time=sim_start_time,
                    end_time=sim_end_time,
                    simulation_successful=True
                )
                
                if not quiet_logger:
                    self.logger.info(f"Simulation completed successfully in {simulation_time:.2f}s")
                    self.logger.info(f"Total runoff: {total_runoff:.2f} m³")
                    self.logger.info(f"Subcatchments found: {len(subcatchments_data)}")
                    if subcatchments_data:
                        self.logger.info(f"Target subcatchment: {subcatchments_data[0].id} - {subcatchments_data[0].runoff_m3:.2f} m³")
                
                return self.results
                
        except Exception as e:
            error_msg = f"SWMM simulation failed: {str(e)}"
            self.logger.error(f"{error_msg}")
            
            return SimulationResults(
                subcatchments=[],
                total_runoff_m3=0.0,
                simulation_time_seconds=0.0,
                flow_units="",
                system_units="",
                start_time=datetime.datetime.now(),
                end_time=datetime.datetime.now(),
                simulation_successful=False,
                error_message=error_msg
            )
    
    def get_top_subcatchments(self, n: int) -> List[SubcatchmentData]:
        """
        Get top N subcatchments by runoff volume
        
        Args:
            n: Number of subcatchments to return
            
        Returns:
            List of top N SubcatchmentData objects
        """
        if self.results is None:
            raise ValueError("No simulation results available. Run simulation first.")
        
        # Data is already sorted by runoff (descending) in run_simulation
        top_subcatchments = self.results.subcatchments[:n]
        
        return top_subcatchments
    
    def get_subcatchment_by_id(self, subcatchment_id: str) -> Optional[SubcatchmentData]:
        """Get subcatchment data by ID"""
        if self.results is None:
            return None
        
        # Create a dictionary for O(1) lookup if not already cached
        if not hasattr(self, '_subcatchment_dict'):
            self._subcatchment_dict = {subc.id: subc for subc in self.results.subcatchments}
        
        return self._subcatchment_dict.get(subcatchment_id)
    
    def get_total_runoff_reduction(self, baseline_runoff: float) -> Tuple[float, float]:
        """
        Calculate runoff reduction compared to baseline
        
        Args:
            baseline_runoff: Baseline total runoff volume
            
        Returns:
            Tuple of (absolute_reduction, percentage_reduction)
        """
        if self.results is None:
            return 0.0, 0.0
        
        current_runoff = self.results.total_runoff_m3
        absolute_reduction = baseline_runoff - current_runoff
        percentage_reduction = (absolute_reduction / baseline_runoff) * 100 if baseline_runoff > 0 else 0.0
        
        return absolute_reduction, percentage_reduction
    
    def update_inp_file_path(self, new_inp_path: str):
        """
        Update the current INP file path (used when LID modifications are applied)
        
        Args:
            new_inp_path: Path to the modified INP file
        """
        self.current_inp_file = Path(new_inp_path)
        self.logger.debug(f"Updated INP file path: {new_inp_path}")


def analyze_example_inp(inp_file_path: str = "inp_file/Example1.inp") -> Dict:
    """
    Analyze Example1.inp file to understand subcatchment characteristics
    
    Args:
        inp_file_path: Path to the INP file
        
    Returns:
        Dictionary with analysis results including baseline data
    """
    logger = logging.getLogger(__name__)
    logger.info("[Analyzing input SWMM model]")
    logger.info(f"Analyzing INP file: {inp_file_path}")
    
    simulator = SWMMSimulator(inp_file_path, logger)
    
    try:
        # Get baseline performance
        baseline = simulator.get_baseline_performance()
        
        if not baseline.simulation_successful:
            return {"error": "Failed to analyze INP file"}
        
        # Find target subcatchment
        target_subcatchment = simulator.find_highest_runoff_subcatchment(baseline)
        
        # Create analysis summary
        analysis = {
            "total_subcatchments": len(baseline.subcatchments),
            "total_runoff_m3": baseline.total_runoff_m3,
            "baseline_results": baseline,  # Include baseline results for reuse
            "target_subcatchment": {
                "id": target_subcatchment.id,
                "area_m2": target_subcatchment.area_m2,
                "impervious_area_m2": target_subcatchment.impervious_area_m2,
                "percent_impervious": target_subcatchment.percent_impervious,
                "runoff_m3": target_subcatchment.runoff_m3
            },
            "all_subcatchments": [
                {
                    "id": sub.id,
                    "area_m2": sub.area_m2,
                    "runoff_m3": sub.runoff_m3,
                    "percent_impervious": sub.percent_impervious
                }
                for sub in sorted(baseline.subcatchments, key=lambda x: x.runoff_m3, reverse=True)
            ]
        }
        
        return analysis
        
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    # Test the simulator with Example1.inp
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analysis = analyze_example_inp()
    
    if "error" not in analysis:
        print("\nSWMM Analysis Results:")
        print(f"   Total Subcatchments: {analysis['total_subcatchments']}")
        print(f"   Total Runoff: {analysis['total_runoff_m3']:.2f} m³")
        print(f"\nTarget Subcatchment: {analysis['target_subcatchment']['id']}")
        print(f"   Area: {analysis['target_subcatchment']['area_m2']:.1f} m²")
        print(f"   Runoff: {analysis['target_subcatchment']['runoff_m3']:.2f} m³")
        print("Analysis completed successfully!")
    else:
        print(f"Analysis failed: {analysis['error']}") 