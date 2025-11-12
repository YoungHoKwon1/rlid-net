"""
SWMM Simulation Manager

This module provides a comprehensive interface for running SWMM simulations,
extracting results, and managing subcatchment data for LID optimization.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from pyswmm import Simulation, Nodes, Subcatchments, Links


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


class SWMMSimulator:
    """
    Enhanced SWMM simulation manager with improved error handling and data management
    """
    
    def __init__(self, inp_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize SWMM simulator
        
        Args:
            inp_file: Path to SWMM input file
            logger: Optional logger instance
        """
        self.inp_file = Path(inp_file)
        if not self.inp_file.exists():
            raise FileNotFoundError(f"SWMM input file not found: {inp_file}")
            
        self.logger = logger or self._setup_logger()
        self.results: Optional[SimulationResults] = None
        
        self.logger.info(f"SWMMSimulator initialized for: {self.inp_file}")
    
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
            logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 변경
        return logger
    
    def run_simulation(self, show_progress: bool = False) -> SimulationResults:
        """
        Run SWMM simulation and extract results
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            SimulationResults object containing all extracted data
        """
        start_time = datetime.datetime.now()
        
        try:
            with Simulation(str(self.inp_file)) as sim:
                self.logger.info("Starting SWMM simulation")
                
                # Extract simulation metadata
                flow_units = sim.flow_units
                system_units = sim.system_units
                sim_start_time = sim.start_time
                sim_end_time = sim.end_time
                
                self.logger.info(f"Flow Units: {flow_units}")
                self.logger.info(f"System Units: {system_units}")
                self.logger.info(f"Simulation Period: {sim_start_time} to {sim_end_time}")
                
                # Initialize data containers
                subcatchments_data = []
                
                # Get subcatchments, nodes, and links
                subcatchments = Subcatchments(sim)
                nodes = Nodes(sim)
                links = Links(sim)
                
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
                    
                    self.logger.debug(f"Subcatchment {subc_data.id}: "
                                    f"Area={area_m2:,.0f}m², Runoff={runoff_m3:.2f}m³, "
                                    f"Impervious={percent_impervious:.1f}% ({subc_data.impervious_area_m2:,.0f}m²)")
                
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
                    end_time=sim_end_time
                )
                
                self.logger.info(f"Simulation completed successfully in {simulation_time:.2f}s")
                self.logger.info(f"Total runoff: {total_runoff:.2f} m³")
                
                return self.results
                
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise
    
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
        
        top_subcatchments = self.results.subcatchments[:n]
        self.logger.info(f"Top {n} subcatchments by runoff:")
        for i, subc in enumerate(top_subcatchments, 1):
            self.logger.info(f"  {i}. {subc.id}: {subc.runoff_m3:.2f} m³ "
                           f"(Area: {subc.area_m2:,.0f} m²)")
        
        return top_subcatchments
    
    def get_subcatchment_by_id(self, subcatchment_id: str) -> Optional[SubcatchmentData]:
        """Get subcatchment data by ID"""
        if self.results is None:
            return None
        
        for subc in self.results.subcatchments:
            if subc.id == subcatchment_id:
                return subc
        return None
    
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
    
    def save_results(self, output_file: str) -> None:
        """Save simulation results to file"""
        if self.results is None:
            raise ValueError("No results to save")
        
        # Implementation for saving results to CSV/JSON
        # This can be expanded based on requirements
        pass 