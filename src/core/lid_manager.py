#!/usr/bin/env python3
"""
LID Manager Module for RLID-NET
Enhanced with proper LID parameter management and INP file generation
"""

import os
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from .swmm_simulator import SubcatchmentData
from ..utils.config import LID_COSTS, LID_TYPES, AREA_PERCENTAGES


class LIDType(Enum):
    """Enumeration of supported LID types with unique SWMM codes"""
    RAIN_GARDEN = ("RG", "Rain Garden")  # Changed from BC to RG
    GREEN_ROOF = ("GR", "Green Roof") 
    PERMEABLE_PAVEMENT = ("PP", "Permeable Pavement")
    INFILTRATION_TRENCH = ("IT", "Infiltration Trench")
    BIO_RETENTION_CELL = ("BC", "Bio-Retention Cell")  # Keeps BC
    RAIN_BARREL = ("RB", "Rain Barrel")
    VEGETATIVE_SWALE = ("VS", "Vegetative Swale")
    ROOFTOP_DISCONNECTION = ("RD", "Rooftop Disconnection")
    
    def __init__(self, code: str, display_name: str):
        self.code = code
        self.display_name = display_name
    
    @classmethod
    def from_display_name(cls, display_name: str):
        """Get LIDType from display name"""
        for lid_type in cls:
            if lid_type.display_name == display_name:
                return lid_type
        raise ValueError(f"Unknown LID type: {display_name}")


@dataclass
class LIDParameters:
    """Container for LID layer parameters"""
    surface: Optional[List[float]] = None
    soil: Optional[List[float]] = None
    storage: Optional[List[float]] = None
    pavement: Optional[List[float]] = None
    drain: Optional[List[float]] = None
    drainmat: Optional[List[float]] = None


@dataclass
class LIDPlacement:
    """Data structure for LID placement information"""
    subcatchment_id: str
    lid_type: str
    area_m2: float
    area_percentage: float  # Percentage of subcatchment's total area
    cost_krw: float
    number_of_units: int = 1
    width: float = 0.0
    initial_saturation: float = 0.0
    from_impervious: float = 100.0
    to_pervious: int = 0
    report_file: str = "*"
    drain_to: str = "*"
    from_pervious: float = 0.0
    
    @property
    def area_percentage_str(self) -> str:
        """Format area percentage for display"""
        return f"{self.area_percentage:.1f}%"


@dataclass
class LIDState:
    """Current state of all LID placements"""
    placements: List[LIDPlacement]
    total_area_m2: float = 0.0
    total_cost_krw: float = 0.0
    
    def __post_init__(self):
        """Calculate totals after initialization"""
        self.total_area_m2 = sum(p.area_m2 for p in self.placements)
        self.total_cost_krw = sum(p.cost_krw for p in self.placements)


class LIDManager:
    """
    Enhanced LID Management System for RLID-NET
    
    Handles LID placement, removal, INP file modifications, and constraint checking
    with proper SWMM parameter management
    """
    
    # LID parameter definitions for each type (verified SWMM parameters)
    DEFAULT_PARAMETERS = {
        LIDType.RAIN_GARDEN: {
            "surface": [150, 0.1, 0.1, 1, 0],
            "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
            "storage": [0, 0.75, 0.5, 0, "NO"]
        },
        LIDType.GREEN_ROOF: {
            "surface": [150, 0.1, 0.1, 1, 0],
            "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
            "drainmat": [3, 0.5, 0.1]
        },
        LIDType.PERMEABLE_PAVEMENT: {
            "surface": [150, 0.1, 0.1, 1, 0],
            "pavement": [150, 0.4, 0.3, 72, 0, 0, 0],
            "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
            "storage": [150, 0.75, 0.5, 0, "NO"],
            "drain": [0, 0.5, 6, 6, 0, 0]
        },
        LIDType.INFILTRATION_TRENCH: {
            "surface": [150, 0.1, 0.1, 1, 0],
            "storage": [150, 0.75, 0.5, 0, "NO"],
            "drain": [0, 0.5, 6, 6, 0, 0]
        },
        LIDType.BIO_RETENTION_CELL: {
            "surface": [150, 0.1, 0.1, 1, 0],
            "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
            "storage": [150, 0.75, 0.5, 0, "NO"],
            "drain": [0, 0.5, 6, 6, 0, 0]
        },
        LIDType.RAIN_BARREL: {
            "storage": [150, 0.75, 0.5, 0, "NO"],
            "drain": [0, 0.5, 6, 6, 0, 0]
        },
        LIDType.VEGETATIVE_SWALE: {
            "surface": [13.0, 0.0, 0.1, 1.0, 5]
        },
        LIDType.ROOFTOP_DISCONNECTION: {
            "surface": [100, 0, 0.015, 1, 0],
            "drain": [0, 0.5, 6, 6, 0, 0]
        }
    }
    
    def __init__(self, base_inp_file: str, target_subcatchment: SubcatchmentData, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize LID Manager
        
        Args:
            base_inp_file: Path to base SWMM INP file
            target_subcatchment: Target subcatchment for LID placement
            logger: Logger instance
        """
        self.base_inp_file = Path(base_inp_file)
        self.target_subcatchment = target_subcatchment
        self.logger = logger or logging.getLogger(__name__)
        
        # LID configuration from config
        self.LID_COSTS = LID_COSTS
        self.LID_TYPES = LID_TYPES
        self.AREA_PERCENTAGES = AREA_PERCENTAGES
        
        # Current LID state
        self.current_placements: Dict[str, LIDPlacement] = {}
        self.current_state = LIDState([])
        
        # Working directory for modified INP files
        self.work_dir: Optional[Path] = None
        self.current_inp_file: Optional[Path] = None
        
        # Constraint limits
        self.max_area_ratio = 0.5  # Maximum 50% of impervious area
        
        self.logger.info("[LID Manager initialized]")
        self.logger.info(f"LID Manager initialized for subcatchment: {target_subcatchment.id}")
        self.logger.info(f"   Available impervious area: {target_subcatchment.impervious_area_m2:.1f} m²")
    
    def setup_working_directory(self, work_dir: Path):
        """
        Setup working directory for INP file modifications
        
        Args:
            work_dir: Working directory path
        """
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy base INP file to working directory
        self.current_inp_file = self.work_dir / "current_model.inp"
        shutil.copy2(self.base_inp_file, self.current_inp_file)
        
        self.logger.info(f"LID Manager working directory: {work_dir}")
    
    def can_apply_action(self, lid_type: str, area_percentage: float) -> Tuple[bool, str]:
        """
        Check if LID action can be applied with improved area validation
        
        Args:
            lid_type: Type of LID to place/modify
            area_percentage: Area percentage of subcatchment
            
        Returns:
            (can_apply, reason) tuple
        """
        area_m2 = self.target_subcatchment.area_m2 * area_percentage / 100
        
        # Check for negative area when no LID exists (removal constraint)
        if area_percentage < 0:
            if lid_type not in self.current_placements:
                return False, f"Cannot remove {lid_type}: not currently placed"
            
            current_area = self.current_placements[lid_type].area_m2
            if abs(area_m2) > current_area:
                return False, f"Cannot remove {abs(area_m2):.1f}m² of {lid_type}: only {current_area:.1f}m² available"
        
        # Calculate total current LID coverage
        total_current_area = sum(placement.area_m2 for placement in self.current_placements.values())
        
        # Calculate area change
        current_lid_area = self.current_placements.get(lid_type, type('obj', (object,), {'area_m2': 0.0})).area_m2
        area_change = area_m2
        new_lid_area = max(0.0, current_lid_area + area_change)
        
        # Calculate new total area
        new_total_area = total_current_area - current_lid_area + new_lid_area
        
        # Check against impervious area limit with safety margin
        impervious_area_m2 = self.target_subcatchment.impervious_area_m2
        max_allowable_area = impervious_area_m2 * 0.95  # 95% safety margin
        
        if new_total_area > max_allowable_area:
            coverage_percentage = (new_total_area / impervious_area_m2) * 100
            return False, f"Total LID area would exceed impervious area limit: {coverage_percentage:.1f}% > 95%"
        
        # Check minimum area constraint
        if area_percentage > 0 and area_m2 < 1.0:
            return False, f"LID area too small: {area_m2:.1f}m² < 1.0m²"
        
        return True, "Action can be applied"
    
    def apply_lid_action(self, lid_type: str, area_percentage: float) -> Tuple[bool, str, Optional[LIDPlacement]]:
        """
        Apply LID placement or removal action
        
        Args:
            lid_type: Type of LID
            area_percentage: Area percentage (negative for removal)
            
        Returns:
            (success, message, placement) tuple
        """
        # Check constraints first
        can_apply, reason = self.can_apply_action(lid_type, area_percentage)
        if not can_apply:
            return False, reason, None
        
        if area_percentage < 0:
            # Remove existing LID
            return self._remove_lid(lid_type, abs(area_percentage))
        elif area_percentage > 0:
            # Add new LID
            return self._add_lid(lid_type, area_percentage)
        else:
            # Zero percentage - no action
            return True, "No action (0% area)", None
    
    def _add_lid(self, lid_type: str, area_percentage: float) -> Tuple[bool, str, LIDPlacement]:
        """
        Add LID to subcatchment
        
        Args:
            lid_type: Type of LID to add
            area_percentage: Area percentage of subcatchment
            
        Returns:
            (success, message, placement) tuple
        """
        # Calculate areas and costs
        area_m2 = self.target_subcatchment.area_m2 * area_percentage / 100
        cost_krw = area_m2 * LID_COSTS[lid_type]
        
        # Calculate from_impervious percentage based on actual LID area vs impervious area
        # from_impervious should be the percentage of impervious area that this LID will treat
        from_impervious_percentage = (area_m2 / self.target_subcatchment.impervious_area_m2) * 100
        from_impervious_percentage = min(100.0, from_impervious_percentage)  # Cap at 100%
        
        # Create placement object
        placement = LIDPlacement(
            subcatchment_id=self.target_subcatchment.id,
            lid_type=lid_type,
            area_m2=area_m2,
            area_percentage=area_percentage,
            cost_krw=cost_krw,
            from_impervious=from_impervious_percentage  # Use calculated percentage
        )
        
        # Update or add to current placements
        if lid_type in self.current_placements:
            # Add to existing LID
            existing = self.current_placements[lid_type]
            existing.area_m2 += area_m2
            existing.area_percentage += area_percentage
            existing.cost_krw += cost_krw
            # Update from_impervious to reflect total area
            total_area = existing.area_m2
            existing.from_impervious = min(100.0, (total_area / self.target_subcatchment.impervious_area_m2) * 100)
        else:
            # New LID placement
            self.current_placements[lid_type] = placement
        
        # Update current state
        self._update_current_state()
        
        # Modify INP file
        success = self._modify_inp_file()
        
        if success:
            message = f"Added {area_m2:.1f}m² ({area_percentage}%) of {lid_type}"
            self.logger.info(f"{message}")
            return True, message, placement
        else:
            # Rollback on failure
            if lid_type in self.current_placements:
                if self.current_placements[lid_type].area_m2 <= area_m2:
                    del self.current_placements[lid_type]
                else:
                    existing = self.current_placements[lid_type]
                    existing.area_m2 -= area_m2
                    existing.area_percentage -= area_percentage
                    existing.cost_krw -= cost_krw
                    # Recalculate from_impervious
                    existing.from_impervious = min(100.0, (existing.area_m2 / self.target_subcatchment.impervious_area_m2) * 100)
            
            self._update_current_state()
            return False, "Failed to modify INP file", None
    
    def _remove_lid(self, lid_type: str, area_percentage: float) -> Tuple[bool, str, Optional[LIDPlacement]]:
        """
        Remove LID from subcatchment
        
        Args:
            lid_type: Type of LID to remove
            area_percentage: Area percentage to remove (positive value)
            
        Returns:
            (success, message, placement) tuple
        """
        if lid_type not in self.current_placements:
            return False, f"{lid_type} not found in current placements", None
        
        # Calculate removal amounts
        removal_area = self.target_subcatchment.area_m2 * area_percentage / 100
        removal_cost = removal_area * LID_COSTS[lid_type]
        
        existing = self.current_placements[lid_type]
        
        # Check if removing all or partial
        if removal_area >= existing.area_m2:
            # Remove entire LID
            removed_placement = existing
            del self.current_placements[lid_type]
            message = f"Removed all {existing.area_m2:.1f}m² of {lid_type}"
        else:
            # Partial removal
            existing.area_m2 -= removal_area
            existing.area_percentage -= area_percentage
            existing.cost_krw -= removal_cost
            
            removed_placement = LIDPlacement(
                subcatchment_id=self.target_subcatchment.id,
                lid_type=lid_type,
                area_m2=removal_area,
                area_percentage=area_percentage,
                cost_krw=removal_cost
            )
            message = f"Removed {removal_area:.1f}m² ({area_percentage}%) of {lid_type}"
        
        # Update current state
        self._update_current_state()
        
        # Modify INP file
        success = self._modify_inp_file()
        
        if success:
            self.logger.info(f"{message}")
            return True, message, removed_placement
        else:
            return False, "Failed to modify INP file", None
    
    def _update_current_state(self):
        """Update current LID state summary"""
        placements = list(self.current_placements.values())
        self.current_state = LIDState(placements)
    
    def _modify_inp_file(self) -> bool:
        """
        Modify INP file with current LID placements using proper SWMM format
        
        Returns:
            Success status
        """
        try:
            # Read current INP file
            with open(self.current_inp_file, 'r') as f:
                lines = f.readlines()
            
            # Remove existing LID sections to prevent duplication
            lines = self._remove_existing_lid_sections(lines)
            
            # Generate LID controls and usage sections
            control_lines = self._generate_lid_control_lines()
            usage_lines = self._generate_lid_usage_lines()
            
            # Only proceed if we have valid LID data
            if not control_lines and not usage_lines:
                self.logger.debug("No LID placements to add - keeping original file")
                return True
            
            if not control_lines:
                self.logger.error("Failed to generate LID control lines")
                return False
            
            if not usage_lines:
                self.logger.error("Failed to generate LID usage lines")
                return False
            
            # Find insertion point (before [JUNCTIONS] section)
            insertion_point = self._find_insertion_point(lines)
            
            # Insert LID_CONTROLS section
            lid_controls_section = []
            lid_controls_section.append("\n[LID_CONTROLS]\n")
            lid_controls_section.append(";;Name           Type/Layer Parameters\n")
            lid_controls_section.append(";;-------------- ---------- ----------\n")
            lid_controls_section.extend(control_lines)
            lid_controls_section.append("\n")
            
            # Insert LID_USAGE section
            lid_usage_section = []
            lid_usage_section.append("[LID_USAGE]\n")
            lid_usage_section.append(";;Subcatchment   LID Process      Number  Area       Width      InitSat    FromImp    ToPerv\n")
            lid_usage_section.append(";;-------------- ---------------- ------- ---------- ---------- ---------- ---------- ----------\n")
            lid_usage_section.extend(usage_lines)
            lid_usage_section.append("\n")
            
            # Insert all LID sections at once
            for i, line in enumerate(lid_controls_section + lid_usage_section):
                lines.insert(insertion_point + i, line)
            
            # Write modified file
            with open(self.current_inp_file, 'w') as f:
                f.writelines(lines)
            
            self.logger.debug(f"Modified INP file with {len(control_lines)} control lines and {len(usage_lines)} usage lines")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to modify INP file: {str(e)}")
            return False
    

    
    def _remove_existing_lid_sections(self, lines: List[str]) -> List[str]:
        """Remove any existing LID_CONTROLS and LID_USAGE sections"""
        result_lines = []
        skip_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check if we're entering a LID section
            if stripped in ['[LID_CONTROLS]', '[LID_USAGE]']:
                skip_section = True
                continue
            
            # Check if we're entering a different section (exit LID section)
            if stripped.startswith('[') and stripped not in ['[LID_CONTROLS]', '[LID_USAGE]']:
                skip_section = False
            
            # Add line if we're not in a LID section
            if not skip_section:
                result_lines.append(line)
        
        return result_lines
    
    def _find_insertion_point(self, lines: List[str]) -> int:
        """Find appropriate insertion point for LID sections (before [JUNCTIONS])"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == '[JUNCTIONS]':
                return i
        
        # If [JUNCTIONS] not found, try other sections
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped in ['[OUTFALLS]', '[CONDUITS]', '[XSECTIONS]']:
                return i
        
        # Fallback: insert before end
        return len(lines) - 1
    
    def _generate_lid_control_lines(self) -> List[str]:
        """Generate LID control lines for INP file based on proper SWMM parameters"""
        lines = []
        processed_types = set()  # Prevent duplicate LID types
        
        for lid_type in self.current_placements.keys():
            # Skip if we already processed this LID type
            if lid_type in processed_types:
                continue
                
            try:
                lid_enum = LIDType.from_display_name(lid_type)
                # Use SWMM code as control name to match LID_USAGE
                control_name = lid_enum.code
                processed_types.add(lid_type)
                
                # Add type declaration
                lines.append(f"{control_name}\t\t{lid_enum.code}\n")
                
                # Get parameters for this LID type
                params_dict = self.DEFAULT_PARAMETERS.get(lid_enum, {})
                
                if not params_dict:
                    self.logger.error(f"No parameters defined for {lid_type}")
                    return []
                
                # Add parameter lines
                if 'surface' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['surface'])
                    lines.append(f"{control_name}\t\tSURFACE\t{param_str}\n")
                
                if 'soil' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['soil'])
                    lines.append(f"{control_name}\t\tSOIL\t{param_str}\n")
                
                if 'storage' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['storage'])
                    lines.append(f"{control_name}\t\tSTORAGE\t{param_str}\n")
                
                if 'pavement' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['pavement'])
                    lines.append(f"{control_name}\t\tPAVEMENT\t{param_str}\n")
                
                if 'drain' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['drain'])
                    lines.append(f"{control_name}\t\tDRAIN\t{param_str}\n")
                
                if 'drainmat' in params_dict:
                    param_str = "\t".join(str(p) for p in params_dict['drainmat'])
                    lines.append(f"{control_name}\t\tDRAINMAT\t{param_str}\n")
                    
            except ValueError as e:
                self.logger.error(f"LID type mapping failed for {lid_type}: {e}")
                return []
        
        return lines
    
    def _generate_lid_usage_line(self, placement: LIDPlacement) -> str:
        """
        Generate LID usage line for INP file with proper width handling
        
        Args:
            placement: LID placement configuration
            
        Returns:
            Formatted LID usage line
        """
        # Get SWMM code for LID type
        lid_type_enum = None
        for enum_type in LIDType:
            if enum_type.value[1] == placement.lid_type:  # Match by display name
                lid_type_enum = enum_type
                break
        
        if lid_type_enum is None:
            # Fallback: use string directly as SWMM code
            swmm_code = placement.lid_type.replace(' ', '_').replace('-', '_')
        else:
            swmm_code = lid_type_enum.value[0]  # Use SWMM code from enum
        
        # Calculate width based on LID type
        if placement.lid_type == "Vegetative Swale":
            # For swales, width is calculated from area and length
            # Assume standard length of 100m for swales
            width = max(1.0, placement.area_m2 / 100.0)
        elif placement.lid_type == "Rooftop Disconnection":
            # For rooftop disconnection, use area as width (linear feature)
            width = max(1.0, placement.area_m2 / 10.0)
        else:
            # For other LID types, calculate width from area (assume square layout)
            width = max(1.0, placement.area_m2 ** 0.5)
        
        # Ensure width is reasonable (between 1-1000m)
        width = min(max(width, 1.0), 1000.0)
        
        # Calculate from_impervious percentage
        from_impervious = (placement.area_m2 / self.target_subcatchment.impervious_area_m2) * 100
        from_impervious = min(from_impervious, 100.0)  # Cap at 100%
        
        # Generate line with proper formatting
        return (f"{placement.subcatchment_id:<15} "
                f"{swmm_code:<16} "
                f"{placement.number_of_units:<7} "
                f"{placement.area_m2:<10.1f} "
                f"{width:<10.2f} "
                f"{placement.initial_saturation:<10.1f} "
                f"{from_impervious:<10.2f} "
                f"{placement.to_pervious:<10.1f}")
    
    def _generate_lid_usage_lines(self) -> List[str]:
        """Generate LID usage lines for INP file with proper line breaks"""
        lines = []
        
        # Add usage lines for current placements
        for placement in self.current_placements.values():
            usage_line = self._generate_lid_usage_line(placement)
            lines.append(usage_line + "\n")  # Ensure proper line break
        
        return lines
    
    def get_current_state_vector(self) -> List[float]:
        """
        Get current LID state as vector for RL agent
        
        Returns:
            State vector with LID areas for each type (8 dimensions)
        """
        state_vector = []
        
        # LID areas in order of LID_TYPES
        for lid_type in LID_TYPES:
            if lid_type in self.current_placements:
                area = self.current_placements[lid_type].area_m2
            else:
                area = 0.0
            state_vector.append(area)
        
        return state_vector
    
    def get_current_state_summary(self) -> Dict:
        """
        Get detailed summary of current LID state
        
        Returns:
            Dictionary with current state information
        """
        return {
            "placements": [
                {
                    "lid_type": p.lid_type,
                    "area_m2": p.area_m2,
                    "area_percentage": p.area_percentage,
                    "cost_krw": p.cost_krw
                }
                for p in self.current_placements.values()
            ],
            "total_area_m2": self.current_state.total_area_m2,
            "total_cost_krw": self.current_state.total_cost_krw,
            "available_impervious_area_m2": self.target_subcatchment.impervious_area_m2,
            "area_utilization_ratio": self.current_state.total_area_m2 / self.target_subcatchment.impervious_area_m2
        }
    
    def reset_all_lids(self):
        """Remove all LID placements and reset to baseline"""
        self.current_placements.clear()
        self._update_current_state()
        
        # Copy base INP file to reset modifications
        if self.current_inp_file:
            shutil.copy2(self.base_inp_file, self.current_inp_file)
        
        self.logger.info("Reset all LID placements")
    
    def get_current_inp_file(self) -> Optional[Path]:
        """Get path to current modified INP file"""
        return self.current_inp_file


if __name__ == "__main__":
    # Test LID Manager functionality
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test with dummy subcatchment data
    from .swmm_simulator import SubcatchmentData
    
    test_subcatchment = SubcatchmentData(
        id="1",
        area_hectares=10.0,
        area_m2=100000,  # 10 hectares
        percent_impervious=50.0,  # 50% impervious
        runoff_m3=1000.0
    )
    
    manager = LIDManager("inp_file/Example1.inp", test_subcatchment)
    
    print("Testing Enhanced LID Manager:")
    print(f"   Target subcatchment: {test_subcatchment.id}")
    print(f"   Available impervious area: {test_subcatchment.impervious_area_m2:.1f} m²")
    
    # Test constraint checking
    can_add, reason = manager.can_apply_action("Rain Garden", 2.0)
    print(f"   Can add Rain Garden (2%): {can_add} - {reason}")
    
    can_remove, reason = manager.can_apply_action("Rain Garden", -1.0)
    print(f"   Can remove Rain Garden (1%): {can_remove} - {reason}")
    
    print("Enhanced LID Manager test completed!") 