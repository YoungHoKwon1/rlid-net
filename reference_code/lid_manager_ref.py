# """
# LID Management System

# This module provides comprehensive LID (Low Impact Development) management
# including configuration, placement, and INP file manipulation.
# """

# import os
# import logging
# from typing import Dict, List, Optional, Tuple, Any
# from dataclasses import dataclass, field
# from pathlib import Path
# from enum import Enum

# from .swmm_simulator import SubcatchmentData


# class LIDType(Enum):
#     """Enumeration of supported LID types"""
#     RAIN_GARDEN = ("RG", "Rain Garden")
#     GREEN_ROOF = ("GR", "Green Roof")
#     PERMEABLE_PAVEMENT = ("PP", "Permeable Pavement")
#     # INFILTRATION_TRENCH = ("IT", "Infiltration Trench")
#     # BIO_RETENTION_CELL = ("BC", "Bio-Retention Cell")
#     # RAIN_BARREL = ("RB", "Rain Barrel")
#     # VEGETATIVE_SWALE = ("VS", "Vegetative Swale")
#     # ROOFTOP_DISCONNECTION = ("RD", "Rooftop Disconnection")
    
#     def __init__(self, code: str, display_name: str):
#         self.code = code
#         self.display_name = display_name


# @dataclass
# class LIDParameters:
#     """Container for LID layer parameters"""
#     surface: Optional[List[float]] = None
#     soil: Optional[List[float]] = None
#     storage: Optional[List[float]] = None
#     pavement: Optional[List[float]] = None
#     drain: Optional[List[float]] = None
#     drainmat: Optional[List[float]] = None


# @dataclass
# class LIDConfiguration:
#     """Complete LID configuration"""
#     name: str
#     lid_type: LIDType
#     parameters: LIDParameters
#     life_cycle_cost_per_m2: float = 100000.0
    
#     def get_type_code(self) -> str:
#         return self.lid_type.code


# @dataclass
# class LIDPlacement:
#     """LID placement configuration for a subcatchment"""
#     subcatchment_id: str
#     lid_config: LIDConfiguration
#     area_percentage: float  # Changed from int to float for flexible constraints
#     area_m2: float
#     number_of_units: int = 1
#     width: float = 0.0
#     initial_saturation: float = 0.0
#     from_impervious: float = 100.0
#     to_pervious: int = 0
#     report_file: str = "*"
#     drain_to: str = "*"
#     from_pervious: float = 0.0
    
#     def calculate_total_cost(self) -> float:
#         """Calculate total LID cost"""
#         return self.area_m2 * self.lid_config.life_cycle_cost_per_m2


# class LIDManager:
#     """
#     Enhanced LID management system with automated configuration and placement
#     """
    
#     # Default parameter sets for different LID types
#     """
#     https://www.epa.gov/sites/default/files/2019-02/documents/epaswmm5_1_manual_master_8-2-15.pdf
#     page 291


#     surface: [berm height, vegetation volume fraction, roughness, slope, swale side slope]
#     soil: [thickness, porosity, field_capacity, wilting_point, conductivity, conductivity_slope, suction_head]
#     storage: [thickness, void_ratio, seepage_rate, clogging_factor, covered]
#     pavement: [thickness, void_ratio, imperv_surf_fraction, permeability, clogging_factor, regen_interval, regen_fraction]
#     drain: [flow_coeff, flow_exponent, offset, drain_delay, open_level, closed_level]
#     drainmat: [thickness, void_fraction, roughness]
#     """
#     DEFAULT_PARAMETERS = {
#         LIDType.RAIN_GARDEN: {
#             "surface": [150, 0.1, 0.1, 1, 0],
#             "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
#             "storage": [0, 0.75, 0.5, 0, "NO"]
#         },
#         LIDType.GREEN_ROOF: {
#             "surface": [150, 0.1, 0.1, 1, 0],
#             "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
#             "drainmat": [3, 0.5, 0.1]
#         },
#         LIDType.PERMEABLE_PAVEMENT: {
#             "surface": [150, 0.1, 0.1, 1, 0],
#             "pavement": [150, 0.4, 0.3, 72, 0, 0, 0],
#             "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
#             "storage": [150, 0.75, 0.5, 0, "NO"],
#             "drain": [0, 0.5, 6, 6, 0, 0]
#         },
#         # LIDType.INFILTRATION_TRENCH: {
#         #     "surface": [150, 0.1, 0.1, 1, 0],
#         #     "storage": [150, 0.75, 0.5, 0, "NO"],
#         #     "drain": [0, 0.5, 6, 6, 0, 0]
#         # },
#         # LIDType.BIO_RETENTION_CELL: {
#         #     "surface": [150, 0.1, 0.1, 1, 0],
#         #     "soil": [500, 0.5, 0.2, 0.1, 50, 10, 50],
#         #     "storage": [150, 0.75, 0.5, 0, "NO"],
#         #     "drain": [0, 0.5, 6, 6, 0, 0]
#         # },
#         # LIDType.RAIN_BARREL: {
#         #     "storage": [150, 0.75, 0.5, 0, "NO"],
#         #     "drain": [0, 0.5, 6, 6, 0, 0]
#         # },
#         # LIDType.VEGETATIVE_SWALE: {
#         #     "surface": [13.0, 0.0, 0.1, 1.0, 5]
#         # },
#         # LIDType.ROOFTOP_DISCONNECTION: {
#         #     "surface": [100, 0, 0.015, 1, 0],
#         #     "drain": [0, 0.5, 6, 6, 0, 0]
#         # }
#     }
    
#     def __init__(self, logger: Optional[logging.Logger] = None):
#         """Initialize LID manager"""
#         self.logger = logger or self._setup_logger()
#         self.configurations: Dict[str, LIDConfiguration] = {}
#         self.placements: List[LIDPlacement] = []
    
#     def _setup_logger(self) -> logging.Logger:
#         """Setup default logger"""
#         logger = logging.getLogger(__name__)
#         if not logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter(
#                 '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#             )
#             handler.setFormatter(formatter)
#             logger.addHandler(handler)
#             logger.setLevel(logging.INFO)
#         return logger
    
#     def create_lid_configuration(
#         self,
#         name: str,
#         lid_type: LIDType,
#         custom_parameters: Optional[Dict[str, List[float]]] = None,
#         life_cycle_cost_per_m2: float = 100000.0
#     ) -> LIDConfiguration:
#         """
#         Create a new LID configuration
        
#         Args:
#             name: Name for the LID configuration
#             lid_type: Type of LID
#             custom_parameters: Custom parameters (uses defaults if None)
#             life_cycle_cost_per_m2: Life cycle cost per square meter
            
#         Returns:
#             LIDConfiguration object
#         """
#         # Use custom parameters or defaults
#         params_dict = custom_parameters or self.DEFAULT_PARAMETERS.get(lid_type, {})
        
#         # Create parameters object
#         parameters = LIDParameters(
#             surface=params_dict.get("surface"),
#             soil=params_dict.get("soil"),
#             storage=params_dict.get("storage"),
#             pavement=params_dict.get("pavement"),
#             drain=params_dict.get("drain"),
#             drainmat=params_dict.get("drainmat")
#         )
        
#         config = LIDConfiguration(
#             name=name,
#             lid_type=lid_type,
#             parameters=parameters,
#             life_cycle_cost_per_m2=life_cycle_cost_per_m2
#         )
        
#         self.configurations[name] = config
#         self.logger.info(f"Created LID configuration: {name} ({lid_type})")
#         self.logger.info(f"  - Parameters: {parameters}")
#         self.logger.info(f"  - Life cycle cost per m2: {life_cycle_cost_per_m2}")
        
#         return config
    
#     def create_lid_placement(
#         self,
#         subcatchment_data: SubcatchmentData,
#         lid_config: LIDConfiguration,
#         area_percentage: float,
#         **kwargs
#     ) -> LIDPlacement:
#         """
#         Create LID placement with from_impervious-based area calculation
        
#         Args:
#             subcatchment_data: Target subcatchment information
#             lid_config: LID configuration to use
#             area_percentage: Percentage of IMPERVIOUS area to use for LID (1-100%)
#                            This value will be used directly as from_impervious
#             **kwargs: Additional placement parameters
            
#         Returns:
#             Created LID placement
#         """
#         # Calculate impervious area
#         impervious_area_m2 = subcatchment_data.area_m2 * (subcatchment_data.percent_impervious / 100) # percent_impervious is a integer (0-100)
        
#         # 직접적인 from_impervious 기반 계산
#         from_impervious = area_percentage
        
#         # from_impervious 퍼센트를 불투수면적에 적용하여 실제 LID 면적 계산
#         lid_area_m2 = (impervious_area_m2 * from_impervious) / 100
        
#         # 전체 소유역 면적 대비 퍼센트 계산 (SWMM LID_USAGE area 필드용)
#         total_area_percentage = (lid_area_m2 / subcatchment_data.area_m2) * 100
        
#         # Ensure from_impervious doesn't exceed 100%
#         from_impervious = min(100.0, from_impervious)
        
#         self.logger.info(f"Creating LID placement: {lid_config.name} on {subcatchment_data.id}")
#         self.logger.info(f"  - from_impervious: {from_impervious:.1f}% of impervious area")
#         self.logger.info(f"  - LID area: {lid_area_m2:.2f} m² ({total_area_percentage:.2f}% of total area)")
#         self.logger.info(f"  - Impervious area: {impervious_area_m2:.2f} m² ({subcatchment_data.percent_impervious:.1f}% of total)")

#         placement = LIDPlacement(
#             subcatchment_id=subcatchment_data.id,
#             lid_config=lid_config,
#             area_percentage=total_area_percentage,  # Store as total area percentage for SWMM
#             area_m2=lid_area_m2,
#             from_impervious=from_impervious,  # Direct from_impervious value
#             **kwargs
#         )
        
#         self.placements.append(placement)
        
#         self.logger.debug(f"LID placement created: {placement.lid_config.name} "
#                          f"(area: {placement.area_m2:.2f} m², "
#                          f"total_area%: {placement.area_percentage:.2f}%, "
#                          f"from_impervious: {placement.from_impervious:.1f}%)")
        
#         return placement
    
#     def optimize_lid_placement(
#         self,
#         subcatchments: List[SubcatchmentData],
#         lid_configs: List[LIDConfiguration],
#         max_placements: int = 3
#     ) -> List[LIDPlacement]:
#         """
#         Automatically optimize LID placement (basic greedy approach)
#         This can be enhanced with RL later
        
#         Args:
#             subcatchments: Available subcatchments (sorted by runoff)
#             lid_configs: Available LID configurations
#             max_placements: Maximum number of placements
            
#         Returns:
#             List of optimized LID placements
#         """
#         optimized_placements = []
        
#         # Simple greedy approach: place LIDs on highest runoff subcatchments
#         for i, subcatchment in enumerate(subcatchments[:max_placements]):
#             # Select LID type (can be enhanced with more sophisticated logic)
#             lid_config = lid_configs[i % len(lid_configs)]
            
#             # Start with moderate area percentage
#             area_percentage = 50
            
#             placement = self.create_lid_placement(
#                 subcatchment_data=subcatchment,
#                 lid_config=lid_config,
#                 area_percentage=area_percentage
#             )
            
#             optimized_placements.append(placement)
        
#         self.logger.info(f"Generated {len(optimized_placements)} optimized LID placements")
#         return optimized_placements
    
#     def add_lid_controls_to_inp(
#         self,
#         inp_file: str,
#         output_file: Optional[str] = None
#     ) -> str:
#         """
#         Add LID controls to INP file
        
#         Args:
#             inp_file: Path to input INP file
#             output_file: Path to output file (auto-generated if None)
            
#         Returns:
#             Path to modified INP file
#         """
#         if output_file is None:
#             base_path = Path(inp_file)
#             output_file = str(base_path.parent / f"{base_path.stem}_with_controls{base_path.suffix}")
        
#         # Read input file
#         with open(inp_file, 'r') as f:
#             lines = f.readlines()
        
#         # Find or create [LID_CONTROLS] section
#         lid_controls_index = self._find_or_create_section(lines, "[LID_CONTROLS]")
        
#         # Add LID control definitions
#         new_lines = []
#         for config in self.configurations.values():
#             new_lines.extend(self._generate_lid_control_lines(config))
        
#         # Calculate the offset for header comments (if they exist)
#         header_offset = 0
#         if lid_controls_index + 1 < len(lines) and lines[lid_controls_index + 1].startswith(';;Name'):
#             header_offset = 2  # Skip header comment lines
        
#         # Insert new lines after header comments
#         for i, line in enumerate(new_lines):
#             lines.insert(lid_controls_index + 1 + header_offset + i, line)
        
#         # Write output file
#         with open(output_file, 'w') as f:
#             f.writelines(lines)
        
#         self.logger.info(f"Added {len(self.configurations)} LID controls to {output_file}")
#         return output_file
    
#     def add_lid_usage_to_inp(
#         self,
#         inp_file: str,
#         output_file: Optional[str] = None
#     ) -> str:
#         """
#         Add LID usage to INP file
        
#         Args:
#             inp_file: Path to input INP file
#             output_file: Path to output file (auto-generated if None)
            
#         Returns:
#             Path to modified INP file
#         """
#         if output_file is None:
#             base_path = Path(inp_file)
#             output_file = str(base_path.parent / f"{base_path.stem}_with_usage{base_path.suffix}")
        
#         # Read input file
#         with open(inp_file, 'r') as f:
#             lines = f.readlines()
        
#         # Find or create [LID_USAGE] section
#         lid_usage_index = self._find_or_create_section(lines, "[LID_USAGE]")
        
#         # Add LID usage definitions
#         new_lines = []
#         for placement in self.placements:
#             new_lines.append(self._generate_lid_usage_line(placement))
        
#         # Calculate the offset for header comments (if they exist)
#         header_offset = 0
#         if lid_usage_index + 1 < len(lines) and lines[lid_usage_index + 1].startswith(';;Subcatchment'):
#             header_offset = 2  # Skip header comment lines
        
#         # Insert new lines after header comments
#         for i, line in enumerate(new_lines):
#             lines.insert(lid_usage_index + 1 + header_offset + i, line)
        
#         # Write output file
#         with open(output_file, 'w') as f:
#             f.writelines(lines)
        
#         self.logger.info(f"Added {len(self.placements)} LID usages to {output_file}")
#         return output_file
    
#     def reset_placements(self) -> None:
#         """Reset all LID placements"""
#         self.placements.clear()
#         self.logger.info("Reset all LID placements")
    
#     def get_total_cost(self) -> float:
#         """Calculate total cost of all LID placements"""
#         return sum(placement.calculate_total_cost() for placement in self.placements)
    
#     def get_total_area(self) -> float:
#         """Calculate total LID area"""
#         return sum(placement.area_m2 for placement in self.placements)
    
#     def load_custom_parameters_from_file(self, file_path: str) -> Dict[LIDType, Dict[str, List[float]]]:
#         """
#         Load custom LID parameters from configuration file
        
#         Args:
#             file_path: Path to the configuration file (JSON or YAML)
            
#         Returns:
#             Dictionary mapping LID types to their custom parameters
#         """
#         import json
#         import yaml
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 if file_path.endswith('.json'):
#                     data = json.load(f)
#                 elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
#                     data = yaml.safe_load(f)
#                 else:
#                     raise ValueError("Unsupported file format. Use JSON or YAML.")
            
#             # Convert string keys to LIDType enum
#             custom_params = {}
#             for lid_type_str, params in data.items():
#                 try:
#                     lid_type = LIDType[lid_type_str.upper()]
#                     custom_params[lid_type] = params
#                 except KeyError:
#                     self.logger.warning(f"Unknown LID type: {lid_type_str}")
#                     continue
            
#             self.logger.info(f"Loaded custom parameters from {file_path}")
#             return custom_params
            
#         except Exception as e:
#             self.logger.error(f"Failed to load custom parameters: {e}")
#             return {}
    
#     def get_user_defined_parameters(self) -> Dict[LIDType, Dict[str, List[float]]]:
#         """
#         Interactive function to get user-defined LID parameters
        
#         Returns:
#             Dictionary mapping LID types to their custom parameters
#         """
#         print()
#         print("=" * 60)
#         print("Define LID parameters")
#         print("=" * 60)
        
#         custom_params = {}
        
#         for lid_type in LIDType:
#             print(f"{lid_type.display_name} ({lid_type.code})")
#             use_custom = input(f"Do you want to use custom parameters? (y/n) [default: n]: ").lower()
            
#             if use_custom == 'y':
#                 params = {}
#                 default_params = self.DEFAULT_PARAMETERS.get(lid_type, {})
                
#                 for layer_name, default_values in default_params.items():
#                     print(f"\n  {layer_name.upper()} layer:")
#                     print(f"    default: {default_values}")
                    
#                     custom_input = input(f"    new value (comma separated, enter for default): ").strip()
                    
#                     if custom_input:
#                         try:
#                             # Parse user input
#                             values = []
#                             for val in custom_input.split(','):
#                                 val = val.strip()
#                                 if val.upper() in ['NO', 'YES']:
#                                     values.append(val.upper())
#                                 else:
#                                     values.append(float(val))
#                             params[layer_name] = values
#                             print(f"    set: {values}")
#                         except ValueError:
#                             print(f"    invalid input format. using default: {default_values}")
#                             params[layer_name] = default_values
#                     else:
#                         params[layer_name] = default_values
                
#                 custom_params[lid_type] = params
#                 print(f"set {lid_type.display_name}")
#             else:
#                 print(f"use default {lid_type.display_name}")
        
#         return custom_params
    
#     def save_custom_parameters(self, custom_params: Dict[LIDType, Dict[str, List[float]]], file_path: str) -> None:
#         """
#         Save custom parameters to file for future use
        
#         Args:
#             custom_params: Custom parameters dictionary
#             file_path: Output file path
#         """
#         import json
        
#         # Convert LIDType enum to string for JSON serialization
#         serializable_params = {}
#         for lid_type, params in custom_params.items():
#             serializable_params[lid_type.name] = params
        
#         try:
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(serializable_params, f, indent=2, ensure_ascii=False)
            
#             self.logger.info(f"Custom parameters saved to {file_path}")
#             print(f"custom parameters saved to {file_path}")
            
#         except Exception as e:
#             self.logger.error(f"Failed to save custom parameters: {e}")
#             print(f"failed to save custom parameters: {e}")

#     def create_all_default_configurations(self, custom_params: Optional[Dict[LIDType, Dict[str, List[float]]]] = None) -> Dict[str, LIDConfiguration]:
#         """
#         Create all default LID configurations with optional custom parameters
        
#         Args:
#             custom_params: Optional custom parameters to override defaults
            
#         Returns:
#             Dictionary of all LID configurations
#         """
#         configurations = {}
        
#         for lid_type in LIDType:
#             name = f"{lid_type.code}_default"
            
#             # Use custom parameters if provided, otherwise use defaults
#             if custom_params and lid_type in custom_params: # if custom_params is not None and lid_type is in custom_params, use custom parameters
#                 params = custom_params[lid_type]
#             else: # if custom_params is None, use default parameters
#                 params = self.DEFAULT_PARAMETERS.get(lid_type, {})
            
#             config = self.create_lid_configuration(
#                 name=name, # e.g. RG_default
#                 lid_type=lid_type, # e.g. LIDType.RAIN_GARDEN
#                 custom_parameters=params # e.g. {surface: [x.xx, x.xx, x.xx], soil: [x.xx, x.xx, x.xx, x.xx, x.xx, x.xx, x.xx], storage: [x.xx, x.xx, x.xx, x.xx, x.xx], pavement: [x.xx, x.xx, x.xx, x.xx, x.xx, x.xx, x.xx], drain: [x.xx, x.xx, x.xx, x.xx, x.xx, x.xx], drainmat: [x.xx, x.xx, x.xx]}
#             )
#             configurations[name] = config # e.g. {'RG_default': LIDConfiguration(name='RG_default', lid_type=<LIDType.RAIN_GARDEN: ('RG', 'Rain Garden')>, parameters=LIDParameters(surface=[150, 0.1, 0.1, 1, 0], soil=[500, 0.5, 0.2, 0.1, 50, 10, 50], storage=[0, 0.75, 0.5, 0, 'NO'], pavement=None, drain=None, drainmat=None), life_cycle_cost_per_m2=0.0),...}
#         return configurations
    
#     def _find_or_create_section(self, lines: List[str], section_name: str) -> int:
#         """Find existing section or create new one"""
#         # Find existing section
#         for i, line in enumerate(lines):
#             if line.strip() == section_name:
#                 return i
        
#         # Section not found, find appropriate location to insert it
#         # For LID sections, insert after [INFILTRATION] and before [JUNCTIONS]
#         insert_index = len(lines)
        
#         if section_name in ['[LID_CONTROLS]', '[LID_USAGE]']:
#             # Look for [JUNCTIONS] section to insert before it
#             for i, line in enumerate(lines):
#                 stripped_line = line.strip()
#                 if stripped_line == '[JUNCTIONS]':
#                     insert_index = i
#                     break
            
#             # If [JUNCTIONS] not found, look for other sections to insert before
#             if insert_index == len(lines):
#                 for i, line in enumerate(lines):
#                     stripped_line = line.strip()
#                     if stripped_line.startswith('[') and stripped_line in ['[OUTFALLS]', '[CONDUITS]', '[XSECTIONS]']:
#                         insert_index = i
#                         break
#         else:
#             # For other sections, use original logic
#             for i, line in enumerate(lines):
#                 stripped_line = line.strip()
#                 if stripped_line.startswith('[') and stripped_line in ['[SYMBOLS]', '[MAP]', '[COORDINATES]']:
#                     insert_index = i
#                     break
        
#         # Insert the section header with proper spacing and header comments
#         lines.insert(insert_index, f"\n{section_name}\n")
        
#         # Add header comments for LID sections
#         if section_name == '[LID_CONTROLS]':
#             lines.insert(insert_index + 1, ";;Name           Type/Layer Parameters\n")
#             lines.insert(insert_index + 2, ";;-------------- ---------- ----------\n")
#         elif section_name == '[LID_USAGE]':
#             lines.insert(insert_index + 1, ";;Subcatchment   LID Process      Number  Area       Width      InitSat    FromImp    ToPerv     RptFile                  DrainTo          FromPerv\n")
#             lines.insert(insert_index + 2, ";;-------------- ---------------- ------- ---------- ---------- ---------- ---------- ------------------------ ---------------- ----------\n")
        
#         return insert_index
    
#     def _generate_lid_control_lines(self, config: LIDConfiguration) -> List[str]:
#         """Generate LID control lines for INP file"""
#         lines = []
#         lines.append(f"{config.name}\t\t{config.get_type_code()}\n")
        
#         # Add parameter lines
#         params = config.parameters
#         if params.surface:
#             param_str = "\t".join(str(p) for p in params.surface)
#             lines.append(f"{config.name}\t\tSURFACE\t{param_str}\n")
        
#         if params.soil:
#             param_str = "\t".join(str(p) for p in params.soil)
#             lines.append(f"{config.name}\t\tSOIL\t{param_str}\n")
        
#         if params.storage:
#             param_str = "\t".join(str(p) for p in params.storage)
#             lines.append(f"{config.name}\t\tSTORAGE\t{param_str}\n")
        
#         if params.pavement:
#             param_str = "\t".join(str(p) for p in params.pavement)
#             lines.append(f"{config.name}\t\tPAVEMENT\t{param_str}\n")
        
#         if params.drain:
#             param_str = "\t".join(str(p) for p in params.drain)
#             lines.append(f"{config.name}\t\tDRAIN\t{param_str}\n")
        
#         if params.drainmat:
#             param_str = "\t".join(str(p) for p in params.drainmat)
#             lines.append(f"{config.name}\t\tDRAINMAT\t{param_str}\n")
        
#         return lines
    
#     def _generate_lid_usage_line(self, placement: LIDPlacement) -> str:
#         """Generate LID usage line for INP file"""
#         # Special handling for Vegetative Swale - all values must be non-zero
#         width = placement.width
#         # if placement.lid_config.lid_type == LIDType.VEGETATIVE_SWALE and width == 0.0:
#         #     # For VS, calculate appropriate width based on area
#         #     # Assuming rectangular swale: width = sqrt(area) for simplicity
#         #     width = max(1.0, (placement.area_m2 ** 0.5))
        
#         # SWMM expects area as percentage of total subcatchment area, not absolute area
#         return (f"{placement.subcatchment_id}\t{placement.lid_config.name}\t"
#                 f"{placement.number_of_units}\t{placement.area_m2:.2f}\t"
#                 f"{width:.2f}\t{placement.initial_saturation:.1f}\t"
#                 f"{placement.from_impervious:.1f}\t{placement.to_pervious}\t"
#                 f"{placement.report_file}\t{placement.drain_to}\t"
#                 f"{placement.from_pervious:.1f}\n") 