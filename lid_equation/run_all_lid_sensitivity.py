import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

try:
    from reference_code.lid_manager_ref import LIDManager, LIDType, LIDConfiguration
    from reference_code.swmm_simulator_ref import SWMMSimulator
    from pyswmm import Simulation, Subcatchments
except ImportError as e:
    print(f"Error importing reference code: {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Font Settings
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis for ALL LID Types")
    parser.add_argument('--inp-file', type=str, required=True, help="Path to the SWMM .inp input file")
    return parser.parse_args()

def run_simulation_for_lid(inp_file_path, lid_type, target_sub, base_total, base_peak):
    print(f"\n>>> Analyzing LID Type: {lid_type.name} ({lid_type.value[1]})")
    
    lid_manager = LIDManager(logger=logger)
    
    # Use Default Parameters by creating configuration without custom_parameters
    # LIDManager will automatically use DEFAULT_PARAMETERS for the given type
    lid_config = lid_manager.create_lid_configuration(
        name=lid_type.name,
        lid_type=lid_type
    )
    
    results = []
    
    # 0% (Baseline)
    results.append({
        'LID_Type': lid_type.name,
        'Ratio': 0,
        'Total_Runoff': base_total,
        'Peak_Runoff': base_peak,
        'Red_Total': 0.0,
        'Red_Peak': 0.0
    })
    
    # Iteration 1% to 100% (Step 1% for detailed analysis)
    ratios = np.arange(1, 101, 1) 
    
    temp_inp = f"temp_all_{lid_type.name}.inp"
    temp_controls = f"temp_controls_{lid_type.name}.inp"
    
    for ratio in ratios:
        if ratio % 10 == 0:
            print(f"  Processing {ratio}%...", end='\r')
        
        lid_manager.reset_placements()
        
        # Calculate width (approximate square)
        expected_lid_area = target_sub.impervious_area_m2 * (ratio / 100.0)
        lid_width = np.sqrt(expected_lid_area) if expected_lid_area > 0 else 1.0
        
        lid_manager.create_lid_placement(
            subcatchment_data=target_sub,
            lid_config=lid_config,
            area_percentage=float(ratio),
            width=lid_width
        )
        
        lid_manager.add_lid_controls_to_inp(inp_file_path, temp_controls)
        lid_manager.add_lid_usage_to_inp(temp_controls, temp_inp)
        
        try:
            with Simulation(temp_inp) as sim:
                sub = Subcatchments(sim)[target_sub.id]
                for step in sim: pass
                
                curr_total = sub.statistics['runoff']
                curr_peak = sub.statistics['peak_runoff_rate']
                
                red_total = (1 - curr_total / base_total) * 100
                red_peak = (1 - curr_peak / base_peak) * 100
                
                results.append({
                    'LID_Type': lid_type.name,
                    'Ratio': ratio,
                    'Total_Runoff': curr_total,
                    'Peak_Runoff': curr_peak,
                    'Red_Total': red_total,
                    'Red_Peak': red_peak
                })
        except Exception as e:
            print(f"  Simulation failed at {ratio}%: {e}")
            break
            
    # Cleanup temps
    for f in [temp_inp, temp_controls]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    for ext in ['.rpt', '.out']:
        if os.path.exists(temp_inp.replace('.inp', ext)):
            try: os.remove(temp_inp.replace('.inp', ext))
            except: pass
            
    return results

def main():
    args = parse_arguments()
    inp_file_path = args.inp_file
    
    if not os.path.exists(inp_file_path):
        print(f"Error: Input file {inp_file_path} not found.")
        return

    # 1. Baseline Simulation
    print("Running Baseline Simulation...")
    baseline_simulator = SWMMSimulator(inp_file_path, logger=logger)
    baseline_results = baseline_simulator.run_simulation(show_progress=False)
    
    if not baseline_results.subcatchments:
        print("Error: No subcatchments found.")
        return
        
    target_sub = baseline_results.subcatchments[0]
    print(f"Target Subcatchment: {target_sub.id} (Area: {target_sub.area_m2:.1f} m2)")
    
    # Get precise baseline
    with Simulation(inp_file_path) as sim:
        sub = Subcatchments(sim)[target_sub.id]
        for step in sim: pass
        base_total = sub.statistics['runoff']
        base_peak = sub.statistics['peak_runoff_rate']
        
    print(f"Baseline: Total={base_total:.4f}, Peak={base_peak:.4f}")
    
    # 2. Iterate ALL LID Types
    all_results = []
    
    # Debug: Check LID Types available
    print(f"\nFound {len(LIDType)} LID Types in LIDType enum.")
    
    for lid_type in LIDType:
        print(f"\nChecking LID Type: {lid_type.name}")
        # Check if default parameters exist for this type
        temp_mgr = LIDManager()
        if lid_type not in temp_mgr.DEFAULT_PARAMETERS:
            print(f"  WARNING: Skipping {lid_type.name}: No default parameters defined in LIDManager.")
            continue
            
        lid_results = run_simulation_for_lid(inp_file_path, lid_type, target_sub, base_total, base_peak)
        all_results.extend(lid_results)
        
    # 3. Save Data
    if not all_results:
        print("No results generated!")
        return
        
    df = pd.DataFrame(all_results)
    output_csv = "all_lid_sensitivity_data.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nAll simulation data saved to {output_csv}")
    
    # 4. Visualization (Comparative Plot)
    visualize_comparison(df, target_sub.id)

def visualize_comparison(df, sub_id):
    print("\nGenerating Comparative Visualization...")
    sns.set_theme(style="whitegrid", palette="tab10")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    # Plot 1: Total Runoff Reduction
    sns.lineplot(data=df, x='Ratio', y='Red_Total', hue='LID_Type', style='LID_Type', 
                 markers=True, dashes=False, linewidth=2, markersize=6, ax=axes[0])
    axes[0].set_title('Total Runoff Reduction by LID Type', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('LID Area (%)', fontsize=14)
    axes[0].set_ylabel('Reduction Rate (%)', fontsize=14)
    axes[0].set_xlim(0, 100)
    # axes[0].set_ylim(bottom=0)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Plot 2: Peak Runoff Reduction
    sns.lineplot(data=df, x='Ratio', y='Red_Peak', hue='LID_Type', style='LID_Type', 
                 markers=True, dashes=False, linewidth=2, markersize=6, ax=axes[1])
    axes[1].set_title('Peak Runoff Reduction by LID Type', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('LID Area (%)', fontsize=14)
    axes[1].set_xlim(0, 100)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.suptitle(f'LID Sensitivity Comparison (Subcatchment: {sub_id})', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_png = "all_lid_comparison_result.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_png}")

if __name__ == "__main__":
    main()
