import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# Add project root to path (assuming we run from root)
sys.path.append(os.getcwd())

try:
    from reference_code.lid_manager_ref import LIDManager, LIDType, LIDConfiguration
    from reference_code.swmm_simulator_ref import SWMMSimulator
except ImportError as e:
    print(f"Error importing reference code: {e}")
    # Fallback for when running directly inside reference_code or similar
    try:
        sys.path.append(os.path.join(os.getcwd(), 'reference_code'))
        import lid_manager_ref
        import swmm_simulator_ref
        LIDManager = lid_manager_ref.LIDManager
        LIDType = lid_manager_ref.LIDType
        LIDConfiguration = lid_manager_ref.LIDConfiguration
        SWMMSimulator = swmm_simulator_ref.SWMMSimulator
    except Exception as e2:
        print(f"Critical Import Error: {e2}")
        sys.exit(1)

# Configure Logging
log_filename = "lid_simulation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Comprehensive LID Analysis (ex1.py)...")

    # 1. Configuration
    inp_file = os.path.join("inp_file", "seocho2.inp")
    if not os.path.exists(inp_file):
        logger.error(f"Error: Input file {inp_file} not found.")
        return

    # Custom Parameters for Permeable Pavement
    pp_custom_params = {
        "surface": [0.1, 0.0, 0.1, 1.0, 5],
        "pavement": [150, 0.15, 0, 100, 0],
        "storage": [300, 0.3, 0.5, 0],
        "drain": [0, 0.5, 0, 6, 0, 0],
        "soil": None,
        "drainmat": None
    }

    # 2. Baseline Simulation (0% LID) - Run once
    logger.info("\nRunning Baseline Simulation...")
    # Use a temporary manager just to get baseline
    temp_manager = LIDManager(logger=logger)
    baseline_simulator = SWMMSimulator(inp_file, logger=logger)
    baseline_results = baseline_simulator.run_simulation(show_progress=False)
    
    # Identify target subcatchment (highest runoff)
    target_sub = baseline_results.subcatchments[0]
    logger.info(f"Target Subcatchment: {target_sub.id} (Area: {target_sub.area_m2:.1f} m2)")

    # Get precise baseline values
    from pyswmm import Simulation, Subcatchments
    with Simulation(inp_file) as sim:
        sub = Subcatchments(sim)[target_sub.id]
        for step in sim: pass
        base_peak_runoff = sub.statistics['peak_runoff_rate']
        base_total_runoff = sub.statistics['runoff']
        
    logger.info(f"Baseline: Total={base_total_runoff:.4f}, Peak={base_peak_runoff:.4f}")

    # 3. Iterate over ALL LID Types
    for lid_type in LIDType:
        logger.info(f"\n" + "="*60)
        logger.info(f"Processing LID Type: {lid_type.display_name} ({lid_type.code})")
        logger.info("="*60)
        
        # Initialize Manager for this run
        lid_manager = LIDManager(logger=logger)
        
        # Determine parameters
        current_params = None
        if lid_type == LIDType.PERMEABLE_PAVEMENT:
            current_params = pp_custom_params
            logger.info("Using Custom Parameters for PP")
        else:
            logger.info("Using Default Parameters")
            
        # Create Configuration
        lid_config = lid_manager.create_lid_configuration(
            name=f"LID_{lid_type.code}",
            lid_type=lid_type,
            custom_parameters=current_params
        )
        
        # Results container
        results = []
        # Add baseline (0%)
        results.append({
            'Ratio': 0,
            'Total Runoff': base_total_runoff,
            'Peak Runoff': base_peak_runoff
        })
        
        # Iterate Ratios (1% to 100%)
        ratios = np.arange(1, 101, 1)
        temp_inp = f"temp_{lid_type.code}_ex1.inp"
        temp_controls = f"temp_controls_{lid_type.code}.inp"
        
        for ratio in ratios:
            if ratio % 20 == 0:
                logger.info(f"  Simulating {ratio}%")
                
            lid_manager.reset_placements()
            
            # Calculate area/width
            expected_lid_area = target_sub.impervious_area_m2 * (ratio / 100.0)
            lid_width = np.sqrt(expected_lid_area) if expected_lid_area > 0 else 1.0
            
            lid_manager.create_lid_placement(
                subcatchment_data=target_sub,
                lid_config=lid_config,
                area_percentage=float(ratio),
                width=lid_width
            )
            
            # Generate INP
            lid_manager.add_lid_controls_to_inp(inp_file, temp_controls)
            lid_manager.add_lid_usage_to_inp(temp_controls, temp_inp)
            
            # Run Simulation
            try:
                with Simulation(temp_inp) as sim:
                    sub = Subcatchments(sim)[target_sub.id]
                    for step in sim: pass
                    
                    curr_total = sub.statistics['runoff']
                    curr_peak = sub.statistics['peak_runoff_rate']
                    
                    results.append({
                        'Ratio': ratio,
                        'Total Runoff': curr_total,
                        'Peak Runoff': curr_peak
                    })
            except Exception as e:
                logger.error(f"  Simulation failed at {ratio}%: {e}")
                break
        
        # Cleanup temps
        try:
            if os.path.exists(temp_inp): os.remove(temp_inp)
            if os.path.exists(temp_controls): os.remove(temp_controls)
            for ext in ['.rpt', '.out']:
                if os.path.exists(temp_inp.replace('.inp', ext)):
                    os.remove(temp_inp.replace('.inp', ext))
        except:
            pass
        
        # Analysis & Visualization
        df = pd.DataFrame(results)
        
        if base_total_runoff > 0:
            df['Red_Total'] = (1 - df['Total Runoff'] / base_total_runoff) * 100
        else:
            df['Red_Total'] = 0.0
            
        if base_peak_runoff > 0:
            df['Red_Peak'] = (1 - df['Peak Runoff'] / base_peak_runoff) * 100
        else:
            df['Red_Peak'] = 0.0

        df['Red_Total'] = df['Red_Total'].clip(lower=0)
        df['Red_Peak'] = df['Red_Peak'].clip(lower=0)
        
        # Save Data
        csv_filename = f"{lid_type.code}_data.csv"
        df.to_csv(csv_filename, index=False)
        logger.info(f"  Data saved to {csv_filename}")
        
        # --- Visualization (adapted from visualize_separation.py) ---
        visualize_results(df, lid_type, csv_filename)


def visualize_results(df, lid_type, csv_filename):
    x_data = df['Ratio'].values
    y_total = df['Red_Total'].values
    y_peak = df['Red_Peak'].values

    # Polynomial Regression (2nd Order)
    try:
        z_total = np.polyfit(x_data, y_total, 2)
        p_total = np.poly1d(z_total)
        
        z_peak = np.polyfit(x_data, y_peak, 2)
        p_peak = np.poly1d(z_peak)

        # R2 Calculation
        r2_total = 1 - (np.sum((y_total - p_total(x_data))**2) / np.sum((y_total - np.mean(y_total))**2))
        r2_peak = 1 - (np.sum((y_peak - p_peak(x_data))**2) / np.sum((y_peak - np.mean(y_peak))**2))

        # Smooth lines
        x_smooth = np.linspace(0, 100, 500)
        y_total_smooth = p_total(x_smooth)
        y_peak_smooth = p_peak(x_smooth)
        
        eq_total = f"$y = {z_total[0]:.4f}x^2 + {z_total[1]:.2f}x + ({z_total[2]:.2f})$\n($R^2={r2_total:.4f}$)"
        eq_peak = f"$y = {z_peak[0]:.4f}x^2 + {z_peak[1]:.2f}x + ({z_peak[2]:.2f})$\n($R^2={r2_peak:.4f}$)"
    except:
        # Fallback if fit fails (e.g. all zeros)
        x_smooth = x_data
        y_total_smooth = y_total
        y_peak_smooth = y_peak
        eq_total = "Fit Failed"
        eq_peak = "Fit Failed"

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Total Runoff
    ax1 = axes[0]
    color_total = "#2980b9" # Blue
    ax1.scatter(x_data, y_total, color=color_total, alpha=0.4, s=30, label='Observed Data')
    ax1.plot(x_smooth, y_total_smooth, color=color_total, linewidth=2.5, linestyle='-', label='Trend Line (Poly)')
    
    ax1.set_title(f'Total Runoff Reduction ({lid_type.display_name})', fontsize=16, fontweight='bold', y=1.02)
    ax1.set_xlabel('LID Area (%)', fontsize=12)
    ax1.set_ylabel('Reduction Rate (%)', fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1.text(0.05, 0.95, eq_total, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color_total))
    ax1.legend(loc='lower right')

    # Peak Runoff
    ax2 = axes[1]
    color_peak = "#c0392b" # Red
    ax2.scatter(x_data, y_peak, color=color_peak, alpha=0.4, s=30, label='Observed Data')
    ax2.plot(x_smooth, y_peak_smooth, color=color_peak, linewidth=2.5, linestyle='-', label='Trend Line (Poly)')
    
    ax2.set_title(f'Peak Runoff Reduction ({lid_type.display_name})', fontsize=16, fontweight='bold', y=1.02)
    ax2.set_xlabel('LID Area (%)', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    ax2.text(0.05, 0.95, eq_peak, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color_peak))
    ax2.legend(loc='lower right')

    plt.suptitle(f'{lid_type.display_name} Efficiency Analysis', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_png = f"{lid_type.code}_result.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    logger.info(f"  Graph saved to {output_png}")
    plt.close()

if __name__ == "__main__":
    main()