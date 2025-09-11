#!/usr/bin/env python
"""
Generate corner plots for the 7-pulse fitted results.
"""

import numpy as np
import matplotlib.pyplot as plt
import anesthetic
import os

def main():
    # Read the chains file
    print("Loading chains from results_7pulses_fitted/chains_dead-birth.txt...")
    
    # Read the column names from the file
    with open('results_7pulses_fitted/chains_dead-birth.txt', 'r') as f:
        header = f.readline().strip()
        columns = header.split()
    
    print(f"Found {len(columns)} columns")
    print(f"Columns: {columns[:10]}...")  # Show first 10
    
    # Load the data manually since the header is in the first row
    data = np.loadtxt('results_7pulses_fitted/chains_dead-birth.txt', skiprows=1)
    
    # Create anesthetic NestedSamples object
    # The last 3 columns are logL, logL_birth, nlive
    samples = anesthetic.NestedSamples(
        data=data[:, :-3],
        logL=data[:, -3],
        logL_birth=data[:, -2],
        columns=columns[:-3]  # Exclude logL, logL_birth, nlive from parameter columns
    )
    
    print(f"Loaded {len(samples)} samples")
    
    # Get weighted statistics for key parameters
    print("\nWeighted statistics:")
    if '$N_{\\text{pulse}}$' in samples.columns:
        npulse_mean = samples['$N_{\\text{pulse}}$'].mean()
        npulse_std = samples['$N_{\\text{pulse}}$'].std()
        print(f"Npulse: {npulse_mean:.2f} ± {npulse_std:.2f}")
    
    # Create output directory for plots
    os.makedirs('results_7pulses_fitted/plots', exist_ok=True)
    
    # 1. Corner plot for amplitudes (first 7 non-zero amplitudes)
    print("\nCreating amplitude corner plot...")
    amp_cols = [f'$A_{{{i}}}$' for i in range(1, 11)]
    
    # Find which amplitudes are active (non-zero)
    active_amps = []
    for col in amp_cols:
        if col in samples.columns:
            if samples[col].mean() > 0.1:  # Only include if mean > 0.1
                active_amps.append(col)
    
    print(f"Active amplitudes: {active_amps}")
    
    if len(active_amps) > 0:
        axes = samples.plot_2d(active_amps[:7])  # Limit to 7 for visibility
        plt.suptitle('Amplitude Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_amplitudes.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_amplitudes.png")
    
    # 2. Corner plot for tau parameters (corresponding to active amplitudes)
    print("\nCreating tau corner plot...")
    tau_cols = []
    for i, amp_col in enumerate(active_amps[:7], 1):
        tau_col = f'$\\tau_{{{i}}}$'
        if tau_col in samples.columns:
            tau_cols.append(tau_col)
    
    if len(tau_cols) > 0:
        axes = samples.plot_2d(tau_cols)
        plt.suptitle('Tau (Decay) Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_tau.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_tau.png")
    
    # 3. Corner plot for arrival times (u parameters)
    print("\nCreating arrival time corner plot...")
    u_cols = []
    for i in range(1, 8):  # First 7
        u_col = f'$u_{{{i}}}$'
        if u_col in samples.columns:
            u_cols.append(u_col)
    
    if len(u_cols) > 0:
        axes = samples.plot_2d(u_cols)
        plt.suptitle('Arrival Time Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_arrival_times.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_arrival_times.png")
    
    # 4. Corner plot for width parameters
    print("\nCreating width corner plot...")
    w_cols = []
    for i in range(1, 8):  # First 7
        w_col = f'$w_{{{i}}}$'
        if w_col in samples.columns:
            w_cols.append(w_col)
    
    if len(w_cols) > 0:
        axes = samples.plot_2d(w_cols)
        plt.suptitle('Width Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_widths.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_widths.png")
    
    # 5. Corner plot for Npulse and sigma
    print("\nCreating Npulse and sigma corner plot...")
    global_params = []
    if '$N_{\\text{pulse}}$' in samples.columns:
        global_params.append('$N_{\\text{pulse}}$')
    if '$\\sigma$' in samples.columns:
        global_params.append('$\\sigma$')
    
    if len(global_params) > 0:
        axes = samples.plot_2d(global_params)
        plt.suptitle('Global Parameters (Npulse and Sigma)', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_global.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_global.png")
    
    # 6. Create a summary plot showing the first few active pulses together
    print("\nCreating summary corner plot...")
    summary_params = []
    
    # Add first 3 of each type for a manageable plot
    for i in range(1, 4):
        for param_type, symbol in [('A', 'A'), ('tau', '\\tau'), ('u', 'u'), ('w', 'w')]:
            if param_type == 'tau':
                col = f'${symbol}_{{{i}}}$'
            else:
                col = f'${param_type}_{{{i}}}$'
            if col in samples.columns:
                # Check if parameter is meaningful (not just noise)
                if param_type == 'A' and samples[col].mean() > 0.1:
                    summary_params.append(col)
                elif param_type != 'A':
                    summary_params.append(col)
    
    # Add global parameters
    summary_params.extend(global_params)
    
    if len(summary_params) > 0:
        # Limit to 15 parameters for readability
        axes = samples.plot_2d(summary_params[:15])
        plt.suptitle('Summary: First 3 Pulses + Global Parameters', fontsize=14)
        plt.tight_layout()
        plt.savefig('results_7pulses_fitted/plots/corner_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved corner_summary.png")
    
    print("\n=== All corner plots saved to results_7pulses_fitted/plots/ ===")
    print("Generated plots:")
    print("  - corner_amplitudes.png")
    print("  - corner_tau.png")
    print("  - corner_arrival_times.png")
    print("  - corner_widths.png")
    print("  - corner_global.png")
    print("  - corner_summary.png")

if __name__ == "__main__":
    main()