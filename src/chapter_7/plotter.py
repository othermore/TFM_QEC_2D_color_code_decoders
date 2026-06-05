import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_thresholds():
    df = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_1_thresholds.csv'))
    
    decoders = df['decoder'].values
    labels = [config.DECODER_LABELS[d] for d in decoders]
    
    x = np.arange(len(decoders))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, df['threshold_x'], width, label='Bit-flip (X) Noise', color='#1f77b4')
    rects2 = ax.bar(x + width/2, df['threshold_dep'], width, label='Depolarizing Noise', color='#ff7f0e')
    
    ax.set_ylabel('Threshold ($p_{th}$)')
    ax.set_title('Threshold Comparison Across Decoders')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_CH7_DIR, 'plot_1_thresholds.png'), dpi=300)
    plt.close()
    print("✓ plot_1_thresholds.png")

def plot_mitigation():
    df = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_2_mitigation.csv'))
    
    for d in config.DISTANCES_MITIGATION:
        df_d = df[df['distance'] == d]
        if df_d.empty: continue
            
        plt.figure(figsize=(9, 6))
        
        for decoder in config.DECODERS:
            df_dec = df_d[df_d['decoder'] == decoder].sort_values('p_physical')
            if df_dec.empty: continue
                
            plt.plot(df_dec['p_physical'], df_dec['mitigation_ratio'], 
                     marker='o', linestyle='-', color=config.COLORS[decoder], 
                     label=config.DECODER_LABELS[decoder])
                     
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.8, label='Break-even (Ratio = 1)')
        
        plt.xscale('log')
        plt.yscale('log')
        
        # Explicit X-axis ticks that match the data range (0.02 to 0.20)
        p_ticks = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
        plt.xticks(p_ticks, [str(p) for p in p_ticks])
        
        # Explicit Y-axis ticks dynamically matched to data
        ymin, ymax = df_d['mitigation_ratio'].min(), df_d['mitigation_ratio'].max()
        min_pow = int(np.floor(np.log10(ymin))) if ymin > 0 else 0
        max_pow = int(np.ceil(np.log10(ymax))) if ymax > 0 else 5
        y_powers = np.arange(min_pow, max_pow + 1)
        plt.yticks(10.0**y_powers, [f'$10^{{{int(i)}}}$' for i in y_powers])
        
        plt.xlabel('Physical Error Rate ($p$)')
        plt.ylabel('Mitigation Capacity ($p_{phys} / p_L$)')
        plt.title(f'Noise Mitigation Capacity (Distance $d={d}$)')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which="both", ls=":", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.FIGURES_CH7_DIR, f'plot_2_mitigation_d{d}.png'), dpi=300)
        plt.close()
    print("✓ plot_2_mitigation_*.png")

def plot_error_scaling():
    df_fit = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_3_error_scaling.csv'))
    
    # We will use p_phys = 0.02 since ALL decoders simulated this exact point.
    P_FIXED = 0.02
    log_p = np.log10(P_FIXED)
    
    plt.figure(figsize=(9, 6))
    
    # Range of d to draw the fitted line, extended up to d=25
    d_range = np.linspace(3, 25, 100)
    
    for decoder in config.DECODERS:
        row_fit = df_fit[df_fit['decoder'] == decoder]
        if row_fit.empty: continue
            
        c0, c1, c2, c3 = row_fit['c0'].iloc[0], row_fit['c1'].iloc[0], row_fit['c2'].iloc[0], row_fit['c3'].iloc[0]
        
        # We evaluate the mathematical line EXACTLY at P_FIXED
        log_pL_fit = c0 + c1*d_range + c2*log_p + c3*d_range*log_p
        pL_fit = 10**log_pL_fit
        
        # Simplification of the 2D equation to a 1D equation in d for this specific p_phys
        beta_0 = c0 + c2 * log_p
        beta_1 = c1 + c3 * log_p
        
        # Format the label with the equation
        sign = "+" if beta_1 >= 0 else "-"
        fit_label = f"{config.DECODER_LABELS[decoder]}: $10^{{{beta_0:.2f} {sign} {abs(beta_1):.2f}d}}$"
        
        plt.plot(d_range, pL_fit, linestyle='--', color=config.COLORS[decoder], alpha=0.5, label=fit_label)
        
        # Extract real data to plot as points
        path = os.path.join(config.DATA_CH6_DIR, f"{config.DECODER_PREFIXES[decoder]}_results_X.csv")
        if os.path.exists(path):
            df_real = pd.read_csv(path)
            # Extract exactly the simulated point at 0.02 (with floating margin)
            df_points = df_real[(np.isclose(df_real['p_physical'], P_FIXED)) & (df_real['p_logical'] > 0)]
            if not df_points.empty:
                plt.scatter(df_points['distance'], df_points['p_logical'], 
                            color=config.COLORS[decoder], 
                            s=50, zorder=5)
                            
    plt.yscale('log')
    plt.xlabel('Code Distance ($d$)')
    plt.ylabel('Logical Error Rate ($p_L$)')
    plt.title(f'Sub-Threshold Error Scaling Extrapolation ($p_{{phys}} \\approx {P_FIXED}$)')
    
    # Fix X-axis to only show integer values
    ax = plt.gca()
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Put legend outside to avoid hiding data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_CH7_DIR, 'plot_3_error_scaling.png'), dpi=300)
    plt.close()
    print("✓ plot_3_error_scaling.png")

def plot_time_scaling():
    df_fit = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_4_time_scaling.csv'))
    
    plt.figure(figsize=(9, 6))
    
    d_range_line = np.linspace(3, 25, 100)
    n_range = (3 * d_range_line**2 + 1) / 4
    
    for decoder in config.DECODERS:
        row_fit = df_fit[df_fit['decoder'] == decoder]
        if row_fit.empty: continue
            
        k = row_fit['exponent_k'].iloc[0]
        a = row_fit['multiplier_a'].iloc[0]
        c = row_fit['constant_c'].iloc[0]
        
        path = os.path.join(config.DATA_CH6_DIR, f"{config.DECODER_PREFIXES[decoder]}_results_X.csv")
        if os.path.exists(path):
            df_real = pd.read_csv(path)
            avg_times = df_real.groupby('distance')['avg_time_sec'].mean().reset_index()
            # N = (3d^2+1)/4
            qubits = (3 * avg_times['distance']**2 + 1) / 4
            
            plt.scatter(qubits, avg_times['avg_time_sec'], color=config.COLORS[decoder], s=50, zorder=5)
            
            # Draw the fitted line
            t_fit = c + a * (n_range**k)
            
            plt.plot(n_range, t_fit, color=config.COLORS[decoder], linestyle='--', 
                     label=f"{config.DECODER_LABELS[decoder]} $\\sim \\mathcal{{O}}(N^{{{k:.2f}}})$")
                     
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Qubits ($N$)')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Time Complexity Scaling')
    
    # Custom X-axis labels for specific distances
    d_ticks = np.arange(5, 25, 2)
    n_ticks = (3 * d_ticks**2 + 1) / 4
    ax = plt.gca()
    ax.set_xticks(n_ticks)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels([int(n) for n in n_ticks], rotation=45)
    
    plt.legend()
    plt.grid(True, which="both", ls=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_CH7_DIR, 'plot_4_time_scaling.png'), dpi=300)
    plt.close()
    print("✓ plot_4_time_scaling.png")

def plot_extrapolations():
    df = pd.read_csv(os.path.join(config.DATA_CH7_DIR, 'analysis_5_6_extrapolations.csv'))
    
    # We plot distance for target pL=1e-12
    target_pL = 1e-12
    df_d = df[np.isclose(df['target_p_logical'], target_pL)]
    
    if df_d.empty:
        return
        
    p1 = config.P_PHYS_TARGETS[0]
    p2 = config.P_PHYS_TARGETS[1]
    
    # Split decoders into optimized C++ libraries and Python
    opt_decoders = ['bposd', 'chromobius']
    py_decoders = ['projection', 'restriction', 'concat_mwpm', 'correlated']
    
    # visual gap between the two groups
    ordered_decoders = opt_decoders + py_decoders
    labels = [config.DECODER_LABELS[d] for d in ordered_decoders]
    
    # Generate X positions with a gap of 1 unit between the two groups
    x_opt = np.arange(len(opt_decoders))
    x_py = np.arange(len(py_decoders)) + len(opt_decoders) + 0.8
    x = np.concatenate([x_opt, x_py])
    
    width = 0.35
    
    d_p1 = [df_d[(df_d['decoder'] == d) & (np.isclose(df_d['fixed_p_physical'], p1))]['required_distance'].iloc[0] for d in ordered_decoders]
    d_p2 = [df_d[(df_d['decoder'] == d) & (np.isclose(df_d['fixed_p_physical'], p2))]['required_distance'].iloc[0] for d in ordered_decoders]
    
    # Colors for the extrapolations
    color_p1 = '#2b5c8f'
    color_p2 = '#d95f02'
    
    # Plot 5: Distances
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, d_p1, width, label=f'$p_{{phys}} = 10^{{{int(np.log10(p1))}}}$', color=color_p1)
    rects2 = ax.bar(x + width/2, d_p2, width, label=f'$p_{{phys}} = 10^{{{int(np.log10(p2))}}}$', color=color_p2)
    
    # Custom Y-axis labels for odd distances
    max_d = int(max(max(d_p1), max(d_p2)))
    ax.set_ylim(0, max_d + 6)
    ax.set_yticks(np.arange(3, max_d + 6, 2))
    
    # Add vertical separator and text (now with guaranteed space)
    ax.axvline(x=len(opt_decoders) - 0.5 + 0.4, color='black', linestyle='-.', alpha=0.5)
    ax.text((len(opt_decoders)-1)/2, max_d + 2, 'Optimized C++\nLibraries', ha='center', va='bottom', fontsize=9, fontweight='bold', alpha=0.8)
    ax.text(len(opt_decoders) + 0.8 + (len(py_decoders)-1)/2, max_d + 2, 'Python Overhead\nWrappers', ha='center', va='bottom', fontsize=9, fontweight='bold', alpha=0.8)
    
    ax.set_ylabel('Required Code Distance ($d$)')
    ax.set_title(f'Required Distance to reach $p_L = 10^{{{int(np.log10(target_pL))}}}$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_CH7_DIR, 'plot_5_required_distances.png'), dpi=300)
    plt.close()
    print("✓ plot_5_required_distances.png")
    
    # Plot 6: Times (in microseconds, with FPGA hardware acceleration assumption)
    # Assuming a specialized FPGA/ASIC is 100x faster (1% of Python CPU time)
    FPGA_SPEEDUP_FACTOR = 0.01
    
    t_p1 = [df_d[(df_d['decoder'] == d) & (np.isclose(df_d['fixed_p_physical'], p1))]['estimated_time_sec_per_shot'].iloc[0] * 1e6 * FPGA_SPEEDUP_FACTOR for d in ordered_decoders]
    t_p2 = [df_d[(df_d['decoder'] == d) & (np.isclose(df_d['fixed_p_physical'], p2))]['estimated_time_sec_per_shot'].iloc[0] * 1e6 * FPGA_SPEEDUP_FACTOR for d in ordered_decoders]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, t_p1, width, label=f'$p_{{phys}} = 10^{{{int(np.log10(p1))}}}$', color=color_p1)
    rects2 = ax.bar(x + width/2, t_p2, width, label=f'$p_{{phys}} = 10^{{{int(np.log10(p2))}}}$', color=color_p2)
    
    max_t = max(max(t_p1), max(t_p2))
    ax.set_ylim(0, max_t * 1.25)
    
    # Add vertical separator and text
    ax.axvline(x=len(opt_decoders) - 0.5 + 0.4, color='black', linestyle='-.', alpha=0.5)
    ax.text((len(opt_decoders)-1)/2, max_t * 1.05, 'Optimized C++\nLibraries', ha='center', va='bottom', fontsize=9, fontweight='bold', alpha=0.8)
    ax.text(len(opt_decoders) + 0.8 + (len(py_decoders)-1)/2, max_t * 1.05, 'Python Overhead\nWrappers', ha='center', va='bottom', fontsize=9, fontweight='bold', alpha=0.8)
    
    # Shade the hardware budget area (0.1 us to 1.0 us for superconducting qubits)
    ax.axhspan(0.1, 1.0, color='green', alpha=0.15, label='SC Hardware Budget\n(100ns - 1µs)')
    
    ax.set_ylabel('Estimated Time per Shot (µs)\n[Assuming 100x FPGA Speedup]')
    ax.set_title(f'Estimated Decoding Time for $p_L = 10^{{{int(np.log10(target_pL))}}}$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    
    ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.65))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_CH7_DIR, 'plot_6_required_times.png'), dpi=300)
    plt.close()
    print("✓ plot_6_required_times.png")

def run_all_plots():
    print("Generando graficas - Capitulo 7...")
    plot_thresholds()
    plot_mitigation()
    plot_error_scaling()
    plot_time_scaling()
    plot_extrapolations()
    print("Graficas completadas.")

if __name__ == '__main__':
    run_all_plots()
