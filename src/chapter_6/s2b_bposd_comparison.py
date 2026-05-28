import os
import csv
import time
import numpy as np
import sinter
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from s1_color_code import ColorCode
from ldpc import SinterBpOsdDecoder

def run_comparison(
    distance=7, 
    p_rates=None, 
    max_shots=1000000, 
    max_errors=300,
    output_file="s2b_bposd_comparison.csv",
    plot_only=False
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(project_root, "data", "chapter_6", output_file)
    
    bp_methods = ['ms', 'ps']
    osd_methods = ['osd_0', 'osd_e', 'osd_cs']
    osd_orders = [5, 10, 20]
    
    # Generate exactly 12 custom decoders
    custom_decoders: Dict[str, sinter.Decoder] = {}
    for bp in bp_methods:
        for osd in osd_methods:
            if osd == 'osd_0':
                orders = [0]
            elif osd == 'osd_e':
                orders = [5]
            else:
                orders = osd_orders
            for order in orders:
                name = f"bposd_{bp}_{osd}_o{order}"
                custom_decoders[name] = SinterBpOsdDecoder(
                    max_iter=30,
                    bp_method=bp,
                    osd_method=osd,
                    osd_order=order
                )
    
    if not plot_only:
        if p_rates is None:
            p_rates = np.linspace(0.01, 0.20, 15)
            
        print(f"Generating circuit and DEM for distance {distance}...")
        cc = ColorCode(distance=distance)
        
        tasks = []
        for p in p_rates:
            circ = cc.get_stim_circuit(rounds=1, p_data=p, p_meas=0.0, p_gate=0.0)
            dem = circ.detector_error_model(decompose_errors=False)
            tasks.append(
                sinter.Task(
                    circuit=circ,
                    detector_error_model=dem,
                    json_metadata={'d': distance, 'p': p, 'rounds': 1}
                )
            )
            
        print(f"Starting Sinter collect for {len(custom_decoders)} decoders...")
        start_time = time.time()
        
        # Windows compatibility for multiprocessing in scripts
        from multiprocessing import freeze_support
        freeze_support()
        
        stats = sinter.collect(
            num_workers=os.cpu_count() or 4,
            tasks=tasks,
            decoders=list(custom_decoders.keys()),
            custom_decoders=custom_decoders,
            max_shots=max_shots,
            max_errors=max_errors,
            print_progress=True
        )
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["decoder", "distance", "rounds", "p_physical", "p_logical", "p_logical_err", "shots", "errors", "avg_time_sec"])
            for s in stats:
                d = s.json_metadata['d']
                r = s.json_metadata['rounds']
                p = s.json_metadata['p']
                p_logical = s.errors / s.shots if s.shots > 0 else 0.0
                p_err = np.sqrt(p_logical * (1 - p_logical) / s.shots) if s.shots > 0 else 0.0
                writer.writerow([s.decoder, d, r, p, p_logical, p_err, s.shots, s.errors, s.seconds / s.shots if s.shots > 0 else 0.0])
                
    plot_results(csv_path)

def plot_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))), "figures", "chapter_6")
    os.makedirs(base_path, exist_ok=True)
    
    # Removed fixed dictionaries for colors, markers, and linestyles
    
    def make_plot(subset_decoders, title_suffix, filename_suffix):
        fig_err, ax_err = plt.subplots(figsize=(10, 6))
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        
        plotted_any = False
        for decoder_name in subset_decoders:
            if decoder_name not in df['decoder'].unique():
                continue
            plotted_any = True
            df_dec = df[df['decoder'] == decoder_name].sort_values('p_physical')
            parts = decoder_name.split('_')
            bp = parts[1]
            osd = parts[2] + '_' + parts[3]
            order = int(parts[4][1:])
            
            label = f"BP:{bp.upper()} OSD:{osd.upper().replace('OSD_', '')} Ord:{order}"
            
            p = ax_err.errorbar(
                df_dec['p_physical'], df_dec['p_logical'], 
                yerr=df_dec['p_logical_err'], 
                label=label, marker='o', linestyle='-', markersize=6, alpha=0.8
            )
            
            ax_time.plot(
                df_dec['p_physical'], df_dec['avg_time_sec'], 
                label=label, color=p[0].get_color(), marker='o', linestyle='-', markersize=6, alpha=0.8
            )
            
        if not plotted_any:
            plt.close(fig_err)
            plt.close(fig_time)
            return

        # Error plot formatting
        ax_err.set_yscale('log')
        ax_err.set_xscale('log')
        ax_err.set_xlabel('Physical Error Rate ($p$)')
        ax_err.set_ylabel('Logical Error Rate ($P_L$)')
        ax_err.set_title(f'BP+OSD (d=7) - {title_suffix} - Logical Error Rate')
        ax_err.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_err.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_err.tight_layout()
        
        # Time plot formatting
        ax_time.set_yscale('linear')
        ax_time.set_xscale('log')
        ax_time.set_xlabel('Physical Error Rate ($p$)')
        ax_time.set_ylabel('Average Execution Time per Shot (s)')
        ax_time.set_title(f'BP+OSD (d=7) - {title_suffix} - Execution Time')
        ax_time.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_time.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig_time.tight_layout()
        
        path_err = os.path.join(base_path, f"s2b_bposd_{filename_suffix}_error.png")
        path_time = os.path.join(base_path, f"s2b_bposd_{filename_suffix}_time.png")
        
        fig_err.savefig(path_err, dpi=300, bbox_inches='tight')
        fig_time.savefig(path_time, dpi=300, bbox_inches='tight')
        plt.close(fig_err)
        plt.close(fig_time)
        print(f"Saved: {path_err}")
        print(f"Saved: {path_time}")

    # Plot 1: BP methods (ms vs ps)
    make_plot(
        ['bposd_ms_osd_cs_o10', 'bposd_ps_osd_cs_o10'], 
        'BP Methods (MS vs PS)', 
        'bp_methods'
    )
    
    # Plot 2: OSD methods (0 vs e vs cs)
    make_plot(
        ['bposd_ps_osd_0_o0', 'bposd_ps_osd_e_o5', 'bposd_ps_osd_cs_o5'], 
        'OSD Methods (0 vs E vs CS)', 
        'osd_methods'
    )
    
    # Plot 3: OSD_CS Orders (5 vs 10 vs 20)
    make_plot(
        ['bposd_ps_osd_cs_o5', 'bposd_ps_osd_cs_o10', 'bposd_ps_osd_cs_o20'], 
        'OSD_CS Orders (5 vs 10 vs 20)', 
        'osd_orders'
    )

if __name__ == '__main__':
    run_comparison()
