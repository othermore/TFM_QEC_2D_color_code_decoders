import numpy as np
import argparse
import os
import pandas as pd
from simulator import run_color_code_simulations
from plotter import plot_threshold, plot_execution_time, calculate_and_save_threshold, plot_threshold_zoom
import config

DECODER_NAME = 'projection'

def run_simulate(distances, execution_mode, allowed_noise_types):
    p_rates = config.get_config(DECODER_NAME, 'p_rates')
    max_shots = config.get_config(DECODER_NAME, 'max_shots')
    max_errors = config.get_config(DECODER_NAME, 'max_errors')
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    for noise_type in allowed_noise_types:
        output_file = f"s3_projection_results_{noise_type}.csv" if noise_type != 'depolarizing' else "s3_projection_results.csv"
        
        print(f"\n--- Running SIMULATE for noise type: {noise_type.upper()} ---")
        run_color_code_simulations(
            distances=distances,
            p_rates=p_rates,
            decoder_name=DECODER_NAME,
            output_file=output_file,
            rounds_func=lambda d: 1, # Code capacity
            max_shots=max_shots,
            max_errors=max_errors,
            execution_mode=execution_mode,
            noise_type=noise_type
        )

def run_plot(distances_filter, allowed_noise_types):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    for noise_type in allowed_noise_types:
        output_file = f"s3_projection_results_{noise_type}.csv" if noise_type != 'depolarizing' else "s3_projection_results.csv"
        csv_path = os.path.join(project_root, "data", "chapter_6", output_file)
        
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Run 'simulate' first.")
            continue
            
        title_suffix = "" if noise_type == 'depolarizing' else " - Pure X Noise"
        
        print(f"\nGenerating Threshold Plot for {noise_type}...")
        plot_threshold(csv_path, title=f"Logical noise vs physical noise - 4.8.8 Color Code (Projection Decoder{title_suffix})", distances_filter=distances_filter)
        
        print(f"Generating Execution Time Plot for {noise_type}...")
        plot_execution_time(csv_path, title=f"Decoding Complexity - Projection Decoder{title_suffix}", distances_filter=distances_filter)
        
def run_zoom(distances, execution_mode, allowed_noise_types):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    zoom_max_shots = config.get_config(DECODER_NAME, 'zoom_max_shots')
    zoom_max_errors = config.get_config(DECODER_NAME, 'zoom_max_errors')
    zoom_range_factor = config.get_config(DECODER_NAME, 'zoom_range_factor')
    zoom_points = config.get_config(DECODER_NAME, 'zoom_points')
    
    for noise_type in allowed_noise_types:
        base_output_file = f"s3_projection_results_{noise_type}.csv" if noise_type != 'depolarizing' else "s3_projection_results.csv"
        zoom_output_file = base_output_file.replace('.csv', '_zoom.csv')
        
        base_csv_path = os.path.join(project_root, "data", "chapter_6", base_output_file)
        
        if not os.path.exists(base_csv_path):
            print(f"Error: Base file {base_csv_path} not found. Cannot calculate threshold for zoom.")
            continue
            
        # Calculate threshold from the base file
        df = pd.read_csv(base_csv_path)
        # Use only the specified distances to calculate the threshold for the zoom
        df_filtered = df[df['distance'].isin(distances)]
        filtered_distances_list = sorted(df_filtered['distance'].unique())
        
        if len(filtered_distances_list) < 2:
             print(f"Not enough distances found in {base_csv_path} to calculate threshold for zoom.")
             continue
             
        est_threshold = calculate_and_save_threshold(base_csv_path, f'PROJECTION_{noise_type.upper()}', filtered_distances_list, df_filtered)
        
        if np.isnan(est_threshold):
            print(f"Could not estimate threshold for {noise_type}. Skipping zoom.")
            continue
            
        zoom_p_rates = np.linspace(zoom_range_factor[0] * est_threshold, zoom_range_factor[1] * est_threshold, zoom_points)
        
        print(f"\n--- Running ZOOM for noise type: {noise_type.upper()} around threshold {est_threshold:.4f} ---")
        run_color_code_simulations(
            distances=distances,
            p_rates=zoom_p_rates,
            decoder_name=DECODER_NAME,
            output_file=zoom_output_file,
            rounds_func=lambda d: 1,
            max_shots=zoom_max_shots,
            max_errors=zoom_max_errors,
            execution_mode=execution_mode,
            noise_type=noise_type
        )
        
        title_suffix = "" if noise_type == 'depolarizing' else " - Pure X Noise"
        plot_threshold_zoom(base_csv_path, est_threshold, title=f"Logical noise vs physical noise - 4.8.8 Color Code (Projection Decoder - Zoom{title_suffix})", distances_filter=distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection Decoder simulation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Common arguments
    def add_common_args(subparser, include_exec_modes=True):
        subparser.add_argument('--distances', nargs='+', type=int, help="Override distances to process.")
        subparser.add_argument('--noise-type', choices=['all', 'depolarizing', 'X'], default='all', help="Run only for specific noise type.")
        if include_exec_modes:
            group = subparser.add_mutually_exclusive_group()
            group.add_argument('--force-rerun', action='store_true', help="Overwrite data for requested parameters.")
            group.add_argument('--clean-all', action='store_true', help="Delete the target CSV completely before running.")
            
    parser_sim = subparsers.add_parser('simulate', help='Run standard simulations.')
    add_common_args(parser_sim)
    
    parser_plot = subparsers.add_parser('plot', help='Regenerate threshold and time plots.')
    add_common_args(parser_plot, include_exec_modes=False)
    
    parser_zoom = subparsers.add_parser('zoom', help='Run high-resolution zoom simulations.')
    add_common_args(parser_zoom)
    
    args = parser.parse_args()
    
    distances = args.distances if args.distances else config.get_config(DECODER_NAME, 'distances')
    allowed_noise_types = ['depolarizing', 'X'] if args.noise_type == 'all' else [args.noise_type]
    
    execution_mode = 'default'
    if hasattr(args, 'force_rerun') and args.force_rerun:
        execution_mode = 'force-rerun'
    elif hasattr(args, 'clean_all') and args.clean_all:
        execution_mode = 'clean-all'
        
    print("=====================================================")
    print("      Projection Decoder Simulation - 4.8.8 Color Code   ")
    print("=====================================================")
    
    if args.command == 'simulate':
        run_simulate(distances, execution_mode, allowed_noise_types)
    elif args.command == 'plot':
        run_plot(distances, allowed_noise_types)
    elif args.command == 'zoom':
        run_zoom(distances, execution_mode, allowed_noise_types)
