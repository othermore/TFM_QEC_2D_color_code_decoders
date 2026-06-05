import os
import csv
import sinter
import numpy as np
import stim
from typing import List, Optional
import multiprocessing

from s1_color_code import ColorCode
from decoder_interfaces import get_custom_decoders

def split_Y_errors_by_coords(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """
    Takes a Stim DetectorErrorModel and returns a new one where all composite 
    Y errors are robustly split into independent X and Z error mechanisms.
    
    Decomposes Y errors into X and Z errors by separating their target detectors.
    This relies on the 4th coordinate (`c_idx`) which is embedded in the DETECTOR 
    instruction metadata: c_idx < 3 means X-detector, c_idx >= 3 means Z-detector.
    """
    # First, map each detector ID to its c_idx
    d_to_c_idx = {}
    for instr in dem:
        if instr.type == "detector":
            d = instr.targets_copy()[0].val
            c_idx = int(instr.args_copy()[3])
            d_to_c_idx[d] = c_idx
            
    new_dem = stim.DetectorErrorModel()
    
    # Check if the logical observable belongs to X or Z
    # If the first error with a logical observable triggers predominantly X-detectors, it's a Z-observable.
    obs_in_X = False
    obs_in_Z = False
    for instr in dem:
        if instr.type == "error":
            targets = instr.targets_copy()
            has_obs = any(t.is_logical_observable_id() for t in targets)
            if has_obs:
                has_X = any(d_to_c_idx[t.val] < 3 for t in targets if t.is_relative_detector_id())
                has_Z = any(d_to_c_idx[t.val] >= 3 for t in targets if t.is_relative_detector_id())
                if has_X and not has_Z:
                    obs_in_X = True
                elif has_Z and not has_X:
                    obs_in_Z = True
                    
    for instr in dem:
        if instr.type == "error":
            p = instr.args_copy()[0]
            targets = instr.targets_copy()
            
            t_X = [t for t in targets if t.is_relative_detector_id() and d_to_c_idx[t.val] < 3]
            t_Z = [t for t in targets if t.is_relative_detector_id() and d_to_c_idx[t.val] >= 3]
            obs = [t for t in targets if t.is_logical_observable_id()]
            
            if len(t_X) == 0 or len(t_Z) == 0:
                # It is already a pure X or Z error (or weight-0), keep it as is
                new_dem.append("error", [p], targets)
            elif len(t_X) > 0 and len(t_Z) > 0:
                # Y error split! Only attach the logical observable to the correct graph
                if obs_in_X:
                    new_dem.append("error", [p], t_X + obs)
                    new_dem.append("error", [p], t_Z)
                else:
                    new_dem.append("error", [p], t_X)
                    new_dem.append("error", [p], t_Z + obs)
        else:
            new_dem.append(instr)
            
    return new_dem

def run_color_code_simulations(
    distances: List[int],
    p_rates: np.ndarray,
    decoder_name: str,
    output_file: str,
    rounds_func=lambda d: d,
    max_shots: int = 1_000_000,
    max_errors: int = 500,
    num_workers: Optional[int] = None,
    execution_mode: str = 'default',
    noise_type: str = 'depolarizing'
) -> str:
    """
    Runs large-scale threshold simulations using Sinter and the 4.8.8 Color Code.
    
    Args:
        distances: List of code distances (e.g. [3, 5, 7, 9]).
        p_rates: Array of physical error rates to simulate.
        decoder_name: Name of the decoder ('bposd', 'pymatching', etc.).
        output_file: Name of the output CSV file (e.g., 's2_bposd_results.csv').
        rounds_func: Function to determine measurement rounds from distance. Default is rounds=distance.
        max_shots: Maximum number of shots per task.
        max_errors: Number of logical errors to observe before stopping a task early.
        num_workers: Number of CPU workers for multiprocessing. Defaults to all available cores.
        execution_mode: 'default' (append/resume), 'force-rerun' (overwrite params), or 'clean-all' (delete csv first).
        noise_type: 'depolarizing' or 'X'.
        
    Returns:
        The path to the generated CSV file.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    # Project root is two levels up from src/chapter_6
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(project_root, "data", "chapter_6")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, output_file)
    
    if execution_mode == 'clean-all':
        if os.path.exists(csv_path):
            print(f"Clean-all mode: Deleting existing file {output_file}...")
            os.remove(csv_path)
    
    tasks = []
    print(f"Generating circuits for {len(distances)} distances and {len(p_rates)} physical error rates...")
    
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    is_first = True
    add_chromobius_tags = (decoder_name == 'chromobius')
    color_codes = {}
    
    # Load existing stats to skip already completed tasks (only in default mode)
    import pandas as pd
    existing_stats = {}
    if execution_mode == 'default' and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                for _, row in df.iterrows():
                    if row['decoder'] == decoder_name:
                        key = (int(row['distance']), int(row['rounds']), float(round(row['p_physical'], 7)))
                        existing_stats[key] = (int(row['shots']), int(row['errors']))
        except Exception as e:
            print(f"Warning: Could not read existing CSV to skip tasks: {e}")
            
    for d in distances:
        rounds = rounds_func(d)
        cc = ColorCode(distance=d)
        color_codes[d] = cc
        print(f"--> Generating circuits and Detector Error Models (DEM) for distance d={d}...", flush=True)
        for p in p_rates:
            key = (d, rounds, float(round(p, 7)))
            
            # Skip if already reached max stats in default mode
            if execution_mode == 'default' and key in existing_stats:
                shots, errors = existing_stats[key]
                if shots >= max_shots or errors >= max_errors:
                    print(f"      Skipping d={d}, p={p:.5f} (Already reached {shots} shots / {errors} errors)", flush=True)
                    continue

            # Caching filenames
            circ_path = os.path.join(cache_dir, f"cc4.8.8_d{d}_r{rounds}_pd{p:.5f}_pm0.00000_{noise_type}.stim")
            dem_path = os.path.join(cache_dir, f"cc4.8.8_d{d}_r{rounds}_pd{p:.5f}_pm0.00000_{noise_type}.dem")
            
            import stim
            if is_first:
                stim_circ_new = cc.get_stim_circuit(rounds=rounds, p_data=p, p_meas=0.0, p_gate=0.0, noise_type=noise_type)
                dem_new = stim_circ_new.detector_error_model(decompose_errors=False)
                
                if os.path.exists(circ_path) and os.path.exists(dem_path):
                    with open(circ_path, 'r') as f:
                        old_circ_str = f.read()
                    with open(dem_path, 'r') as f:
                        old_dem_str = f.read()
                        
                    new_circ_str = str(stim_circ_new) + '\n'
                    new_dem_str = str(dem_new) + '\n'
                    
                    if new_circ_str == old_circ_str and new_dem_str == old_dem_str:
                        stim_circ = stim_circ_new
                        dem = dem_new
                    else:
                        print("Cache is dirty. Deleting all cached .stim and .dem files...")
                        import glob
                        for f in glob.glob(os.path.join(cache_dir, "*.stim")) + glob.glob(os.path.join(cache_dir, "*.dem")):
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                        
                        stim_circ = stim_circ_new
                        dem = dem_new
                        stim_circ.to_file(circ_path)
                        dem.to_file(dem_path)
                else:
                    stim_circ = stim_circ_new
                    dem = dem_new
                    stim_circ.to_file(circ_path)
                    dem.to_file(dem_path)
                is_first = False
            else:
                if os.path.exists(circ_path) and os.path.exists(dem_path):
                    stim_circ = stim.Circuit.from_file(circ_path)
                    dem = stim.DetectorErrorModel.from_file(dem_path)
                else:
                    p_meas_val = p if rounds > 1 else 0.0
                    stim_circ = cc.get_stim_circuit(rounds=rounds, p_data=p, p_meas=p_meas_val, p_gate=0.0, noise_type=noise_type)
                    dem = stim_circ.detector_error_model(decompose_errors=False)
                    stim_circ.to_file(circ_path)
                    dem.to_file(dem_path)
            
            tasks.append(
                sinter.Task(
                    circuit=stim_circ,
                    detector_error_model=dem,
                    json_metadata={'d': d, 'p': p, 'rounds': rounds}
                )
            )
            
    custom_decoders = get_custom_decoders()
    
    if decoder_name in ['projection', 'restriction', 'concat_mwpm', 'correlated'] and decoder_name in custom_decoders:
        decoder = custom_decoders[decoder_name]
            
    if not tasks:
        print("No tasks left to run (all requested parameters are already completed).")
        return csv_path
    
    print(f"Starting Sinter parallel sampling across {num_workers} workers...")
    print(f"Decoder: {decoder_name} | Max shots: {max_shots} | Max errors: {max_errors}")
    
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=[decoder_name],
        custom_decoders=custom_decoders,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True
    )
    
    append_mode = execution_mode in ['default', 'force-rerun'] and os.path.exists(csv_path)
    
    print(f"{'Appending/Overwriting' if append_mode else 'Saving'} results to {csv_path}...")
    mode = 'a' if append_mode else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(["decoder", "distance", "rounds", "p_physical", "p_logical", "p_logical_err", "shots", "errors", "avg_time_sec"])
        
        for s in stats:
            d = s.json_metadata['d']
            r = s.json_metadata['rounds']
            p = s.json_metadata['p']
            shots = s.shots
            errors = s.errors
            
            # Calculate the error percentage and the standard error to display it in the charts
            if shots > 0:
                p_L = errors / shots
                p_L_err = np.sqrt(p_L * (1 - p_L) / shots)
                avg_time = s.seconds / shots if (s.seconds and shots > 0) else 0.0
            else:
                p_L = 0.0
                p_L_err = 0.0
                avg_time = 0.0
                
            writer.writerow([decoder_name, d, r, p, p_L, p_L_err, shots, errors, avg_time])
            
    if append_mode:
        df = pd.read_csv(csv_path)
        # Drop duplicates by taking the LAST one (the newly appended one)
        df['p_rounded'] = df['p_physical'].round(7)
        df = df.drop_duplicates(subset=['decoder', 'distance', 'rounds', 'p_rounded'], keep='last')
        df = df.drop(columns=['p_rounded'])
        df = df.sort_values(by=['distance', 'p_physical']).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
            
    print(f"Simulation complete. Results saved at {csv_path}")
    return csv_path

def run_cli_pipeline(decoder_name: str, file_prefix: str, plot_title: str):
    """
    Unified CLI pipeline to run simulations, plots, and zooms without duplicating code.
    """
    import argparse
    import config
    from plotter import plot_threshold, plot_execution_time, calculate_and_save_threshold, plot_threshold_zoom
    
    def run_simulate(distances, execution_mode, allowed_noise_types):
        p_rates = config.get_config(decoder_name, 'p_rates')
        max_shots = config.get_config(decoder_name, 'max_shots')
        max_errors = config.get_config(decoder_name, 'max_errors')
        
        for noise_type in allowed_noise_types:
            output_file = f"{file_prefix}_results_{noise_type}.csv" if noise_type != 'depolarizing' else f"{file_prefix}_results.csv"
            
            print(f"\n--- Running SIMULATE for noise type: {noise_type.upper()} ---")
            run_color_code_simulations(
                distances=distances,
                p_rates=p_rates,
                decoder_name=decoder_name,
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
            output_file = f"{file_prefix}_results_{noise_type}.csv" if noise_type != 'depolarizing' else f"{file_prefix}_results.csv"
            csv_path = os.path.join(project_root, "data", "chapter_6", output_file)
            
            if not os.path.exists(csv_path):
                print(f"Error: {csv_path} not found. Run 'simulate' first.")
                continue
                
            title_suffix = "" if noise_type == 'depolarizing' else " - Pure X Noise"
            
            print(f"\nGenerating Threshold Plot for {noise_type}...")
            plot_threshold(csv_path, title=f"Logical noise vs physical noise - 4.8.8 Color Code ({plot_title}{title_suffix})", distances_filter=distances_filter)
            
            print(f"Generating Execution Time Plot for {noise_type}...")
            plot_execution_time(csv_path, title=f"Decoding Complexity - {plot_title}{title_suffix}", distances_filter=distances_filter)

    def run_zoom(distances, execution_mode, allowed_noise_types):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        zoom_max_shots = config.get_config(decoder_name, 'zoom_max_shots')
        zoom_max_errors = config.get_config(decoder_name, 'zoom_max_errors')
        zoom_range_factor = config.get_config(decoder_name, 'zoom_range_factor')
        zoom_points = config.get_config(decoder_name, 'zoom_points')
        
        for noise_type in allowed_noise_types:
            base_output_file = f"{file_prefix}_results_{noise_type}.csv" if noise_type != 'depolarizing' else f"{file_prefix}_results.csv"
            zoom_output_file = base_output_file.replace('.csv', '_zoom.csv')
            base_csv_path = os.path.join(project_root, "data", "chapter_6", base_output_file)
            
            if not os.path.exists(base_csv_path):
                print(f"Error: Base file {base_csv_path} not found. Cannot calculate threshold for zoom.")
                continue
                
            import pandas as pd
            df = pd.read_csv(base_csv_path)
            df_filtered = df[df['distance'].isin(distances)]
            filtered_distances_list = sorted(df_filtered['distance'].unique())
            
            if len(filtered_distances_list) < 2:
                 print(f"Not enough distances found in {base_csv_path} to calculate threshold for zoom.")
                 continue
                 
            est_threshold = calculate_and_save_threshold(base_csv_path, f'{decoder_name.upper()}_{noise_type.upper()}', filtered_distances_list, df_filtered)
            
            if np.isnan(est_threshold):
                print(f"Could not estimate threshold for {noise_type}. Skipping zoom.")
                continue
                
            zoom_p_rates = np.linspace(zoom_range_factor[0] * est_threshold, zoom_range_factor[1] * est_threshold, zoom_points)
            
            print(f"\n--- Running ZOOM for noise type: {noise_type.upper()} around threshold {est_threshold:.4f} ---")
            run_color_code_simulations(
                distances=distances,
                p_rates=zoom_p_rates,
                decoder_name=decoder_name,
                output_file=zoom_output_file,
                rounds_func=lambda d: 1,
                max_shots=zoom_max_shots,
                max_errors=zoom_max_errors,
                execution_mode=execution_mode,
                noise_type=noise_type
            )
            
            title_suffix = "" if noise_type == 'depolarizing' else " - Pure X Noise"
            plot_threshold_zoom(base_csv_path, est_threshold, title=f"Logical noise vs physical noise - 4.8.8 Color Code ({plot_title} - Zoom{title_suffix})", distances_filter=distances)

    # Argparse and Execution Block
    parser = argparse.ArgumentParser(description=f"{plot_title} simulation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
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
    
    distances = args.distances if args.distances else config.get_config(decoder_name, 'distances')
    allowed_noise_types = ['depolarizing', 'X'] if args.noise_type == 'all' else [args.noise_type]
    
    execution_mode = 'default'
    if hasattr(args, 'force_rerun') and args.force_rerun:
        execution_mode = 'force-rerun'
    elif hasattr(args, 'clean_all') and args.clean_all:
        execution_mode = 'clean-all'
        
    print("=====================================================")
    print(f"      {plot_title} Simulation - 4.8.8 Color Code   ")
    print("=====================================================")
    
    if args.command == 'simulate':
        run_simulate(distances, execution_mode, allowed_noise_types)
    elif args.command == 'plot':
        run_plot(distances, allowed_noise_types)
    elif args.command == 'zoom':
        run_zoom(distances, execution_mode, allowed_noise_types)

if __name__ == "__main__":
    pass
