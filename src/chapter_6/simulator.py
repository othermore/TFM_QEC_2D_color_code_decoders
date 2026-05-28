import os
import csv
import sinter
import numpy as np
from typing import List, Optional
import multiprocessing

from s1_color_code import ColorCode
from decoder_interfaces import get_custom_decoders

def run_color_code_simulations(
    distances: List[int],
    p_rates: np.ndarray,
    decoder_name: str,
    output_file: str,
    rounds_func=lambda d: d,
    max_shots: int = 1_000_000,
    max_errors: int = 500,
    num_workers: Optional[int] = None
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
    
    tasks = []
    print(f"Generating circuits for {len(distances)} distances and {len(p_rates)} physical error rates...")
    
    cache_dir = os.path.join(results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    is_first = True
    
    for d in distances:
        rounds = rounds_func(d)
        cc = ColorCode(distance=d)
        print(f"--> Generating circuits and Detector Error Models (DEM) for distance d={d}...", flush=True)
        for p in p_rates:
            # Caching filenames
            circ_path = os.path.join(cache_dir, f"cc4.8.8_d{d}_r{rounds}_pd{p:.5f}_pm0.00000.stim")
            dem_path = os.path.join(cache_dir, f"cc4.8.8_d{d}_r{rounds}_pd{p:.5f}_pm0.00000.dem")
            
            import stim
            if is_first:
                stim_circ_new = cc.get_stim_circuit(rounds=rounds, p_data=p, p_meas=0.0, p_gate=0.0)
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
                    stim_circ = cc.get_stim_circuit(rounds=rounds, p_data=p, p_meas=p_meas_val, p_gate=0.0)
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
    
    print(f"Saving results to {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["decoder", "distance", "rounds", "p_physical", "p_logical", "p_logical_err", "shots", "errors", "avg_time_sec"])
        
        for s in stats:
            d = s.json_metadata['d']
            r = s.json_metadata['rounds']
            p = s.json_metadata['p']
            shots = s.shots
            errors = s.errors
            
            # Calculate the error percentage and the standard error to display it in the charts
            # We use the standard error formula: sqrt( p * (1 - p) / N )
            # where p is the average and N the sample size.
            if shots > 0:
                p_L = errors / shots
                p_L_err = np.sqrt(p_L * (1 - p_L) / shots)
                
                # Sinter provides the execution time for all shots
                # which is the sum of the time reported by each thread
                avg_time = s.seconds / shots if (s.seconds and shots > 0) else 0.0
            else:
                p_L = 0.0
                p_L_err = 0.0
                avg_time = 0.0
                
            writer.writerow([decoder_name, d, r, p, p_L, p_L_err, shots, errors, avg_time])
            
    print(f"Simulation complete. Results saved at {csv_path}")
    return csv_path

if __name__ == "__main__":
    pass
