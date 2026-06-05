import argparse
import subprocess
import sys
import os

DECODER_SCRIPTS = {
    "bposd": "s2_bposd_simulation.py",
    "projection": "s3_projection_simulation.py",
    "restriction": "s4_restriction_simulation.py",
    "mobius": "s5_mobius_decoder.py",
    "concatenated": "s6_concatenated_simulation.py",
    "correlated": "s7_correlated_simulation.py"
}

def main():
    parser = argparse.ArgumentParser(description="Orchestrator to run all color code decoder simulations sequentially.")
    parser.add_argument("command", choices=['simulate', 'plot', 'zoom'], help="The action to perform across all decoders.")
    parser.add_argument("--distances", nargs="+", type=int, help="Override distances to process.")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun the simulations (overwrite existing params).")
    parser.add_argument("--clean-all", action="store_true", help="Clean all data for the given action before running.")
    parser.add_argument("--noise-type", choices=['all', 'depolarizing', 'X'], default='all', help="Run only for specific noise type.")
    parser.add_argument("--decoders", nargs="+", choices=list(DECODER_SCRIPTS.keys()), help="List of decoders to run. If not provided, runs all.")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts_to_run = DECODER_SCRIPTS.values()
    if args.decoders:
        scripts_to_run = [DECODER_SCRIPTS[d] for d in args.decoders]
    
    for script in scripts_to_run:
        cmd = [sys.executable, os.path.join(script_dir, script), args.command]
        
        if args.distances:
            cmd.extend(["--distances"] + [str(d) for d in args.distances])
            
        if args.noise_type != 'all':
            cmd.extend(["--noise-type", args.noise_type])
            
        if args.command in ['simulate', 'zoom']:
            if args.force_rerun:
                cmd.append("--force-rerun")
            if args.clean_all:
                cmd.append("--clean-all")
                
        # Pretty print command for logging
        display_cmd = []
        for c in cmd:
            if c == sys.executable:
                display_cmd.append("python")
            elif ".py" in c:
                display_cmd.append(os.path.basename(c))
            else:
                display_cmd.append(c)
                
        print(f"\n=========================================================================")
        print(f" Executing: {' '.join(display_cmd)}")
        print(f"=========================================================================\n")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error executing {script}. Aborting orchestrator.")
            sys.exit(result.returncode)
            
    print("\nOrchestrator finished successfully!")

if __name__ == "__main__":
    main()
