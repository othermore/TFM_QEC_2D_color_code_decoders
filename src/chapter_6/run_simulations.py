import argparse
import subprocess
import sys
import os

SCRIPTS = [
    "s2_bposd_simulation.py",
    "s3_projection_simulation.py",
    "s4_restriction_simulation.py",
    "s5_mobius_decoder.py",
    "s6_concatenated_simulation.py"
]

def main():
    parser = argparse.ArgumentParser(description="Orchestrator to run all color code decoder simulations sequentially.")
    parser.add_argument("command", choices=['simulate', 'plot', 'zoom'], help="The action to perform across all decoders.")
    parser.add_argument("--distances", nargs="+", type=int, help="Override distances to process.")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun the simulations (overwrite existing params).")
    parser.add_argument("--clean-all", action="store_true", help="Clean all data for the given action before running.")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in SCRIPTS:
        cmd = [sys.executable, os.path.join(script_dir, script), args.command]
        
        if args.distances:
            cmd.extend(["--distances"] + [str(d) for d in args.distances])
            
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
