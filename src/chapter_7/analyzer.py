import argparse
import data_processing
import plotter

def main():
    parser = argparse.ArgumentParser(description="Chapter 7 Data Analyzer and Plotter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # process command
    parser_process = subparsers.add_parser("process", help="Process raw data and generate extrapolated CSVs")
    
    # plot command
    parser_plot = subparsers.add_parser("plot", help="Generate all plots from the processed CSVs")
    
    args = parser.parse_args()
    
    if args.command == "process":
        data_processing.run_all_processing()
    elif args.command == "plot":
        plotter.run_all_plots()

if __name__ == "__main__":
    main()
