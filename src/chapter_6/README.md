# Chapter 6: Simulating 2D Color Code Decoders

This directory contains the simulation pipeline for analyzing the threshold and performance of four different decoders on the 4.8.8 Color Code:
1. **BP+OSD** (`s2_bposd_simulation.py`)
2. **Projection Decoder** (`s3_projection_simulation.py`)
3. **Restriction Decoder** (`s4_restriction_simulation.py`)
4. **Möbius Decoder** (`s5_mobius_decoder.py`)

## Architecture & Data Structure

All scripts are powered by a central configuration file (`config.py`) and a shared simulation engine (`simulator.py`).

### Data Storage
- Results are stored in `../../data/chapter_6/`.
- **Standard data** (wide range of physical errors) is saved as `sX_[decoder]_results.csv` and `sX_[decoder]_results_X.csv` (for pure X noise).
- **Zoom data** (high-resolution sampling around the threshold) is saved separately in `sX_[decoder]_results_zoom.csv` and `sX_[decoder]_results_X_zoom.csv`.
- **Threshold Summary** is stored in `thresholds_summary.csv` and is automatically updated during plotting.

### Plots
- Plots are saved in `../../figures/chapter_6/`.
- Three plots are generated per noise model:
  - `sX_[decoder]_results_threshold.png`
  - `sX_[decoder]_results_threshold_zoom.png`
  - `sX_[decoder]_results_time.png`

## Execution Commands

Each simulation script exposes three main sub-commands: `simulate`, `plot`, and `zoom`.

### 1. `simulate`
Runs the standard simulation over the default broad range of physical error rates.
```bash
python s5_mobius_decoder.py simulate --distances 9 11 13
```
- By default, it will **append** new simulations and **skip** parameter configurations that have already reached `max_shots` or `max_errors`.
- `--force-rerun`: Forces the script to simulate the requested distances from scratch and overwrites their old rows in the CSV, while preserving data from other distances.
- `--clean-all`: Deletes the standard CSV files entirely before simulating.

### 2. `plot`
Calculates the numerical threshold, updates `thresholds_summary.csv`, and regenerates all `.png` plots using the existing standard and zoom CSV files.
```bash
python s5_mobius_decoder.py plot --distances 9 11 13
```
- The `--distances` flag acts as a **filter**. If provided, the threshold will be calculated using *only* those distances, and the plots will only show those distances. 

### 3. `zoom`
Reads the standard CSV to calculate the exact threshold, then runs a high-resolution simulation specifically in a narrow window around that threshold.
```bash
python s5_mobius_decoder.py zoom --distances 9 11 13
```
- Outputs data exclusively to the `_zoom.csv` files to keep datasets cleanly separated.
- Supports `--force-rerun` and `--clean-all` with the same behavior as `simulate` (but affecting only the `_zoom.csv` files).

### Orchestrator: `run_simulations.py`
To avoid running the above commands manually for all four decoders one by one, you can use the global orchestrator.
```bash
python src/chapter_6/run_simulations.py [command] [--distances ...] [--force-rerun] [--clean-all]
```
For example, to recalculate all thresholds and generate all images for every decoder, run:
```bash
python src/chapter_6/run_simulations.py plot
```

## Configuration (`config.py`)

All default parameters are located in `config.py`.
- **Global Defaults**: Defined in `DEFAULT_CONFIG`.
- **Decoder Overrides**: You can override any default parameter for a specific decoder by adding it to the `DECODER_CONFIGS` dictionary. 

For example, `bposd` uses a different `p_rates` array by default because it requires a wider noise range.

**Zoom Parameters**:
- `zoom_max_shots`: Independent max shots for the zoom phase.
- `zoom_max_errors`: Independent max errors for the zoom phase.
- `zoom_range_factor`: The multiplier window around the threshold (e.g., `(0.75, 1.25)`).
- `zoom_points`: The number of discrete points to sample within that window.
