# 2D Topological Color Code Decoders

This repository contains the code, data, and manuscript for a Master's Thesis presenting a comprehensive study on quantum error correction (QEC), specifically focused on 2D topological color codes. Leveraging state-of-the-art simulation tools such as `stim` and `sinter`, alongside specialized libraries like `ldpc` and `pymatching`, this project analyzes decoder performance on 4.8.8 lattice geometries. The study concludes with a comparative analysis of the decoding results and explores potential improvements for the error correction process.

This work provides a systematic and structured framework to evaluate the performance of QEC codes and decoders by implementing a selected subset of these algorithms within an extensible architecture, allowing for future expansions. Furthermore, this manuscript and its accompanying codebase are designed to serve as an accessible introduction for researchers and students entering the field.

## Project Structure

The repository is organized into the following main directories and files:

- **`src/`**: Contains the source code. This is where the simulation framework, the color code lattice generation, the implementation of the different decoders, and the data analysis scripts are located.
- **`thesis/`**: Contains the LaTeX source code for the thesis manuscript.
- **`data/`**: Stores the raw and processed data generated from the quantum error correction simulations and the outputs from the analysis scripts.
- **`figures/`**: Contains the plots, graphs, and visual results generated directly by the simulation and data analysis scripts.
- **`thesis/figures/`**: Contains custom figures, diagrams, and illustrations specifically created for the thesis manuscript (e.g., lattice diagrams, failure modes, architectural workflows).
  - **`thesis/figures/unused/`**: A historical archive containing discarded diagram iterations, early drafts, and alternative visual approaches that were not ultimately included in the final thesis manuscript.
- **`thesis_pdf/`**: Contains the final compiled PDF versions of the thesis. It includes both a digital format optimized for screen reading and a format ready for printing (complete with the corresponding front and back covers).
- **`requirements.txt`**: Lists all the Python dependencies required to run the simulation and analysis code.

## Getting Started

To execute the simulations or reproduce the analysis, it is highly recommended to use a virtual environment (such as `venv` or `conda`) to manage your dependencies.

Here is how you can set up the environment using Conda:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/othermore/TFM_QEC_2D_color_code_decoders.git
   cd TFM_QEC_2D_color_code_decoders
   ```
2. Create a new Conda environment with Python 3.10+ (for example, Python 3.11):
   ```bash
   conda create -n tfm_qec python=3.11
   ```
3. Activate the newly created environment:
   ```bash
   conda activate tfm_qec
   ```
4. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
5. You can now execute the relevant Python scripts directly from your terminal. For example, to run an analysis script:
   ```bash
   python src/chapter_7/analyzer.py
   ```

> [!WARNING]
> Please note that executing the Python scripts in `src/` will overwrite and **re-generate the data and images** that are currently saved in the `data/` and `figures/` directories of this repository.

For detailed instructions on how to use each of the simulation and analysis scripts, please refer to the dedicated **`src/README.md`** file.
