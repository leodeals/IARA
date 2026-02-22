# IARA: Interface Analysis and Recognition Architecture

> *"Iara is the mother of the waters. In Brazilian folklore, she shapes the rivers and the depths. In structural biology, it is the displacement and structuring of water—the hydrophobic effect—that fundamentally drives proteins to fold, bind, and interact."*

**IARA** is a Graph Neural Network (GNN) trained on synthetic protein interfaces (generated via RFdiffusion) to predict chemically favorable, highly designable interaction hotspots on natural protein surfaces. Unbound by evolutionary history, IARA identifies the most thermodynamically stable geometric pockets for *de novo* binder engineering.

## Installation

We recommend using Conda to manage dependencies.

```bash
conda env create -f environment.yml
conda activate prospector_env
```

## Usage

You must download the official model weights (`IARA.pth`) and place them in the directory.

### Predicting a Single Structure

```bash
python predict.py --model IARA.pth --input target.pdb --outdir predictions/
```

### Predicting a Directory of Structures

```bash
python predict.py --model IARA.pth --input /path/to/pdbs/ --outdir predictions/
```

## GUI Plugin Installation

We provide lightweight plugins for standard structural visualization tools to run predictions and visualize hotspots directly in the viewport.

### Prerequisites (Python Environment)
IARA uses a Graph Neural Network (PyTorch) to analyze your structures. Both plugins execute IARA locally in the background. Therefore, **you must have a valid Python Conda environment installed on your computer**.

If you do not have one set up, please open your terminal (or Anaconda Prompt on Windows) and run:
`conda create -n iara_env python=3.10`
`conda activate iara_env`
`pip install torch torch-geometric prody pandas scipy`

*(The plugins automatically look for an environment named `iara_env` across Windows, macOS, and Linux).*

### Installing & Configuring

1. Download the entire **IARA GitHub repository** folder. Ensure the `IARA.pth` model weights are inside the extracted folder.
2. Open your visualization software (PyMOL or UCSF ChimeraX).
3. Load the plugin script for your software using its native command prompt:
   - **PyMOL:** `run /absolute/path/to/extracted/IARA/Deployment/PyMOL/iara_plugin_pymol.py`
      *(Alternatively: `Plugin` -> `Plugin Manager` -> `Install New Plugin` and choose the script).*
   - **ChimeraX:** `open /absolute/path/to/extracted/IARA/Deployment/ChimeraX/iara_plugin_chimerax.py`
4. Tell the plugin where your repository folder is located (run this once):
   `iara_configure /absolute/path/to/extracted/IARA/folder`
5. *(Optional)* If your Conda environment is named something other than `iara_env`, tell the plugin:
   `iara_env_name my_custom_env_name`

*(These configurations are permanently saved to your computer (`~/.iara_config.txt`), so you will never have to configure them again!)*

### Running Predictions

- **PyMOL:** Load a structure (e.g., `fetch 1cse`) and type: `iara_predict 1cse`
- **ChimeraX:** Load a structure (e.g., `open 1a22`) and type: `iara_predict #1`

## Output Interpretation

The GUI plugins will automatically download the scored `.pdb` structure and apply a B-factor color scheme. Regions colored deep red represent high-confidence binding hotspots (score > 50).
