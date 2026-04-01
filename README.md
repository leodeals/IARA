# IARA: Interface Analysis and Recognition Architecture

> *"Iara is the mother of the waters. In Brazilian folklore, she shapes the rivers and the depths. In structural biology, it is the displacement and structuring of water—the hydrophobic effect—that fundamentally drives proteins to fold and interact."*

**IARA** is a Graph Neural Network (GNN) able to predict regions most likely to yield successful *de novo* binders with RFdiffusion, BindCraft and BoltzGen.

IARA is designed to fit your workflow. You can run it three ways:
1. **As a Standalone Command-Line Script:** For batch processing and integration into other pipelines.
2. **As a GUI Plugin (PyMOL & UCSF ChimeraX):** For an interactive, single-click design experience right inside your viewport.
3. **As a Google Colab Notebook:** For a zero-install, cloud-based prediction directly in your browser.
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leodeals/IARA/blob/main/IARA_Colab.ipynb)

## 👩‍🔬 Getting Started: A Quick Guide on the Basics of Computational Tools for Unexperienced Users
*If you are comfortable with the command line and Conda, please skip to the [Quick Start (For Experienced Users)](#-quick-start-for-experienced-users) below!*

Welcome! Setting up deep learning tools for the first time can feel incredibly daunting because computational papers often assume you already know a lot of jargon. I want you to be able to use IARA effortlessly either from a terminal or from right inside PyMOL or ChimeraX, without needing a computer science degree. I am going to walk you through exactly what these terms mean and how to set everything up, step-by-step.

### 1. What is Conda, and why do I need it?
IARA runs on **PyTorch**, a complex underlying AI engine that requires very specific, matched versions of code (Python) and math libraries to function. If we mix different versions, the program crashes. 
**Conda** is simply a "package manager"—it's a tool that creates a safe, isolated bubble (an **environment**) on your computer where IARA's specific math libraries can live happily without interfering with any other software on your machine.

**Do I have Conda?**
Let’s check! Open your **Terminal** (on Mac: press `Cmd + Space` and type "Terminal"; on Windows: open the Start menu, type "Anaconda Prompt").
In that black window, type this exact text and press Enter:
```bash
conda --version
```
If it prints out a version number (like `conda 24.1.0`), you are good to go! **Skip to Step 2.**

If it says "command not found", don't worry! You just need to install Miniconda:
- **Mac/Linux/Windows Users:** Go to the [Miniconda Installation Page](https://docs.anaconda.com/free/miniconda/) and download the installer for your computer. Run the installer just like any normal application.
- *Note for Windows:* Always use the "Anaconda Prompt" app from your Start menu instead of the standard "cmd.exe" when we mention the terminal.

### 2. Creating Your IARA Environment
Now that you have Conda installed, let's create that safe "bubble" for IARA to live in. Open your Terminal (or Anaconda Prompt) and type this line carefully, exactly as written, then press Enter:
```bash
conda create -n iara_env python=3.10
```
*(Conda will eventually ask if it's okay to proceed by typing `y` and pressing Enter).*

Now, tell your computer to step *inside* that new bubble:
```bash
conda activate iara_env
```
Finally, let's install the actual math and biology engines. Copy this line, paste it in the terminal, and press Enter (this will take a minute or two):
```bash
pip install torch torch-geometric prody pandas scipy
```
**Congratulations!** The hard computational part is over. You never have to type those setup commands again.

### 3. Downloading IARA 📥
1. Go to our official repository: [https://github.com/leodeals/IARA](https://github.com/leodeals/IARA)
2. Follow the standard process to download the repository folder (usually clicking the green "Code" button -> "Download ZIP"). Unzip this file in a folder where the script will be stored.
   **(If you are the type of person with the "barbaric habit" ⚔️🛡️ of keeping absolutely everything on your desktop or the downloads folder, I won't hold this against you).**
3. Keep the AI brain (`IARA.pth`) securely *inside* that unzipped folder, right next to the file called `predict.py`. Without the brain, the code won't know what to do!

### 4. Running IARA from the Command Line 🧑‍💻
You can run IARA directly from your Terminal! This is great if you want to process many structural files at once, or if you're the kind of person who enjoys looking intensely at black screens with scrolling text to impress your lab mates.

1. Make sure your terminal is open and your bubble is active (you should see `(iara_env)` on the left side of the prompt). If not, type `conda activate iara_env`.
2. Tell your terminal to travel into the IARA folder you just downloaded. For example, if you unzipped it on your Desktop, type `cd Desktop/IARA` and press Enter.
3. You can process a **single PDB file** or an **entire folder of PDBs at once**!
4. IARA also accepts protein complexes and the predictions will look at the complex surface as a whole. ‼️ **for this reason, if you run an isolated subunit you will get a slightly different prediction from the full complex, as new interaction surfaces are revealed and cross-subunit interfaces may be lost** ‼️

Here are the only commands you need to worry about:

| Flag | Status | What it does | Example |
|---|---|---|---|
| `--input` | **Obligatory 🚨** | Points to your target PDB file, OR a whole folder containing multiple PDBs. | `--input target.pdb` <br> `--input /my_pdbs/` |
| `--model` | *Optional* 🤷‍♂️ | Points to the AI brain. You only need this if you moved `IARA.pth` away from `predict.py`. | `--model /custom/path/IARA.pth` |
| `--outdir` | *Optional* 📁 | Saves the scored files into a specific folder. By default, it just saves them right next to the original files! | `--outdir predictions/` |

**Example 1: Scoring a Single Protein** (let's say you have a file called `target.pdb` in that folder)
```bash
python predict.py --input target.pdb
```
That's it! Your computer will think for a moment and automatically save the results locally. 

**Example 2: Scoring a Full Folder** (because doing things manually one-by-one is for peasants 👑)
```bash
python predict.py --input /path/to/my/folder/of/pdbs/ --outdir predictions/
```

### 5. Installing the PyMOL or ChimeraX Plugins 🎨
If you prefer visual tools instead of the command line, we can connect IARA seamlessly to the 3D software you already know and love!

**For PyMOL:**
1. Open PyMOL normally.
2. In PyMOL's command line box (at the top or bottom of the screen), type `run ` and then drag-and-drop the `iara_plugin_pymol.py` file from your unzipped folder into the PyMOL window to auto-fill the path, like this:
   `run /Users/yourname/Desktop/IARA/Deployment/PyMOL/iara_plugin_pymol.py` and press Enter.
3. Tell the plugin where the AI brain is located by typing `iara_configure ` and dragging your main unzipped `IARA` folder into PyMOL:
   `iara_configure /Users/yourname/Desktop/IARA` and press Enter.

**For UCSF ChimeraX:**
1. Open ChimeraX.
2. In the ChimeraX command line prompt at the bottom, type `open ` and drag-and-drop the `iara_plugin_chimerax.py` file:
   `open /Users/yourname/Desktop/IARA/Deployment/ChimeraX/iara_plugin_chimerax.py`
3. Tell the plugin where the AI brain is located by typing `iara_configure ` and dragging your main unzipped `IARA` folder into ChimeraX:
   `iara_configure /Users/yourname/Desktop/IARA`

*(We permanently save this configuration setting on your computer, so next week when you restart your computer, you only ever need to do step #2!)*

### 6. Running Your First Prediction in 3D! 🧊
You are all set! Load any protein structure you want into your viewport (for example, in PyMOL you can type `fetch 1cse`).
To let the AI scan the surface and find the optimal binding hotspots, just type:
* **PyMOL:** `iara_predict 1cse`
* **ChimeraX:** `iara_predict #1`

Your screen will freeze for a few seconds while the GNN thinks, and then your protein will seamlessly update with a 3D heatmap! The regions colored **deep red** (probability > 50%) are the high-confidence binding hotspots perfect for *de novo* design. 

#### Wait, I didn't install the Plugin! How do I visualize it manually? 🧐
If you ran the command-line script from Step 4 and just got back a `_IARA.pdb` file, here is how you manually reveal the heatmap:
* **In PyMOL:** Open the `_IARA.pdb` file, click the command line, type `spectrum b, blue_white_red, minimum=0, maximum=100` and press Enter. (Boom. Magic. 🪄)
* **In ChimeraX:** Open the `_IARA.pdb` file, go to your top menu bar, click **Tools -> Depiction -> Render by Attribute**, make sure the attribute is set to `b-factor`, select your color palette, and apply!

---

## 💻 Quick Start (For Experienced Users)

### CLI Installation & Usage

```bash
git clone https://github.com/leodeals/IARA
cd IARA
conda env create -f environment.yml
conda activate prospector_env
```
Ensure you have downloaded the model weights (`IARA.pth`) to the repository root.

**Minimal Call (recommended)**
```bash
# --model and --outdir are optional:
# model defaults to IARA.pth next to predict.py
# output defaults to the same folder as the input
python predict.py --input target.pdb
```

**Single Structure — custom output folder**
```bash
python predict.py --input target.pdb --outdir predictions/
```

**Batch Directory**
```bash
python predict.py --input /path/to/pdbs/
```

**Override model path (only if IARA.pth was moved)**
```bash
python predict.py --input target.pdb --model /custom/path/IARA.pth
```

### GUI Plugin Installation

Make sure your Conda environment is named `iara_env`, or see the override below.

- **PyMOL:** `run /path/to/IARA/Deployment/PyMOL/iara_plugin_pymol.py`
- **ChimeraX:** `open /path/to/IARA/Deployment/ChimeraX/iara_plugin_chimerax.py`

**Configuration (Run Once):**
```bash
iara_configure /path/to/IARA
```
*(Optional) If your Conda environment is named something else, pass the override:* `iara_env_name custom_env_name`

**Running Predictions:**
- **PyMOL:** `iara_predict <object_name>`
- **ChimeraX:** `iara_predict #1`

### Output Interpretation
The script outputs a new `.pdb` file (e.g., `target_IARA.pdb`) where the B-factor column of every C-alpha atom has been replaced with the model's predicted hotspot probability (scaled 0 to 100). The GUI plugins automatically load this file and apply a gradient color scheme where red represents high-confidence binding hotspots.
