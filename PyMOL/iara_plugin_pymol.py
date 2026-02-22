import os
import sys
import tempfile
from pymol import cmd

# -----------------------------------------------------------------------------
# IARA: Interface Analysis and Recognition Architecture
# PyMOL Plugin
# -----------------------------------------------------------------------------

def iara_predict(*args, **kwargs):
    # PyMOL splits arguments by comma, so we join them back together for complex selections
    selection = ",".join(args) if args else "all"
    """
    DESCRIPTION
    
        Runs the IARA Graph Neural Network on the specified selection to predict 
        thermodynamic interaction hotspots. The predicted probability (0-100) 
        is written to the B-factor column, and the protein is recolored as a heatmap.
        
    USAGE
    
        iara_predict [selection]
        
    EXAMPLE
    
        fetch 1cse
        iara_predict 1cse
    """
    print(f"\n🌊 IARA Inference (Interface Analysis and Recognition Architecture) 🌊")
    print(f"Targeting selection: {selection}")
    
    # Check if we have python packages we need. PyMOL ships with its own python.
    # Users will need to install torch, torch_geometric, and prody into PyMOL's python environment
    # or point this plugin to their external executable. 
    # For this elegant plugin, we assume they have the GitHub CLI version installed and we 
    # will call it as an external process to avoid corrupting PyMOL's internal conda env.
    
    # Read the permanent configuration file
    config_file = os.path.expanduser("~/.iara_config.txt")
    
    if not os.path.exists(config_file):
        print("❌ Error: IARA installation path not configured.")
        print("Please tell PyMOL where you extracted the IARA GitHub folder by typing this into the PyMOL console:")
        print("    iara_configure /absolute/path/to/extracted/IARA/folder")
        return
        
    with open(config_file, "r") as f:
        PLUGIN_DIR = f.read().strip()
        
    IARA_SCRIPT = os.path.join(PLUGIN_DIR, "predict.py")
    IARA_MODEL = os.path.join(PLUGIN_DIR, "IARA.pth")
    
    if not os.path.exists(IARA_SCRIPT) or not os.path.exists(IARA_MODEL):
        print("❌ Error: IARA predict.py or model weights not found.")
        print(f"IARA is currently configured to look in: {PLUGIN_DIR}")
        print("Please ensure both files are there, or re-run 'iara_configure' with the correct path.")
        return
        
    temp_dir = tempfile.mkdtemp()
    temp_pdb = os.path.join(temp_dir, "temp_target.pdb")
    
    # Save the PyMOL selection to a temporary PDB file
    try:
        cmd.save(temp_pdb, selection)
    except Exception as e:
        print(f"❌ Failed to save selection: {e}")
        return
        
    print(f"🏃 Running IARA Neural Network...")
    
    import subprocess
    import sys
    
    # -------------------------------------------------------------------------
    # Cross-Platform Conda Execution
    # PyMOL hijacks the 'python' command. We must force it to use the user's Conda.
    # -------------------------------------------------------------------------
    
    # Check if user specified a custom conda environment name, otherwise default to iara_env
    env_config_file = os.path.expanduser("~/.iara_env_config.txt")
    conda_env = "iara_env"
    if os.path.exists(env_config_file):
        with open(env_config_file, "r") as f:
            conda_env = f.read().strip()
            
    if os.name == 'nt':  # Windows
        # Windows requires routing through cmd.exe to pick up conda hooks
        command = ["cmd.exe", "/c", f"conda run -n {conda_env} python {IARA_SCRIPT} --model {IARA_MODEL} --input {temp_pdb} --outdir {temp_dir}"]
    else:  # Mac / Linux
        # Unix requires a login shell to source ~/.bashrc or ~/.zshrc
        command = ["bash", "-l", "-c", f"conda run -n {conda_env} python {IARA_SCRIPT} --model {IARA_MODEL} --input {temp_pdb} --outdir {temp_dir}"]
    
    try:
        # Capture the output so we can show EXACTLY why it failed in the PyMOL console
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ IARA Inference Failed! Backend Error Traceback:")
            print("-" * 50)
            print(result.stdout)
            print(result.stderr)
            print("-" * 50)
            print("Troubleshooting: Does 'python' in your current terminal have torch, torch_geometric, and prody installed?")
            print("Try running PyMOL from the same Conda environment where you installed IARA.")
            return
    except Exception as e:
        print(f"❌ Failed to launch backend script: {e}")
        return
        
    # The script names the output <stem>_IARA.pdb
    scored_pdb = os.path.join(temp_dir, "temp_target_IARA.pdb")
    
    if not os.path.exists(scored_pdb):
        print("❌ Could not locate the scored output file.")
        return
        
    # Load it back into PyMOL
    new_obj_name = f"{selection}_IARA"
    cmd.load(scored_pdb, new_obj_name)
    
    # Beautify the visualization
    print("🎨 Applying IARA Heatmap Visualization...")
    cmd.hide("everything", new_obj_name)
    cmd.show("cartoon", new_obj_name)
    
    # Scale B-factors from 0-100 down to 0.0-1.0 for standard interpretation
    cmd.alter(new_obj_name, "b = b / 100.0")
    
    # PyMOL B-factor coloring spectrum: Blue (0.0) -> White (0.5) -> Red (1.0)
    cmd.spectrum("b", "blue_white_red", new_obj_name, minimum=0.0, maximum=1.0)
    
    # Generate an explicit color legend (ramp) using a dummy gaussian map
    # This is standard practice in PyMOL for adding color bars without complex plugins
    dummy_map = f"{new_obj_name}_map"
    legend_name = f"IARA_Scale_{new_obj_name}"
    
    # Remove old legends if re-running
    if legend_name in cmd.get_names():
        cmd.delete(legend_name)
        cmd.delete(dummy_map)
        
    cmd.map_new(dummy_map, "gaussian", 1, new_obj_name)
    cmd.ramp_new(legend_name, dummy_map, [0.0, 0.5, 1.0], ["blue", "white", "red"])
    
    # Hide the ugly invisible map, but keep the beautiful color bar legend visible
    cmd.disable(dummy_map)
    cmd.disable(selection)
    
    print(f"✅ IARA Prediction Complete! ({new_obj_name})")
    print(f"🔥 Red patches indicate highly designable thermodynamic hotspots.")

# Extend the PyMOL command line
cmd.extend("iara_predict", iara_predict)

# Enable TAB auto-completion for loaded PyMOL objects as the first argument
cmd.auto_arg[0]['iara_predict'] = [cmd.object_sc, 'object', '']


def iara_configure(installation_path, **kwargs):
    """
    DESCRIPTION
    
        Configures the IARA plugin with the absolute path to your downloaded GitHub folder.
        
    USAGE
    
        iara_configure [path]
    """
    config_file = os.path.expanduser("~/.iara_config.txt")
    
    # Clean up the path
    clean_path = os.path.expanduser(installation_path.strip(" '\""))
    
    # Verify it looks correct
    if not os.path.exists(os.path.join(clean_path, "predict.py")) or not os.path.exists(os.path.join(clean_path, "IARA.pth")):
        print(f"⚠️ Warning: Could not find predict.py or IARA.pth inside {clean_path}")
        print("Please make sure you point directly to the extracted IARA folder containing those files.")
        
    try:
        with open(config_file, "w") as f:
            f.write(clean_path)
        print(f"✅ IARA configured successfully! Path saved to {config_file}")
        print("You can now run 'iara_predict' on any loaded structure.")
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")

cmd.extend("iara_configure", iara_configure)


def iara_env_name(env_name):
    """
    DESCRIPTION
    
        Overrides the default Conda environment name ('iara_env') used by the plugin.
        
    USAGE
    
        iara_env_name [my_custom_env]
    """
    env_config_file = os.path.expanduser("~/.iara_env_config.txt")
    clean_env = env_name.strip(" '\"")
    
    try:
        with open(env_config_file, "w") as f:
            f.write(clean_env)
        print(f"✅ IARA Conda environment set to '{clean_env}'")
    except Exception as e:
        print(f"❌ Error saving environment configuration: {e}")

cmd.extend("iara_env_name", iara_env_name)


# Optional: Add it to the GUI menu (Requires Tkinter which PyMOL usually has)
def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('IARA Hotspot Predictor', lambda: cmd.scene(''), 'iara_predict')
