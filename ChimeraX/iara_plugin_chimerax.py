import os
import tempfile
import subprocess
from chimerax.core.commands import CmdDesc, StringArg, register
from chimerax.atomic import AtomicStructuresArg
from chimerax.core.errors import UserError
from chimerax.core.commands import run

def iara_configure(session, path):
    config_file = os.path.expanduser("~/.iara_config.txt")
    clean_path = os.path.expanduser(path.strip(" '\""))
    
    script_path = os.path.join(clean_path, "predict.py")
    model_path = os.path.join(clean_path, "IARA.pth")
    if not os.path.exists(script_path) or not os.path.exists(model_path):
        session.logger.warning(f"Warning: Could not find predict.py or IARA.pth inside {clean_path}")
        
    try:
        with open(config_file, "w") as f:
            f.write(clean_path)
        session.logger.info(f"IARA configured successfully! Path saved to {config_file}")
    except Exception as e:
        raise UserError(f"Error saving configuration: {e}")

def iara_env_name(session, env_name):
    env_config_file = os.path.expanduser("~/.iara_env_config.txt")
    clean_env = env_name.strip(" '\"")
    try:
        with open(env_config_file, "w") as f:
            f.write(clean_env)
        session.logger.info(f"IARA Conda environment set to '{clean_env}'")
    except Exception as e:
        raise UserError(f"Error saving environment configuration: {e}")

def iara_predict(session, structures):
    if len(structures) == 0:
        raise UserError("No structures provided to predict.")
    
    structure = structures[0] # Target only the first selected model
    
    # Load Paths
    config_file = os.path.expanduser("~/.iara_config.txt")
    if not os.path.exists(config_file):
        raise UserError("IARA installation path not configured. Run: iara_configure /your/path")
    with open(config_file, "r") as f:
        plugin_dir = f.read().strip()
        
    iara_script = os.path.join(plugin_dir, "predict.py")
    iara_model = os.path.join(plugin_dir, "IARA.pth")
        
    env_config_file = os.path.expanduser("~/.iara_env_config.txt")
    conda_env = "iara_env"
    if os.path.exists(env_config_file):
        with open(env_config_file, "r") as f:
            conda_env = f.read().strip()

    # Save temp PDB
    temp_dir = tempfile.mkdtemp()
    temp_pdb = os.path.join(temp_dir, "temp_target.pdb")
    
    run(session, f"save {temp_pdb} models #{structure.id_string}")
    session.logger.info("Running IARA Neural Network...")
    
    # OS Execution Bridge
    if os.name == 'nt':
        command = ["cmd.exe", "/c", f"conda run -n {conda_env} python {iara_script} --model {iara_model} --input {temp_pdb} --outdir {temp_dir}"]
    else:
        command = ["bash", "-l", "-c", f"conda run -n {conda_env} python {iara_script} --model {iara_model} --input {temp_pdb} --outdir {temp_dir}"]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            session.logger.warning(result.stdout)
            session.logger.warning(result.stderr)
            raise UserError(f"IARA Inference Failed! Conda env '{conda_env}' may be missing. Check Log.")
    except Exception as e:
        raise UserError(f"Failed to launch external script: {e}")
        
    scored_pdb = os.path.join(temp_dir, "temp_target_IARA.pdb")
    if not os.path.exists(scored_pdb):
        raise UserError("Could not locate output file.")
        
    # Open and Render inside ChimeraX
    models = run(session, f"open {scored_pdb}")
    new_model = models[0]
    
    # Keep probabilities scaled 0-100 to match PyMOL behavior

    run(session, f"hide #{new_model.id_string} atoms")
    run(session, f"show #{new_model.id_string} cartoons")
    
    # ChimeraX's native brilliant command for coloring directly by bfactor onto a color key
    run(session, f"color bfactor #{new_model.id_string} palette blue:white:red range 0,100 key true")
    
    # Hide original to prevent 3D clipping overlap
    run(session, f"hide #{structure.id_string} models")
    session.logger.info("IARA Prediction Complete!")

# Register commands globally using the `session` object that ChimeraX automatically injects
def register_iara_commands(sess):
    try:
        predict_desc = CmdDesc(required=[("structures", AtomicStructuresArg)], synopsis="Run IARA on selected model")
        configure_desc = CmdDesc(required=[("path", StringArg)], synopsis="Set IARA path")
        env_name_desc = CmdDesc(required=[("env_name", StringArg)], synopsis="Set Conda environment")

        register("iara_predict", predict_desc, iara_predict, logger=sess.logger)
        register("iara_configure", configure_desc, iara_configure, logger=sess.logger)
        register("iara_env_name", env_name_desc, iara_env_name, logger=sess.logger)
        
        sess.logger.info("IARA Plugin commands loaded successfully! Try: iara_configure /path/to/IARA")
    except Exception as e:
        sess.logger.error(f"Failed to load IARA commands: {e}")

# The 'session' variable is provided by the ChimeraX 'open' command runtime.
register_iara_commands(session)
