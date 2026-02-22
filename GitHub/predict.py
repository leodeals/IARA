import os
import sys
import glob
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from prody import parsePDB, writePDB, confProDy
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

warnings.filterwarnings('ignore')
confProDy(verbosity='none')

# --- AA LOOKUPS ---
MAX_SASA = {'ALA':129,'ARG':274,'ASN':195,'ASP':193,'CYS':167,'GLN':225,'GLU':223,'GLY':104,'HIS':224,'ILE':197,'LEU':201,'LYS':236,'MET':224,'PHE':240,'PRO':159,'SER':155,'THR':172,'TRP':285,'TYR':263,'VAL':174}
AA_PROPS = {'ALA':{'h':1.8,'c':0},'ARG':{'h':-4.5,'c':1},'ASN':{'h':-3.5,'c':0},'ASP':{'h':-3.5,'c':-1},'CYS':{'h':2.5,'c':0},'GLN':{'h':-3.5,'c':0},'GLU':{'h':-3.5,'c':-1},'GLY':{'h':-0.4,'c':0},'HIS':{'h':-3.2,'c':0},'ILE':{'h':4.5,'c':0},'LEU':{'h':3.8,'c':0},'LYS':{'h':-3.9,'c':1},'MET':{'h':1.9,'c':0},'PHE':{'h':2.8,'c':0},'PRO':{'h':-1.6,'c':0},'SER':{'h':-0.8,'c':0},'THR':{'h':-0.7,'c':0},'TRP':{'h':-0.9,'c':0},'TYR':{'h':-1.3,'c':0},'VAL':{'h':4.2,'c':0}}

# --- MODEL DEFINITION ---
class BindGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(7, 64, heads=4)
        self.conv2 = GATv2Conv(64*4, 64, heads=4)
        self.conv3 = GATv2Conv(64*4, 32, heads=1)
        self.out   = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))

    def forward(self, data):
        x, e = data.x, data.edge_index
        x = torch.nn.functional.elu(self.conv1(x, e))
        x = torch.nn.functional.elu(self.conv2(x, e))
        x = torch.nn.functional.elu(self.conv3(x, e))
        return self.out(x)

# --- INFERENCE HELPERS ---
def extract_features(ca_atoms, coords, distance_threshold=10.0):
    tree = cKDTree(coords)
    dist_m = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    dens8 = np.sum(dist_m < 8.0, axis=1) - 1
    dens15 = np.sum(dist_m < 15.0, axis=1) - 1
    charges = [AA_PROPS.get(a.getResname(), {'c': 0})['c'] for a in ca_atoms]
    
    x_list = []
    for i, atom in enumerate(ca_atoms):
        res = atom.getResname()
        sasa_p = (1.0 - (dens8[i]/15.0)) * 100.0
        rsasa = sasa_p / MAX_SASA.get(res, 1.0)
        p = AA_PROPS.get(res, {'h':0,'c':0})
        patch = np.mean([charges[j] for j in tree.query_ball_point(coords[i], 10.0)])/0.12
        x_list.append([p['h'], p['c'], sasa_p/100.0, rsasa, dens8[i]/10.0, dens15[i]/50.0, patch])
    
    pairs = list(tree.query_pairs(distance_threshold))
    if not pairs:
        return None, None
    
    ei = torch.tensor(np.array(pairs).T, dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    return torch.tensor(x_list, dtype=torch.float), ei

def smooth_predictions(coords, probs, radius=8.0):
    tree = cKDTree(coords)
    smoothed = np.zeros_like(probs)
    for i in range(len(probs)):
        nb = tree.query_ball_point(coords[i], radius)
        smoothed[i] = (probs[i] + np.mean(probs[nb])) / 2.0
    return smoothed

def score_structure(input_file, model, device, output_dir):
    try:
        struct = parsePDB(input_file)
        if struct is None:
            print(f"   ⚠️  Failed to parse {input_file}, skipping.")
            return

        ca_atoms = struct.select('name CA and protein')
        if ca_atoms is None or len(ca_atoms) < 20:
            print(f"   ⚠️  Not enough CA atoms found in {input_file}, skipping.")
            return

        coords = ca_atoms.getCoords()
        x, ei = extract_features(ca_atoms, coords)
        
        if x is None:
            print(f"   ⚠️  Failed to extract structural features for {input_file}, skipping.")
            return

        x_tensor = x.to(device)
        ei_tensor = ei.to(device)

        with torch.no_grad():
            data = Data(x=x_tensor, edge_index=ei_tensor)
            raw_probs = torch.sigmoid(model(data)).cpu().numpy().flatten()

        smooth_p = smooth_predictions(coords, raw_probs)

        # Clear existing B-factors and replace with probabilities scaled (0-100)
        sv = struct.copy()
        sv.setBetas(0)

        for i, p in enumerate(smooth_p):
            sel = sv.select(f"resindex {ca_atoms[i].getResindex()}")
            if sel: 
                sel.setBetas(float(p) * 100.0)

        # Build output filename
        file_path = Path(input_file)
        out_name = f"{file_path.stem}_IARA.pdb"
        out_path = os.path.join(output_dir, out_name)

        writePDB(out_path, sv)
        print(f"   ✅ Scored {file_path.name} -> Saved to {out_path}")

    except Exception as e:
        print(f"   ❌ Error processing {input_file}: {e}")

# --- MAIN RUNNER ---
def main():
    parser = argparse.ArgumentParser(description="IARA Inference Tool - Interface Analysis and Recognition Architecture")
    parser.add_argument("-i", "--input", required=True, help="Input directory OR a single .pdb / .cif file")
    parser.add_argument("-o", "--outdir", default="scored_predictions", help="Output directory to save scored PDBs")
    parser.add_argument("-m", "--model", required=True, help="Path to the IARA.pth model file")
    
    args = parser.parse_args()

    print("\n🌊 IARA Inference (Interface Analysis and Recognition Architecture)\n")

    # 1. Setup Environment
    if not os.path.exists(args.model):
        print(f"❌ Error: Model weights not found at {args.model}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Hardware: Using {device}")

    # 2. Load Model
    model = BindGNN()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully\n")

    # 3. Setup Input Files
    os.makedirs(args.outdir, exist_ok=True)
    input_path = Path(args.input)
    files_to_process = []

    if input_path.is_file():
        files_to_process.append(str(input_path))
    elif input_path.is_dir():
        # Find all PDB and CIF files in directory
        files_to_process.extend(glob.glob(os.path.join(args.input, "*.pdb")))
        files_to_process.extend(glob.glob(os.path.join(args.input, "*.cif")))
        files_to_process.extend(glob.glob(os.path.join(args.input, "*.pdb.gz")))
        files_to_process.extend(glob.glob(os.path.join(args.input, "*.ent")))
    else:
        print(f"❌ Error: Invalid input path {args.input}")
        sys.exit(1)

    if not files_to_process:
        print(f"⚠️  No valid PDB or CIF structures found in {args.input}")
        sys.exit(0)

    print(f"🚀 Processing {len(files_to_process)} structure(s)...\n")
    
    # 4. Process Inputs
    for file_path in files_to_process:
        score_structure(file_path, model, device, args.outdir)

    print(f"\n✨ Complete! All predicted structures saved to '{args.outdir}/'.")
    print("🎨 Open these structures in PyMOL and color by B-factor to visualize the hotspots!")

if __name__ == "__main__":
    main()
