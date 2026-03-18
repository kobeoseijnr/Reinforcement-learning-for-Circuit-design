"""
Original AutoCkt Evaluation Script
Runs AutoCkt evaluation using the original rollout.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_evaluation(checkpoint_path, num_val_specs=1000, traj_len=60):
    """
    Run AutoCkt evaluation using the original rollout.py script
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        num_val_specs: Number of validation specs to test (default: 1000)
        traj_len: Length of each trajectory (default: 60)
    """
    BASE_DIR = Path(__file__).parent
    
    print("="*80)
    print("ORIGINAL AUTOCKT EVALUATION")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    specs_file = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
    if not specs_file.exists():
        print(f"ERROR: Specs file not found: {specs_file}")
        return False
    
    print(f"\n[1] Checkpoint: {checkpoint_path}")
    print(f"[2] Specs file: {specs_file}")
    print(f"[3] Number of validation specs: {num_val_specs}")
    print(f"[4] Trajectory length: {traj_len}")
    
    cmd = [
        "python", 
        str(BASE_DIR / "autockt" / "rollout.py"),
        checkpoint_path,
        "--run", "PPO",
        "--env", "opamp-v0",
        "--num_val_specs", str(num_val_specs),
        "--traj_len", str(traj_len),
        "--no-render"
    ]
    
    print("\n[5] Running evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    os.chdir(BASE_DIR)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=BASE_DIR)
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print("\nResults saved to:")
        print(f"  - {BASE_DIR / 'opamp_obs_reached_test'}")
        print(f"  - {BASE_DIR / 'opamp_obs_nreached_test'}")
        print("\nNext step: Run 'python main.py' to process results")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Evaluation failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\nERROR: Python or required dependencies not found")
        return False

if __name__ == "__main__":
    print("Use main.py as the only entry point: python main.py --evaluate <checkpoint_path>")
    sys.exit(1)
