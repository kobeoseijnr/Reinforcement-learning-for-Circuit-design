"""
MORL Evaluation Script
Evaluates MORL+AutoCkt model on the dataset
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Add methodology to path
BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))

# Enable NGSpice simulation environment
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

# Import MORL components
try:
    from autockt.envs.autockt_mo_env import AutoCktMOEnv
    from autockt.agents.mo_agent import MO_DQN_Agent
    from autockt.utils.mo_utils import generate_preference_vectors
    print("MORL components imported successfully\n")
except ImportError as e:
    print(f"ERROR: Could not import MORL components: {e}")
    sys.exit(1)

def load_trained_model(model_path):
    """Load trained MORL model"""
    print(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get dimensions from checkpoint or use defaults
        state_dim = checkpoint.get('state_dim', 64)
        action_dim = checkpoint.get('action_dim', 64)
        reward_dim = checkpoint.get('reward_dim', 4)
        
        print(f"Model dimensions: state_dim={state_dim}, action_dim={action_dim}, reward_dim={reward_dim}")
        
        agent = MO_DQN_Agent(state_dim=state_dim, action_dim=action_dim, reward_dim=reward_dim)
        
        # Try different checkpoint formats
        if 'agent_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
            if 'preference_net_state_dict' in checkpoint:
                agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
        elif 'model_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['state_dict'])
        elif 'q_network_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        else:
            # Try to load directly (might be just the state dict)
            try:
                agent.q_network.load_state_dict(checkpoint)
            except:
                # If that fails, check if it's nested
                if isinstance(checkpoint, dict) and len(checkpoint) > 0:
                    # Try first key
                    first_key = list(checkpoint.keys())[0]
                    if isinstance(checkpoint[first_key], dict):
                        agent.q_network.load_state_dict(checkpoint[first_key])
                    else:
                        raise ValueError("Could not find model state dict in checkpoint")
        
        agent.q_network.eval()
        print("Model loaded successfully")
        return agent
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def check_target_reached(actual_specs, target_specs, specs_id):
    """Check if solution meets target using tolerance-based evaluation"""
    actual_dict = dict(zip(specs_id, actual_specs))
    target_dict = dict(zip(specs_id, target_specs))
    
    gain_actual = actual_dict.get('gain_min', actual_specs[0] if len(actual_specs) > 0 else 0)
    ugbw_actual = actual_dict.get('ugbw_min', actual_specs[3] if len(actual_specs) > 3 else 0)
    phm_actual = actual_dict.get('phm_min', actual_specs[2] if len(actual_specs) > 2 else 0)
    ibias_actual = actual_dict.get('ibias_max', actual_specs[1] if len(actual_specs) > 1 else 0)
    
    gain_target = target_dict.get('gain_min', target_specs[0] if len(target_specs) > 0 else 0)
    ugbw_target = target_dict.get('ugbw_min', target_specs[3] if len(target_specs) > 3 else 0)
    phm_target = target_dict.get('phm_min', target_specs[2] if len(target_specs) > 2 else 0)
    ibias_target = target_dict.get('ibias_max', target_specs[1] if len(target_specs) > 1 else 0)
    
    # Convert gain to linear if needed
    if gain_actual < 100:
        gain_actual_linear = 10 ** (gain_actual / 20)
    else:
        gain_actual_linear = gain_actual
    
    if gain_target < 100:
        gain_target_linear = 10 ** (gain_target / 20)
    else:
        gain_target_linear = gain_target
    
    # Tolerance-based evaluation
    tolerance = 0.15
    gain_ok = gain_actual_linear >= gain_target_linear * (1 - tolerance)
    ugbw_ok = ugbw_actual >= ugbw_target * (1 - tolerance)
    phm_ok = phm_actual >= phm_target * (1 - tolerance)
    ibias_ok = ibias_actual <= ibias_target * (1 + tolerance)
    
    if not (gain_ok and ugbw_ok and phm_ok and ibias_ok):
        close_tolerance = 0.05
        gain_close = gain_actual_linear >= gain_target_linear * (1 - tolerance - close_tolerance)
        ugbw_close = ugbw_actual >= ugbw_target * (1 - tolerance - close_tolerance)
        phm_close = phm_actual >= phm_target * (1 - tolerance - close_tolerance)
        ibias_close = ibias_actual <= ibias_target * (1 + tolerance + close_tolerance)
        
        met_count = sum([gain_ok, ugbw_ok, phm_ok, ibias_ok])
        close_count = sum([gain_close, ugbw_close, phm_close, ibias_close])
        if met_count >= 3 and close_count == 4:
            gain_ok = gain_close
            ugbw_ok = ugbw_close
            phm_ok = phm_close
            ibias_ok = ibias_close
    
    return gain_ok and ugbw_ok and phm_ok and ibias_ok

def evaluate_spec_morl(env, agent, target_spec, specs_id, max_steps=120, num_preferences=10, early_stop=True):
    """Evaluate one spec with MORL methodology"""
    try:
        env.base_env.specs_ideal = np.array(target_spec)
        env.reset()
    except Exception as e:
        print(f"ERROR: Failed to set specs_ideal or reset: {e}")
        return False, []
    
    # Generate preference vectors
    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=10)
        if len(preferences) > 10:
            preferences = preferences[:10]
    except:
        preferences = generate_preference_vectors(4, method='random', num_vectors=10)
    
    pareto_solutions = []
    reached = False
    
    for pref_idx, preference in enumerate(preferences):
        if early_stop and reached:
            break
        
        try:
            agent.set_preference(preference)
            state = env.reset()
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                action = agent.select_action(state, preference)
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                elif len(step_result) == 4:
                    next_state, reward, done, info = step_result
                else:
                    print(f"  Error: Unexpected step return: {len(step_result)} values")
                    break
                
                state = next_state
                steps += 1
            
            # Get final specs
            if hasattr(env.base_env, 'cur_specs'):
                actual_specs = env.base_env.cur_specs.copy()
                solution_reached = check_target_reached(actual_specs, target_spec, specs_id)
                if solution_reached:
                    reached = True
                
                pareto_solutions.append({
                    'actual_specs': actual_specs.tolist() if hasattr(actual_specs, 'tolist') else list(actual_specs),
                    'preference': preference.tolist() if hasattr(preference, 'tolist') else list(preference),
                    'target_reached': solution_reached,
                    'steps': steps
                })
        except Exception as e:
            print(f"  Error evaluating preference {pref_idx}: {e}")
            continue
    
    return reached, pareto_solutions

def run_evaluation(model_path, dataset_path, output_dir, num_targets=1000, num_preferences=10, max_steps=120, target_specs_path=None):
    """Run MORL evaluation on the dataset"""
    print("="*80)
    print("MORL EVALUATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load dataset (same as original_autockt: gen_specs pickle only, no JSON)
    print("[1] Loading dataset (gen_specs pickle, same as original)...")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        specs = pickle.load(f)
    
    num_specs = len(list(specs.values())[0])
    print(f"Loaded {num_specs} target specifications from dataset")
    
    # Build target specs from pickle (same format as original_autockt; no separate JSON)
    target_specs_json = {}
    for i in range(num_specs):
        gain_linear = float(specs['gain_min'][i])
        ugbw_Hz = float(specs['ugbw_min'][i])
        pm_deg = float(specs['phm_min'][i])
        ibias_A = float(specs['ibias_max'][i])
        target_specs_json[str(i)] = {
            'target_gain_linear': gain_linear,
            'target_ugbw_mhz': ugbw_Hz / 1e6,
            'target_pm_deg': pm_deg,
            'target_ibias_ma': ibias_A * 1000.0,
        }
    
    # Setup environment
    print("\n[2] Initializing environment...")
    env = AutoCktMOEnv(generalize=True, num_valid=num_specs, run_valid=True)
    print("Environment initialized")
    
    # Load model
    print("\n[3] Loading trained model...")
    agent = load_trained_model(model_path)
    
    # Evaluation
    print("\n[4] Starting evaluation...")
    print(f"Evaluating {num_targets} targets with {num_preferences} preference vectors each...")
    print(f"Max steps per preference: {max_steps}")
    print("="*80)
    
    results = {
        'all_solutions': [],
        'reached_count': 0,
        'total_evaluated': 0
    }
    
    # Evaluate each target
    for target_idx in range(min(num_targets, num_specs)):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"Progress: {target_idx}/{num_targets} ({target_idx*100/num_targets:.1f}%)")
        
        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            target_spec = env.base_env.specs_ideal.copy()
            specs_id = env.base_env.specs_id
            
            # Evaluate
            reached, pareto_solutions = evaluate_spec_morl(
                env, agent, target_spec, specs_id,
                max_steps, num_preferences, early_stop=True
            )
            
            # Process solutions
            target_key = str(target_idx)
            target_data = target_specs_json.get(target_key, {})
            
            for sol_idx, sol in enumerate(pareto_solutions):
                actual_specs = sol.get('actual_specs', None)
                if actual_specs is None:
                    continue
                
                # Extract values
                spec_dict = dict(zip(specs_id, actual_specs))
                gain_val = spec_dict.get('gain', spec_dict.get('gain_min', actual_specs[0] if len(actual_specs) > 0 else 0))
                ugbw_val = spec_dict.get('ugbw', spec_dict.get('ugbw_min', actual_specs[1] if len(actual_specs) > 1 else 0))
                phm_val = spec_dict.get('phm', spec_dict.get('phm_min', actual_specs[2] if len(actual_specs) > 2 else 0))
                ibias_val = spec_dict.get('ibias_max', actual_specs[3] if len(actual_specs) > 3 else 0)
                
                # Convert gain
                if gain_val < 100:
                    gain_linear = 10 ** (gain_val / 20)
                    gain_db = gain_val
                else:
                    gain_linear = gain_val
                    gain_db = 20 * np.log10(gain_val) if gain_val > 0 else 0
                
                results['all_solutions'].append({
                    'spec': target_idx + 1,
                    'solution': sol_idx + 1,
                    'target_gain_linear': target_data.get('target_gain_linear'),
                    'target_ugbw_mhz': target_data.get('target_ugbw_mhz'),
                    'target_pm_deg': target_data.get('target_pm_deg'),
                    'target_ibias_ma': target_data.get('target_ibias_ma'),
                    'output_gain_linear': float(gain_linear),
                    'output_gain_db': float(gain_db),
                    'output_ugbw_mhz': float(ugbw_val / 1e6),
                    'output_pm_deg': float(phm_val),
                    'output_ibias_ma': float(ibias_val * 1000),
                    'target_reached': bool(sol.get('target_reached', False)),
                    'preference': sol.get('preference', [0.25, 0.25, 0.25, 0.25])
                })
            
            if reached:
                results['reached_count'] += 1
            results['total_evaluated'] += 1
            
        except Exception as e:
            print(f"  Error evaluating spec {target_idx}: {e}")
            continue
    
    # Save results
    print("\n[5] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "morl_autockt_results_raw.json"
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        return obj
    
    with open(json_path, 'w') as f:
        results_python = convert_to_python(results)
        json.dump(results_python, f, indent=2)
    print(f"Saved: {json_path}")
    
    # Save CSV
    if results['all_solutions']:
        df = pd.DataFrame(results['all_solutions'])
        csv_path = output_dir / "morl_autockt_results_raw.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total targets evaluated: {results['total_evaluated']}")
    print(f"Targets reached: {results['reached_count']}")
    print(f"Success rate: {results['reached_count']/results['total_evaluated']*100:.2f}%")
    print(f"Total solutions: {len(results['all_solutions'])}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    print("Use main.py as the only entry point: python main.py --evaluate <model_path>")
    sys.exit(1)
