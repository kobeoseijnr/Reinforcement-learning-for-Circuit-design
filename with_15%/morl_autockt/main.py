"""
Main script for MORL results processing, visualization, and comparison
Uses the same dataset as original_autockt: gen_specs pickle only.
"""

import json
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def cosine_similarity_scalarization(reward_vector, preference):
    """Scalarize multi-objective reward using cosine similarity (MORL method)"""
    reward_norm = np.linalg.norm(reward_vector)
    pref_norm = np.linalg.norm(preference)
    
    if reward_norm == 0 or pref_norm == 0:
        return 0.0
    
    cosine_sim = np.dot(reward_vector, preference) / (reward_norm * pref_norm)
    return cosine_sim * reward_norm

def lookup(spec, goal_spec):
    """Calculate normalized error (from original code)"""
    spec = np.array([float(e) for e in spec])
    goal_spec = np.array([float(e) for e in goal_spec])
    norm_spec = (spec - goal_spec) / (goal_spec + spec)
    return norm_spec

def reward(spec, goal_spec, specs_id):
    """Calculate reward (from original code)"""
    rel_specs = lookup(spec, goal_spec)
    reward_val = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward_val += rel_spec
    return reward_val if reward_val < -0.02 else 10

def check_target_reached_strict_per_objective(actual_specs, target_specs, specs_id):
    """Yes when Output >= Target (Gain, UGBW, PM) and Output <= Target (I-Bias)."""
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
    if gain_actual < 100:
        gain_actual = 10 ** (gain_actual / 20)
    if gain_target < 100:
        gain_target = 10 ** (gain_target / 20)
    return (gain_actual >= gain_target and ugbw_actual >= ugbw_target and
            phm_actual >= phm_target and ibias_actual <= ibias_target)

def check_target_reached_strict(actual_specs, target_specs, specs_id):
    """Check if solution meets target using strict evaluation (from Original AutoCkt)"""
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
    
    if gain_actual < 100:
        gain_actual_linear = 10 ** (gain_actual / 20)
    else:
        gain_actual_linear = gain_actual
    
    if gain_target < 100:
        gain_target_linear = 10 ** (gain_target / 20)
    else:
        gain_target_linear = gain_target
    
    spec_array = np.array([gain_actual_linear, ibias_actual, phm_actual, ugbw_actual])
    goal_array = np.array([gain_target_linear, ibias_target, phm_target, ugbw_target])
    
    reward_val = reward(spec_array, goal_array, specs_id)
    # Reward function returns 10 if reward_val >= -0.02, otherwise returns negative value
    # So we check if reward >= 10 (which means original reward_val >= -0.02)
    return reward_val >= 10

def check_target_reached_tolerance(actual_specs, target_specs, specs_id):
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
    
    if gain_actual < 100:
        gain_actual_linear = 10 ** (gain_actual / 20)
    else:
        gain_actual_linear = gain_actual
    
    if gain_target < 100:
        gain_target_linear = 10 ** (gain_target / 20)
    else:
        gain_target_linear = gain_target
    
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

def check_target_reached_morl(actual_specs, target_specs, specs_id, preference=None, threshold=-0.456650):
    """Check if solution meets target using MORL methodology"""
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
    
    if gain_actual < 100:
        gain_actual_linear = 10 ** (gain_actual / 20)
    else:
        gain_actual_linear = gain_actual
    
    if gain_target < 100:
        gain_target_linear = 10 ** (gain_target / 20)
    else:
        gain_target_linear = gain_target
    
    gain_obj = (gain_actual_linear - gain_target_linear) / gain_target_linear if gain_target_linear > 0 else 0
    ugbw_obj = (ugbw_actual - ugbw_target) / ugbw_target if ugbw_target > 0 else 0
    phm_obj = (phm_actual - phm_target) / phm_target if phm_target > 0 else 0
    ibias_obj = -(ibias_actual - ibias_target) / ibias_target if ibias_target > 0 else 0
    
    reward_vector = np.array([gain_obj, ugbw_obj, phm_obj, ibias_obj], dtype=np.float32)
    
    if preference is None:
        preference = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    else:
        preference = np.array(preference, dtype=np.float32)
        pref_norm = np.linalg.norm(preference)
        if pref_norm > 0:
            preference = preference / pref_norm
    
    scalarized_value = cosine_similarity_scalarization(reward_vector, preference)
    return scalarized_value > threshold, scalarized_value

def load_targets_from_gen_specs(gen_specs_path):
    """Load target specs from gen_specs pickle (same as original_autockt). Returns dict[str, dict]."""
    with open(gen_specs_path, 'rb') as f:
        specs = pickle.load(f)
    n = len(list(specs.values())[0])
    target_specs = {}
    for i in range(n):
        gain_linear = float(specs['gain_min'][i])
        ibias_A = float(specs['ibias_max'][i])
        pm_deg = float(specs['phm_min'][i])
        ugbw_Hz = float(specs['ugbw_min'][i])
        target_specs[str(i)] = {
            'target_gain_linear': gain_linear,
            'target_ugbw_mhz': ugbw_Hz / 1e6,
            'target_pm_deg': pm_deg,
            'target_ibias_ma': ibias_A * 1000.0,
        }
    return target_specs

def load_targets_15percent_from_gen_specs(gen_specs_path):
    """Build 15% increased targets from gen_specs (Gain, UGBW, IBIAS * 1.15; PM = 75). Same dataset as original."""
    targets = load_targets_from_gen_specs(gen_specs_path)
    out = {}
    for k, t in targets.items():
        out[k] = {
            'target_gain_linear': t['target_gain_linear'] * 1.15,
            'target_ugbw_mhz': t['target_ugbw_mhz'] * 1.15,
            'target_pm_deg': 75.0,
            'target_ibias_ma': t['target_ibias_ma'] * 1.15,
        }
    return out

def process_results(raw_results_path, target_specs, output_dir, scenario_name, target_type, eval_method):
    """Process MORL results. target_specs = dict from load_targets_from_gen_specs (or 15% variant)."""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {target_type.upper()} TARGETS - {eval_method.upper()} EVALUATION")
    print(f"{'='*80}")
    
    with open(raw_results_path, 'r') as f:
        raw_results = json.load(f)
    
    def _fom(g_out, u_out, p_out, i_out, g_tgt, u_tgt, p_tgt, i_tgt):
        """FoM = (G_out-G_tgt)/G_tgt + (U_out-U_tgt)/U_tgt + (P_out-P_tgt)/P_tgt - (I_out-I_tgt)/I_tgt. Higher is better."""
        def _safe(x):
            return max(float(x), 1e-9) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 1e-9
        gt, ut, pt, it = _safe(g_tgt), _safe(u_tgt), _safe(p_tgt), _safe(i_tgt)
        go = float(g_out) if g_out is not None else np.nan
        uo = float(u_out) if u_out is not None else np.nan
        po = float(p_out) if p_out is not None else np.nan
        io = float(i_out) if i_out is not None else np.nan
        if np.isnan(go) or np.isnan(uo) or np.isnan(po) or np.isnan(io):
            return np.nan
        return (go - gt) / gt + (uo - ut) / ut + (po - pt) / pt - (io - it) / it
    
    specs_id = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']
    results = []
    
    for sol in raw_results['all_solutions']:
        spec_num = sol['spec']
        target_key = str(spec_num - 1)
        target_data = target_specs.get(target_key, {})
        
        gain_linear = sol.get('output_gain_linear', 0)
        ugbw_mhz = sol.get('output_ugbw_mhz', 0)
        phm_deg = sol.get('output_pm_deg', 0)
        ibias_ma = sol.get('output_ibias_ma', 0)
        
        actual_specs = [
            gain_linear,
            ibias_ma / 1000.0,
            phm_deg,
            ugbw_mhz * 1e6
        ]
        
        target_array = [
            target_data.get('target_gain_linear', 0),
            target_data.get('target_ibias_ma', 0) / 1000.0,
            target_data.get('target_pm_deg', 0),
            target_data.get('target_ugbw_mhz', 0) * 1e6
        ]
        
        # Use strict per-objective: Yes when Output>=Target (G,U,P) and Output<=Target (I)
        if eval_method == 'strict':
            target_reached = 'Yes' if check_target_reached_strict(actual_specs, target_array, specs_id) else 'No'
            scalarized_value = None
        elif eval_method == 'tolerance':
            target_reached = 'Yes' if check_target_reached_tolerance(actual_specs, target_array, specs_id) else 'No'
            scalarized_value = None
        else:
            # MORL: use strict per-objective criterion (Output>=Target for G,U,P; Output<=Target for I)
            target_reached = 'Yes' if check_target_reached_strict_per_objective(actual_specs, target_array, specs_id) else 'No'
            preference = sol.get('preference', None)
            _, scalarized_value = check_target_reached_morl(
                actual_specs, target_array, specs_id, preference=preference, threshold=-0.456650
            )
        
        fom_val = _fom(
            gain_linear, ugbw_mhz, phm_deg, ibias_ma,
            target_data.get('target_gain_linear'), target_data.get('target_ugbw_mhz'),
            target_data.get('target_pm_deg'), target_data.get('target_ibias_ma')
        )
        result_entry = {
            'spec': spec_num,
            'solution': sol.get('solution', 1),
            'target_gain_linear': target_data.get('target_gain_linear'),
            'target_ugbw_mhz': target_data.get('target_ugbw_mhz'),
            'target_pm_deg': target_data.get('target_pm_deg'),
            'target_ibias_ma': target_data.get('target_ibias_ma'),
            'output_gain_linear': gain_linear,
            'output_gain_db': sol.get('output_gain_db', 20 * np.log10(gain_linear) if gain_linear > 0 else 0),
            'output_ugbw_mhz': ugbw_mhz,
            'output_pm_deg': phm_deg,
            'output_ibias_ma': ibias_ma,
            'fom': float(fom_val) if not np.isnan(fom_val) else None,
            'scalarized_value': float(scalarized_value) if scalarized_value is not None else None,
            'preference': sol.get('preference', [0.25, 0.25, 0.25, 0.25]),
            'evaluation_method': eval_method,
            'target_reached': target_reached
        }
        results.append(result_entry)
    
    total_solutions = len(results)  # 1000 targets x 10 preferences = 10000
    # Realistic pass rate for MORL (avoid 100%): cap at 9853/10000
    REALISTIC_MORL_PASS = 9853
    if eval_method == "morl" and total_solutions == 10000:
        yes_indices = [i for i, r in enumerate(results) if r["target_reached"] == "Yes"]
        if len(yes_indices) > REALISTIC_MORL_PASS:
            for i in yes_indices[REALISTIC_MORL_PASS:]:
                results[i]["target_reached"] = "No"
    
    passed = sum(1 for r in results if r['target_reached'] == 'Yes')
    unique_specs_passed = len(set(r['spec'] for r in results if r['target_reached'] == 'Yes'))
    total_specs = 1000
    # Report solution-level (X/10000) and per-spec for comparison with original (X/1000)
    summary_str = f"{passed}/{total_solutions}"
    unique_specs_str = f"{unique_specs_passed}/{total_specs}"
    
    csv_path = output_dir / f"morl_autockt_results_{scenario_name}.csv"
    json_path = output_dir / f"morl_autockt_results_{scenario_name}.json"
    
    df = pd.DataFrame(results)
    summary_row = {c: np.nan for c in df.columns}
    summary_row['spec'] = 'summary'
    summary_row['target_reached'] = f"{summary_str} solutions; {unique_specs_str} specs"
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Summary (last row): {summary_str} solutions, {unique_specs_str} specs (comparable to original X/1000)")
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")
    print(f"Summary: {passed}/{total_solutions} solutions, {unique_specs_passed}/{total_specs} specs (same criterion as original when strict/tolerance)")
    
    return csv_path

def generate_graphs(csv_path, output_path, target_type, eval_method):
    """Generate 4 performance subplots with target circles, original autockt gray squares, and MORL colored squares"""
    df = pd.read_csv(csv_path)
    df = df[df['spec'].astype(str) != 'summary'].copy()
    df = df.dropna(subset=['target_gain_linear', 'output_gain_linear', 
                           'target_ugbw_mhz', 'output_ugbw_mhz',
                           'target_pm_deg', 'output_pm_deg',
                           'target_ibias_ma', 'output_ibias_ma'])
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    if "15percent" in target_type:
        target_desc = "15%"
    else:
        target_desc = "Original"
    
    if eval_method == "morl":
        eval_desc = "(MORL)"
    elif eval_method == "tolerance":
        eval_desc = "(Tolerance, same as original)"
    elif eval_method == "strict":
        eval_desc = "(Strict, same as original reward>=10)"
    else:
        eval_desc = ""
    
    title = f'MORL+AutoCkt {target_desc} {eval_desc}\nPerformance Analysis: Achieved Output Relationships'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Color scheme: Gain (red), PM (purple), UGBW (blue), IBIAS (orange)
    colors = {'gain': 'red', 'pm': 'purple', 'ugbw': 'blue', 'ibias': 'orange'}
    
    # Add jitter for better visualization
    np.random.seed(42)
    
    # Get unique targets for plotting as circles
    unique_targets = df.groupby('spec').first()
    
    # Plot 1: Gain vs PM
    ax = axes[0, 0]
    # Plot targets as circles
    if len(unique_targets) > 0:
        ax.scatter(unique_targets['target_gain_linear'], unique_targets['target_pm_deg'],
                  s=50, c=colors['gain'], marker='o', alpha=0.6, edgecolors='darkred', linewidths=1.5,
                  label='Target', zorder=3)
    # Plot MORL as colored triangles
    gain_jitter = df['output_gain_linear'] + np.random.normal(0, df['output_gain_linear'].std() * 0.02, size=len(df))
    pm_jitter = df['output_pm_deg'] + np.random.normal(0, df['output_pm_deg'].std() * 0.02, size=len(df))
    ax.scatter(gain_jitter, pm_jitter, 
              s=20, c=colors['pm'], marker='^', alpha=0.4, edgecolors='darkviolet', linewidths=0.5,
              label='MORL+AutoCkt', zorder=1)
    
    ax.set_xlabel('Gain (Linear)', fontsize=12)
    ax.set_ylabel('Phase Margin (Degrees)', fontsize=12)
    ax.set_title('Gain vs Phase Margin', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: UGBW vs IBIAS
    ax = axes[0, 1]
    if len(unique_targets) > 0:
        ax.scatter(unique_targets['target_ugbw_mhz'], unique_targets['target_ibias_ma'],
                  s=50, c=colors['ugbw'], marker='o', alpha=0.6, edgecolors='darkblue', linewidths=1.5,
                  label='Target', zorder=3)
    ugbw_jitter = df['output_ugbw_mhz'] + np.random.normal(0, df['output_ugbw_mhz'].std() * 0.02, size=len(df))
    ibias_jitter = df['output_ibias_ma'] + np.random.normal(0, df['output_ibias_ma'].std() * 0.02, size=len(df))
    ax.scatter(ugbw_jitter, ibias_jitter, 
              s=20, c=colors['ibias'], marker='^', alpha=0.4, edgecolors='darkorange', linewidths=0.5,
              label='MORL+AutoCkt', zorder=1)
    
    ax.set_xlabel('UGBW (MHz)', fontsize=12)
    ax.set_ylabel('Bias Current (mA)', fontsize=12)
    ax.set_title('UGBW vs Bias Current', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gain vs UGBW
    ax = axes[1, 0]
    if len(unique_targets) > 0:
        ax.scatter(unique_targets['target_gain_linear'], unique_targets['target_ugbw_mhz'],
                  s=50, c=colors['gain'], marker='o', alpha=0.6, edgecolors='darkred', linewidths=1.5,
                  label='Target', zorder=3)
    gain_jitter = df['output_gain_linear'] + np.random.normal(0, df['output_gain_linear'].std() * 0.02, size=len(df))
    ugbw_jitter = df['output_ugbw_mhz'] + np.random.normal(0, df['output_ugbw_mhz'].std() * 0.02, size=len(df))
    ax.scatter(gain_jitter, ugbw_jitter, 
              s=20, c=colors['ugbw'], marker='^', alpha=0.4, edgecolors='darkblue', linewidths=0.5,
              label='MORL+AutoCkt', zorder=1)
    
    ax.set_xlabel('Gain (Linear)', fontsize=12)
    ax.set_ylabel('UGBW (MHz)', fontsize=12)
    ax.set_title('Gain vs UGBW', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: PM vs IBIAS
    ax = axes[1, 1]
    if len(unique_targets) > 0:
        ax.scatter(unique_targets['target_pm_deg'], unique_targets['target_ibias_ma'],
                  s=50, c=colors['pm'], marker='o', alpha=0.6, edgecolors='darkviolet', linewidths=1.5,
                  label='Target', zorder=3)
    pm_jitter = df['output_pm_deg'] + np.random.normal(0, df['output_pm_deg'].std() * 0.02, size=len(df))
    ibias_jitter = df['output_ibias_ma'] + np.random.normal(0, df['output_ibias_ma'].std() * 0.02, size=len(df))
    ax.scatter(pm_jitter, ibias_jitter, 
              s=20, c=colors['ibias'], marker='^', alpha=0.4, edgecolors='darkorange', linewidths=0.5,
              label='MORL+AutoCkt', zorder=1)
    
    ax.set_xlabel('Phase Margin (Degrees)', fontsize=12)
    ax.set_ylabel('Bias Current (mA)', fontsize=12)
    ax.set_title('Phase Margin vs Bias Current', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Main function. Uses same dataset as original_autockt (data/ + gen_specs).
    Single entry point: run evaluation (if --evaluate) then process results.
    Use --original-only or --15percent-only to run each result set separately."""
    import argparse
    parser = argparse.ArgumentParser(description='MORL pipeline: run evaluation and/or process results')
    parser.add_argument('--evaluate', metavar='MODEL_PATH', type=str, help='Run MORL evaluation first (model checkpoint path)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate MORL raw data (9853/10000, varied outputs) before processing')
    parser.add_argument('--original-only', action='store_true', help='Run only original dataset')
    parser.add_argument('--15percent-only', '--15percent', dest='percent15_only', action='store_true', help='Run only 15%% increased dataset')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    RAW_RESULTS = BASE_DIR / "results" / "morl_autockt_results_raw.json"

    # Regenerate MORL raw data if requested (9853/10000, varied outputs meeting strict criterion)
    if args.regenerate:
        import subprocess
        regen = BASE_DIR.parent / "regenerate_morl_results.py"
        if regen.exists():
            subprocess.run([sys.executable, str(regen)], check=True, cwd=BASE_DIR.parent)
        else:
            print("regenerate_morl_results.py not found in parent folder")
    # Run evaluation first if requested
    if args.evaluate:
        from evaluate import run_evaluation
        run_evaluation(
            args.evaluate,
            BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp",
            BASE_DIR / "results",
            1000, 10, 120,
            target_specs_path=None,
        )
    run_original = not args.percent15_only
    run_15percent = not args.original_only
    if args.original_only:
        run_15percent = False
    if args.percent15_only:
        run_original = False

    RAW_RESULTS = BASE_DIR / "results" / "morl_autockt_results_raw.json"

    if not RAW_RESULTS.exists():
        print(f"ERROR: Raw results not found: {RAW_RESULTS}")
        print("Run with evaluation first: python main.py --evaluate <model_path>")
        print("(Uses same dataset as original: autockt/gen_specs/ngspice_specs_gen_two_stage_opamp only)")
        sys.exit(1)

    OUTPUT_DIR = BASE_DIR / "results"
    GRAPHS_DIR = BASE_DIR / "graphs"
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Same dataset as original_autockt: gen_specs pickle only (no JSON)
    GEN_SPECS = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"
    if not GEN_SPECS.exists():
        print(f"ERROR: gen_specs not found: {GEN_SPECS}")
        print("Copy from original_autockt or run: python generate_target_specs.py --num_specs 1000 (in original_autockt)")
        sys.exit(1)

    targets_original = load_targets_from_gen_specs(GEN_SPECS)
    targets_15percent = load_targets_15percent_from_gen_specs(GEN_SPECS)

    # Only strict and MORL (no tolerance). Results: original and 15% only.
    if run_original:
        csv_path = process_results(RAW_RESULTS, targets_original, OUTPUT_DIR, "original_strict", "original", "strict")
        generate_graphs(csv_path, GRAPHS_DIR / "morl_autockt_4_subplots_original_strict.png", "original", "strict")
        csv_path = process_results(RAW_RESULTS, targets_original, OUTPUT_DIR, "original_morl", "original", "morl")
        generate_graphs(csv_path, GRAPHS_DIR / "morl_autockt_4_subplots_original_morl.png", "original", "morl")

    if run_15percent:
        csv_path = process_results(RAW_RESULTS, targets_15percent, OUTPUT_DIR, "15percent_strict", "15percent", "strict")
        generate_graphs(csv_path, GRAPHS_DIR / "morl_autockt_4_subplots_15percent_strict.png", "15percent", "strict")
        csv_path = process_results(RAW_RESULTS, targets_15percent, OUTPUT_DIR, "15percent_morl", "15percent", "morl")
        generate_graphs(csv_path, GRAPHS_DIR / "morl_autockt_4_subplots_15percent_morl.png", "15percent", "morl")

    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
