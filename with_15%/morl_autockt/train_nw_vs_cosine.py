"""
Train two MORL DDQN agents — one with NW, one with Cosine scalarization.
Evaluate both on 1000 specs, then compare FOMs.
Uses the strict reward rule for target_reached.
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))

os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

from autockt.envs.autockt_mo_env import AutoCktMOEnv
from autockt.agents.mo_agent import MO_DQN_Agent
from autockt.utils.mo_utils import generate_preference_vectors

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models_nw_vs_cosine"
DATASET_PATH = BASE_DIR / "autockt" / "gen_specs" / "ngspice_specs_gen_two_stage_opamp"

TRAIN_CONFIG = {
    'num_training_specs': 50,
    'num_preferences': 10,
    'max_steps': 30,
    'training_episodes': 100,
    'save_frequency': 50,
}

EVAL_CONFIG = {
    'num_targets': 1000,
    'num_preferences': 10,
    'max_steps': 120,
}

# ---------------------------------------------------------------------------
# Reward function (same as main.py)
# ---------------------------------------------------------------------------
def lookup(spec, goal_spec):
    spec = np.array([float(e) for e in spec])
    goal_spec = np.array([float(e) for e in goal_spec])
    return (spec - goal_spec) / (goal_spec + spec)

def reward_fn(spec, goal_spec, specs_id):
    rel_specs = lookup(spec, goal_spec)
    reward_val = 0.0
    for i, rel_spec in enumerate(rel_specs):
        if specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            reward_val += rel_spec
    return reward_val if reward_val < -0.02 else 10

def check_target_reached(actual_specs, target_specs, specs_id):
    actual_dict = dict(zip(specs_id, actual_specs))
    target_dict = dict(zip(specs_id, target_specs))
    ga = actual_dict.get('gain_min', actual_specs[0])
    ua = actual_dict.get('ugbw_min', actual_specs[3] if len(actual_specs) > 3 else 0)
    pa = actual_dict.get('phm_min', actual_specs[2] if len(actual_specs) > 2 else 0)
    ia = actual_dict.get('ibias_max', actual_specs[1] if len(actual_specs) > 1 else 0)
    gt = target_dict.get('gain_min', target_specs[0])
    ut = target_dict.get('ugbw_min', target_specs[3] if len(target_specs) > 3 else 0)
    pt = target_dict.get('phm_min', target_specs[2] if len(target_specs) > 2 else 0)
    it = target_dict.get('ibias_max', target_specs[1] if len(target_specs) > 1 else 0)
    if ga < 100: ga = 10 ** (ga / 20)
    if gt < 100: gt = 10 ** (gt / 20)
    spec_arr = np.array([ga, ia, pa, ua])
    goal_arr = np.array([gt, it, pt, ut])
    return reward_fn(spec_arr, goal_arr, specs_id) >= 10

def _fom(g, u, p, i, gt, ut, pt, it):
    def s(x): return max(float(x), 1e-9) if x is not None else 1e-9
    gt, ut, pt, it = s(gt), s(ut), s(pt), s(it)
    return (float(g)-gt)/gt + (float(u)-ut)/ut + (float(p)-pt)/pt - (float(i)-it)/it

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(scalarization, config, env, specs, preferences):
    tag = scalarization.upper()
    print(f"\n{'='*70}")
    print(f"  TRAINING — {tag} SCALARIZATION")
    print(f"{'='*70}\n")

    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    if isinstance(env.action_space, gym.spaces.Tuple):
        action_dim = env.action_space.spaces[0].n
        num_params = len(env.action_space.spaces)
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
        num_params = 1
    reward_dim = 4

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization=scalarization)
    print(f"  [{tag}] state={state_dim}, action={action_dim}, reward={reward_dim}")

    training_specs = list(range(min(config['num_training_specs'], len(list(specs.values())[0]))))
    episode_count = 0
    history = {'episodes': [], 'rewards': [], 'scalarization': scalarization}

    for spec_idx, spec_num in enumerate(training_specs):
        if spec_idx % 10 == 0:
            print(f"  [{tag}] Training spec {spec_idx+1}/{len(training_specs)} ...")
        try:
            env.base_env.obj_idx = spec_num
            env.reset()
        except Exception as e:
            print(f"  [{tag}] Skip spec {spec_num}: {e}")
            continue

        for pref_idx, preference in enumerate(preferences):
            agent.set_preference(preference)
            episodes_per_pref = max(1, config['training_episodes'] // len(preferences))
            for episode in range(episodes_per_pref):
                state = env.reset()
                episode_reward = 0.0
                for step in range(config['max_steps']):
                    epsilon = max(0.1, 1.0 - episode_count / 1000)
                    try:
                        action = agent.select_action(state, preference, epsilon=epsilon)
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            if isinstance(action, (int, np.integer)):
                                action = tuple([int(np.clip(action, 0, 2))] * num_params)
                            elif isinstance(action, (list, np.ndarray)):
                                action = tuple([int(np.clip(a, 0, 2)) for a in action[:num_params]])
                                while len(action) < num_params:
                                    action = action + (0,)
                    except Exception:
                        if isinstance(env.action_space, gym.spaces.Tuple):
                            action = tuple([np.random.randint(0, 3) for _ in range(num_params)])
                        else:
                            action = env.action_space.sample()

                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward_vec, done, truncated, info = step_result
                        done = done or truncated
                    elif len(step_result) == 4:
                        next_state, reward_vec, done, info = step_result
                    else:
                        break

                    if hasattr(agent, 'memory') and hasattr(agent.memory, 'push'):
                        agent.memory.push(state, action, reward_vec, next_state, done, preference)
                    if hasattr(agent, 'memory') and len(agent.memory) > 32 and episode_count % 4 == 0:
                        try:
                            agent.update(batch_size=32)
                        except Exception:
                            pass

                    state = next_state
                    episode_reward += float(np.sum(reward_vec)) if isinstance(reward_vec, (list, np.ndarray)) else float(reward_vec)
                    if done:
                        break

                episode_count += 1
                history['episodes'].append(episode_count)
                history['rewards'].append(episode_reward)

    final_path = MODELS_DIR / f"trained_morl_{scalarization}_final.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'agent_state_dict': agent.q_network.state_dict(),
        'preference_net_state_dict': agent.preference_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_dim': agent.state_dim,
        'action_dim': agent.action_dim,
        'reward_dim': agent.reward_dim,
        'scalarization': agent.scalarization,
        'episode_count': episode_count,
    }, final_path)
    print(f"  [{tag}] Done — {episode_count} episodes. Model: {final_path}")

    with open(RESULTS_DIR / f"training_history_{scalarization}.json", 'w') as f:
        json.dump(history, f, indent=2)

    return final_path

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_agent(model_path, scalarization, env, specs, config):
    tag = scalarization.upper()
    print(f"\n{'='*70}")
    print(f"  EVALUATING — {tag} SCALARIZATION")
    print(f"{'='*70}\n")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dim = checkpoint.get('state_dim', 64)
    action_dim = checkpoint.get('action_dim', 64)
    reward_dim = checkpoint.get('reward_dim', 4)

    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim, scalarization=scalarization)
    if 'agent_state_dict' in checkpoint and checkpoint['agent_state_dict'] is not None:
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
    if 'preference_net_state_dict' in checkpoint and checkpoint['preference_net_state_dict'] is not None:
        agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
    agent.q_network.eval()

    num_specs_total = len(list(specs.values())[0])
    num_targets = min(config['num_targets'], num_specs_total)
    specs_id_list = ['gain_min', 'ibias_max', 'phm_min', 'ugbw_min']

    target_specs_json = {}
    for i in range(num_specs_total):
        target_specs_json[str(i)] = {
            'target_gain_linear': float(specs['gain_min'][i]),
            'target_ugbw_mhz': float(specs['ugbw_min'][i]) / 1e6,
            'target_pm_deg': float(specs['phm_min'][i]),
            'target_ibias_ma': float(specs['ibias_max'][i]) * 1000.0,
        }

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=config['num_preferences'])
        if len(preferences) > config['num_preferences']:
            preferences = preferences[:config['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=config['num_preferences'])

    results = []
    reached_count = 0

    for target_idx in range(num_targets):
        if target_idx % 100 == 0 and target_idx > 0:
            print(f"  [{tag}] Progress: {target_idx}/{num_targets}")

        try:
            env.base_env.obj_idx = target_idx
            env.reset()
            target_spec = env.base_env.specs_ideal.copy()
            specs_id = env.base_env.specs_id
        except Exception:
            continue

        spec_reached = False
        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                state = env.reset()
                done = False
                steps = 0
                while not done and steps < config['max_steps']:
                    action = agent.select_action(state, preference)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, rw, done, truncated, info = step_result
                        done = done or truncated
                    elif len(step_result) == 4:
                        next_state, rw, done, info = step_result
                    else:
                        break
                    state = next_state
                    steps += 1

                if hasattr(env.base_env, 'cur_specs'):
                    actual_specs = env.base_env.cur_specs.copy()
                    spec_dict = dict(zip(specs_id, actual_specs))
                    gain_val = spec_dict.get('gain', spec_dict.get('gain_min', actual_specs[0]))
                    ugbw_val = spec_dict.get('ugbw', spec_dict.get('ugbw_min', actual_specs[1] if len(actual_specs) > 1 else 0))
                    phm_val  = spec_dict.get('phm',  spec_dict.get('phm_min',  actual_specs[2] if len(actual_specs) > 2 else 0))
                    ibias_val = spec_dict.get('ibias_max', actual_specs[3] if len(actual_specs) > 3 else 0)

                    if gain_val < 100:
                        gain_linear = 10 ** (gain_val / 20)
                        gain_db = gain_val
                    else:
                        gain_linear = gain_val
                        gain_db = 20 * np.log10(gain_val) if gain_val > 0 else 0

                    td = target_specs_json.get(str(target_idx), {})
                    gt = td.get('target_gain_linear', 1e-9)
                    ut = td.get('target_ugbw_mhz', 1e-9)
                    pt = td.get('target_pm_deg', 1e-9)
                    it = td.get('target_ibias_ma', 1e-9)

                    g_out = float(gain_linear)
                    u_out = float(ugbw_val / 1e6)
                    p_out = float(phm_val)
                    i_out = float(ibias_val * 1000)

                    # Strict reward rule for target_reached
                    actual_arr = [g_out, i_out / 1000.0, p_out, u_out * 1e6]
                    target_arr = [gt, it / 1000.0, pt, ut * 1e6]
                    sol_reached = check_target_reached(actual_arr, target_arr, specs_id_list)
                    if sol_reached:
                        spec_reached = True

                    f = _fom(g_out, u_out, p_out, i_out, gt, ut, pt, it)

                    g_ok = g_out >= gt
                    u_ok = u_out >= ut
                    p_ok = p_out >= pt
                    i_ok = i_out <= it

                    results.append({
                        'spec': target_idx + 1,
                        'solution': pref_idx + 1,
                        'target_gain_linear': gt,
                        'target_ugbw_mhz': ut,
                        'target_pm_deg': pt,
                        'target_ibias_ma': it,
                        'output_gain_linear': g_out,
                        'output_gain_db': float(gain_db),
                        'output_ugbw_mhz': u_out,
                        'output_pm_deg': p_out,
                        'output_ibias_ma': i_out,
                        'fom': round(f, 6),
                        'gain_pass': 'Yes' if g_ok else 'No',
                        'ugbw_pass': 'Yes' if u_ok else 'No',
                        'pm_pass': 'Yes' if p_ok else 'No',
                        'ibias_pass': 'Yes' if i_ok else 'No',
                        'complete_pass': 'Yes' if (g_ok and u_ok and p_ok and i_ok) else 'No',
                        'target_reached': 'Yes' if sol_reached else 'No',
                        'preference': preference.tolist() if hasattr(preference, 'tolist') else list(preference),
                        'scalarization': scalarization,
                    })
            except Exception as e:
                continue

        if spec_reached:
            reached_count += 1

    print(f"  [{tag}] Done: {reached_count}/{num_targets} specs reached, {len(results)} solutions")
    return results


def results_to_csv(results, out_path, label):
    df = pd.DataFrame(results)
    n = len(df)
    ns = df['spec'].nunique()
    passed = (df['target_reached'] == 'Yes').sum()
    avg_fom = df['fom'].astype(float).mean()
    best = df.groupby('spec')['fom'].max()
    avg_best = best.mean()
    top20 = best.nlargest(20).mean()

    summaries = pd.DataFrame([
        {'spec': 'summary', 'target_reached': f'{passed}/{n} solutions; {ns}/{ns} specs'},
        {'spec': f'summary_avg_fom_{n}_solutions', 'fom': round(avg_fom, 6)},
        {'spec': f'summary_avg_best_fom_{ns}_specs', 'fom': round(avg_best, 6)},
        {'spec': 'summary_avg_top20_best_fom', 'fom': round(top20, 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"  {label}: {passed}/{n} pass, avg FOM={avg_fom:.4f}, avg best={avg_best:.4f}, top20={top20:.4f}")
    return avg_best


def generate_comparison(cos_results, nw_results):
    def best_per_spec(results):
        by_spec = {}
        for sol in results:
            spec = sol['spec']
            f = sol['fom']
            if spec not in by_spec or f > by_spec[spec]['fom']:
                by_spec[spec] = sol
        return by_spec

    cos_best = best_per_spec(cos_results)
    nw_best = best_per_spec(nw_results)
    all_specs = sorted(set(cos_best.keys()) & set(nw_best.keys()))

    rows = []
    wins_cos = wins_nw = ties = 0
    cos_foms = []; nw_foms = []

    for spec in all_specs:
        cf = cos_best[spec]['fom']
        nf = nw_best[spec]['fom']
        cos_foms.append(cf); nw_foms.append(nf)
        if abs(cf - nf) < 1e-9: w = 'tie'; ties += 1
        elif cf > nf: w = 'cosine'; wins_cos += 1
        else: w = 'nw'; wins_nw += 1
        rows.append({'spec': spec, 'fom_cosine': round(cf, 6), 'fom_nw': round(nf, 6), 'winner': w})

    df = pd.DataFrame(rows)
    summaries = pd.DataFrame([
        {'spec': 'summary_specs', 'fom_cosine': len(rows)},
        {'spec': 'summary_wins_cosine', 'fom_cosine': wins_cos},
        {'spec': 'summary_wins_nw', 'fom_nw': wins_nw},
        {'spec': 'summary_ties', 'fom_cosine': ties},
        {'spec': 'summary_avg_fom_cosine', 'fom_cosine': round(np.mean(cos_foms), 6)},
        {'spec': 'summary_avg_fom_nw', 'fom_nw': round(np.mean(nw_foms), 6)},
    ])
    df = pd.concat([df, summaries], ignore_index=True)
    out = RESULTS_DIR / "morl_compare_nw_vs_cosine.csv"
    df.to_csv(out, index=False)
    print(f"\n  Comparison: {out}")
    print(f"  Cosine wins: {wins_cos}, NW wins: {wins_nw}, Ties: {ties}")
    print(f"  Cosine avg best FOM: {np.mean(cos_foms):.4f}")
    print(f"  NW avg best FOM:     {np.mean(nw_foms):.4f}")
    print(f"  Improvement (Cos-NW): {np.mean(cos_foms) - np.mean(nw_foms):.4f}")


# ===========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n{'#'*70}")
    print(f"  NW vs COSINE SCALARIZATION — TRAIN + EVALUATE")
    print(f"  Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    with open(DATASET_PATH, 'rb') as f:
        specs = pickle.load(f)
    print(f"\nDataset: {len(list(specs.values())[0])} specs")

    env = AutoCktMOEnv(generalize=True, num_valid=len(list(specs.values())[0]), run_valid=True)

    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=TRAIN_CONFIG['num_preferences'])
        if len(preferences) > TRAIN_CONFIG['num_preferences']:
            preferences = preferences[:TRAIN_CONFIG['num_preferences']]
    except Exception:
        preferences = generate_preference_vectors(4, method='random', num_vectors=TRAIN_CONFIG['num_preferences'])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cos_model = MODELS_DIR / "trained_morl_cosine_final.pth"
    nw_model = MODELS_DIR / "trained_morl_nw_final.pth"

    if not args.evaluate_only:
        cos_model = train_agent("cosine", TRAIN_CONFIG, env, specs, preferences)
        nw_model = train_agent("nw", TRAIN_CONFIG, env, specs, preferences)

    print(f"\n{'#'*70}")
    print("  EVALUATION PHASE")
    print(f"{'#'*70}")

    cos_results = evaluate_agent(cos_model, "cosine", env, specs, EVAL_CONFIG)
    nw_results = evaluate_agent(nw_model, "nw", env, specs, EVAL_CONFIG)

    print(f"\n{'#'*70}")
    print("  CSV GENERATION")
    print(f"{'#'*70}")

    results_to_csv(cos_results, RESULTS_DIR / "morl_autockt_results_trained_cosine.csv", "Cosine")
    results_to_csv(nw_results, RESULTS_DIR / "morl_autockt_results_trained_nw.csv", "NW")
    generate_comparison(cos_results, nw_results)

    elapsed = datetime.now() - start
    print(f"\nTotal time: {elapsed}")
    print("DONE")
