"""
Multi-Objective Evaluation System for AutoCkt
Evaluates circuit designs across multiple conflicting objectives and computes Pareto metrics
"""

import numpy as np
import torch
import pickle
import os
from collections import defaultdict
from autockt.utils.mo_utils import (
    generate_preference_vectors, 
    CircuitParetoAnalyzer,
    circuit_specific_preferences
)

class CircuitMOEvaluator:
    
    def __init__(self, env, objective_names=None, evaluation_preferences=None):
        self.env = env
        if objective_names is None:
            self.objective_names = ['ugbw', 'gain', 'phm', 'ibias_max']
        else:
            self.objective_names = objective_names
        self.num_objectives = len(self.objective_names)
        
        if evaluation_preferences is None:
            uniform_prefs = generate_preference_vectors(self.num_objectives, method='uniform')
            circuit_prefs = list(circuit_specific_preferences().values())
            self.evaluation_preferences = np.vstack([uniform_prefs, circuit_prefs])
        else:
            self.evaluation_preferences = evaluation_preferences
        
        self.pareto_analyzer = CircuitParetoAnalyzer(objective_names)
        
    def evaluate_agent(self, agent, num_episodes=1, max_steps=50, verbose=True):

        results = {
            'objective_values': [],
            'preference_vectors': [],
            'episode_rewards': [],
            'design_parameters': [],
            'convergence_steps': [],
            'success_rate': 0.0
        }
        
        successful_designs = 0
        total_evaluations = 0
        
        for pref_idx, preference in enumerate(self.evaluation_preferences):
        if verbose:
            print("Evaluating preference {}/{}: {}".format(pref_idx+1, len(self.evaluation_preferences), preference))            # Set agent preference
            agent.set_preference(preference)
            
            for episode in range(num_episodes):
                total_evaluations += 1
                
                state = self.env.reset()
                total_reward = np.zeros(self.num_objectives)
                episode_rewards = []
                steps = 0
                done = False
                
                while not done and steps < max_steps:
                    action = agent.select_action(state, preference)
                    next_state, reward, done, info = self.env.step(action)
                    
                    if isinstance(reward, (list, np.ndarray)) and len(reward) > 1:
                        vector_reward = np.array(reward)
                    else:
                        if hasattr(self.env, 'cur_specs') and len(self.env.cur_specs) >= self.num_objectives:
                            vector_reward = self._convert_specs_to_objectives(self.env.cur_specs, self.env.specs_ideal)
                        else:
                            vector_reward = np.array([reward] * self.num_objectives)
                    
                    total_reward += vector_reward
                    episode_rewards.append(vector_reward)
                    state = next_state
                    steps += 1
                
                results['objective_values'].append(total_reward)
                results['preference_vectors'].append(preference)
                results['episode_rewards'].append(episode_rewards)
                results['convergence_steps'].append(steps)
                
                if hasattr(self.env, 'cur_params_idx'):
                    results['design_parameters'].append(self.env.cur_params_idx.copy())
                
                if done or self._evaluate_design_success(total_reward, preference):
                    successful_designs += 1
                    
        results['success_rate'] = successful_designs / total_evaluations if total_evaluations > 0 else 0.0
        
        return results
    
    def _convert_specs_to_objectives(self, current_specs, target_specs):
        objectives = []
        
        for i, (current, target) in enumerate(zip(current_specs, target_specs)):
            if target != 0:
                relative_performance = (current - target) / abs(target)
            else:
                relative_performance = current
                
            if i == len(current_specs) - 1:  # Assuming last spec is power/current
                objective = -relative_performance
            else:
                objective = relative_performance
                
            objectives.append(objective)
        
        return np.array(objectives)
    
    def _evaluate_design_success(self, objective_values, preference, threshold=0.8):
        scalarized_value = np.dot(objective_values, preference)
        
        return scalarized_value > threshold
    
    def compute_pareto_metrics(self, results):
       
        
        objective_values = np.array(results['objective_values'])
        
        if len(objective_values) == 0:
            return {'error': 'No evaluation results available'}
        
        pareto_efficient = self.pareto_analyzer.is_pareto_efficient(objective_values)
        pareto_front = objective_values[pareto_efficient]
        
        metrics = {
            'num_solutions': len(objective_values),
            'num_pareto_solutions': len(pareto_front),
            'pareto_ratio': len(pareto_front) / len(objective_values),
            'hypervolume': self.pareto_analyzer.calculate_hypervolume(pareto_front),
            'diversity': self.pareto_analyzer.evaluate_solution_diversity(pareto_front),
            'success_rate': results['success_rate'],
            'pareto_front': pareto_front,
            'all_solutions': objective_values
        }
        
        obj_stats = {}
        for i, obj_name in enumerate(self.objective_names):
            obj_values = objective_values[:, i]
            obj_stats[obj_name] = {
                'mean': np.mean(obj_values),
                'std': np.std(obj_values),
                'min': np.min(obj_values),
                'max': np.max(obj_values),
                'pareto_mean': np.mean(pareto_front[:, i]) if len(pareto_front) > 0 else 0
            }
        
        metrics['objective_statistics'] = obj_stats
        
        return metrics
    
    def compare_to_baseline(self, mo_results, baseline_results=None):
        
        mo_metrics = self.compute_pareto_metrics(mo_results)
        
        comparison = {
            'mo_metrics': mo_metrics,
            'improvement_summary': {}
        }
        
        if baseline_results is not None:
            baseline_objectives = np.array(baseline_results.get('objective_values', []))
            
            if len(baseline_objectives) > 0:
                mo_objectives = np.array(mo_results['objective_values'])
                
                for i, obj_name in enumerate(self.objective_names):
                    mo_values = mo_objectives[:, i]
                    baseline_value = np.mean(baseline_objectives) if baseline_objectives.ndim == 1 else np.mean(baseline_objectives[:, i])
                    
                    improvement = (np.mean(mo_values) - baseline_value) / abs(baseline_value) * 100
                    comparison['improvement_summary'][obj_name] = improvement
                
                comparison['average_improvement'] = np.mean(list(comparison['improvement_summary'].values()))
        
        return comparison
    
    def save_results(self, results, save_path, include_plots=True):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        metrics = self.compute_pareto_metrics(results)
        
        save_data = {
            'results': results,
            'metrics': metrics,
            'evaluation_config': {
                'objective_names': self.objective_names,
                'num_preferences': len(self.evaluation_preferences),
                'evaluation_preferences': self.evaluation_preferences
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print("Results saved to {}".format(save_path))
        
        if include_plots and len(results['objective_values']) > 0:
            plot_dir = os.path.dirname(save_path)
            plot_name = os.path.splitext(os.path.basename(save_path))[0]
            
            plot_path = os.path.join(plot_dir, "{}_pareto_front.png".format(plot_name))
            self.pareto_analyzer.plot_pareto_front(
                results['objective_values'], 
                save_path=plot_path,
                title="AutoCkt Multi-Objective Pareto Front"
            )
            
            print("Plots saved to {}".format(plot_dir))
        
        return metrics

class CircuitDesignBenchmark:
    
    def __init__(self):
        self.benchmark_specs = {
            'high_performance': {
                'ugbw': 1e7,  # 10 MHz
                'gain': 80,   # 80 dB
                'phm': 60,    # 60 degrees
                'ibias_max': 1e-4  # 100 µA
            },
            'low_power': {
                'ugbw': 1e6,  # 1 MHz
                'gain': 60,   # 60 dB  
                'phm': 45,    # 45 degrees
                'ibias_max': 1e-5  # 10 µA
            },
            'balanced': {
                'ugbw': 5e6,  # 5 MHz
                'gain': 70,   # 70 dB
                'phm': 50,    # 50 degrees
                'ibias_max': 5e-5  # 50 µA
            }
        }
    
    def evaluate_on_benchmarks(self, agent, env, evaluator):
        benchmark_results = {}
        
        for benchmark_name, specs in self.benchmark_specs.items():
            
            if hasattr(env, 'specs_ideal'):
                original_specs = env.specs_ideal.copy()
                env.specs_ideal = np.array([specs[obj] for obj in evaluator.objective_names])
            
            results = evaluator.evaluate_agent(agent, num_episodes=5, verbose=False)
            metrics = evaluator.compute_pareto_metrics(results)
            
            benchmark_results[benchmark_name] = {
                'target_specs': specs,
                'results': results,
                'metrics': metrics
            }
            
            if hasattr(env, 'specs_ideal'):
                env.specs_ideal = original_specs
            
            print("  Success rate: {:.1%}".format(metrics['success_rate']))
            print("  Pareto solutions: {}/{}".format(metrics['num_pareto_solutions'], metrics['num_solutions']))
            print("  Hypervolume: {:.4f}".format(metrics['hypervolume']))
        
        return benchmark_results
    
    def generate_report(self, benchmark_results, save_path=None):
        
        report = {
            'summary': {},
            'detailed_results': benchmark_results,
            'recommendations': []
        }
        
        total_success_rate = np.mean([r['metrics']['success_rate'] for r in benchmark_results.values()])
        avg_pareto_ratio = np.mean([r['metrics']['pareto_ratio'] for r in benchmark_results.values()])
        avg_hypervolume = np.mean([r['metrics']['hypervolume'] for r in benchmark_results.values()])
        
        report['summary'] = {
            'overall_success_rate': total_success_rate,
            'average_pareto_ratio': avg_pareto_ratio,
            'average_hypervolume': avg_hypervolume,
            'num_benchmarks': len(benchmark_results)
        }
        
        if total_success_rate < 0.5:
            report['recommendations'].append("Consider adjusting reward function or training parameters")
        
        if avg_pareto_ratio < 0.1:
            report['recommendations'].append("increase diversity in preference vectors")
        
        if avg_hypervolume < 0.01:
            report['recommendations'].append("improve solution quality")
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(report, f)
        
        return report
