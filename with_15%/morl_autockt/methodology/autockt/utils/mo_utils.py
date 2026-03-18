"""
Preference Generation and Management Utilities for Multi-Objective AutoCkt
Based on PD-MORL framework adapted for circuit design optimization
"""

import numpy as np
import itertools
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

def generate_preference_vectors(num_objectives, num_vectors=None, method='uniform'):
    
    if method == 'uniform':
        if num_vectors is None:
            step_size = 0.1 if num_objectives <= 3 else 0.2
        else:
            step_size = 1.0 / (num_vectors ** (1/num_objectives))
            
        mesh_arrays = []
        for _ in range(num_objectives):
            mesh_arrays.append(np.arange(0, 1 + step_size, step_size))
        
        preference_vectors = np.array(list(itertools.product(*mesh_arrays)))
        
        valid_vectors = preference_vectors[np.abs(preference_vectors.sum(axis=1) - 1) < 1e-6]
        
        return np.unique(valid_vectors, axis=0)
        
    elif method == 'corners':
        vectors = np.eye(num_objectives)
        return vectors
        
    elif method == 'focused':
        vectors = []
        
        vectors.extend(np.eye(num_objectives))
        
        for i in range(num_objectives):
            for j in range(i+1, num_objectives):
                pref = np.zeros(num_objectives)
                pref[i] = 0.5
                pref[j] = 0.5
                vectors.append(pref)
        
        equal_pref = np.ones(num_objectives) / num_objectives
        vectors.append(equal_pref)
        
        return np.array(vectors)
        
    elif method == 'random':
        if num_vectors is None:
            num_vectors = 50
            
        alpha = np.ones(num_objectives)  # Uniform Dirichlet
        vectors = np.random.dirichlet(alpha, num_vectors)
        
        return vectors
        
    else:
        raise ValueError("Unknown method: {}".format(method))

def circuit_specific_preferences():
    
    preferences = {
        'high_speed': np.array([0.6, 0.2, 0.1, 0.1]),      # Prioritize bandwidth
        'high_gain': np.array([0.1, 0.6, 0.2, 0.1]),       # Prioritize gain
        'stable': np.array([0.1, 0.2, 0.6, 0.1]),          # Prioritize phase margin
        'low_power': np.array([0.1, 0.1, 0.1, 0.7]),       # Prioritize low bias current
        'balanced': np.array([0.25, 0.25, 0.25, 0.25]),    # Equal weights
        'performance': np.array([0.4, 0.4, 0.15, 0.05]),   # Speed + Gain focused
        'robust': np.array([0.2, 0.2, 0.5, 0.1]),          # Stability focused
        'efficient': np.array([0.3, 0.3, 0.1, 0.3]),       # Performance + Power
    }
    
    return preferences

class PreferenceScheduler:

    def __init__(self, num_objectives, schedule_type='curriculum'):
        self.num_objectives = num_objectives
        self.schedule_type = schedule_type
        self.training_step = 0
        self.performance_history = []
        
        self.base_preferences = generate_preference_vectors(num_objectives, method='focused')
        self.current_preference_idx = 0
        
    def get_next_preference(self, performance_metrics=None):
    
        if self.schedule_type == 'curriculum':
            return self._curriculum_schedule()
        elif self.schedule_type == 'adaptive':
            return self._adaptive_schedule(performance_metrics)
        elif self.schedule_type == 'random':
            return self._random_schedule()
        elif self.schedule_type == 'fixed':
            return self._fixed_schedule()
        else:
            raise ValueError("Unknown schedule type: {}".format(self.schedule_type))
    
    def _curriculum_schedule(self):
        total_preferences = len(self.base_preferences)
        
        if self.training_step < 1000:
            corner_prefs = self.base_preferences[:self.num_objectives]
            idx = self.training_step % len(corner_prefs)
            preference = corner_prefs[idx]
        
        elif self.training_step < 3000:
            pairwise_start = self.num_objectives
            pairwise_end = pairwise_start + (self.num_objectives * (self.num_objectives - 1) // 2)
            pairwise_prefs = self.base_preferences[pairwise_start:pairwise_end]
            if len(pairwise_prefs) > 0:
                idx = (self.training_step - 1000) % len(pairwise_prefs)
                preference = pairwise_prefs[idx]
            else:
                preference = self.base_preferences[-1]  # Equal weighting
        
        else:
            idx = self.training_step % total_preferences
            preference = self.base_preferences[idx]
        
        self.training_step += 1
        return preference
    
    def _adaptive_schedule(self, performance_metrics):
        if performance_metrics is None:
            return self._random_schedule()
        
        self.performance_history.append(performance_metrics)
        
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.array([p.get('objective_rewards', [0]*self.num_objectives) 
                                         for p in self.performance_history[-10:]])
            avg_performance = np.mean(recent_performance, axis=0)
            
            worst_obj = np.argmin(avg_performance)
            preference = np.zeros(self.num_objectives)
            preference[worst_obj] = 0.7
            remaining_weight = 0.3 / (self.num_objectives - 1)
            for i in range(self.num_objectives):
                if i != worst_obj:
                    preference[i] = remaining_weight
            
            return preference
        
        return self._random_schedule()
    
    def _random_schedule(self):
        idx = np.random.randint(len(self.base_preferences))
        return self.base_preferences[idx]
    
    def _fixed_schedule(self):
        preference = self.base_preferences[self.current_preference_idx]
        self.current_preference_idx = (self.current_preference_idx + 1) % len(self.base_preferences)
        return preference

class CircuitParetoAnalyzer:
    def __init__(self, objective_names=None):
        if objective_names is None:
            self.objective_names = ['UGBW', 'Gain', 'Phase Margin', 'Power Efficiency']
        else:
            self.objective_names = objective_names
        self.num_objectives = len(self.objective_names)
        
    def is_pareto_efficient(self, costs):
        costs = np.array(costs)
        if costs.ndim == 1:
            costs = costs.reshape(1, -1)
            
        costs_min = costs.copy()
        costs_min[:, :-1] = -costs_min[:, :-1]  
        
        pareto_efficient = np.ones(costs_min.shape[0], dtype=bool)
        for i, c in enumerate(costs_min):
            if pareto_efficient[i]:
                pareto_efficient[pareto_efficient] = np.any(costs_min[pareto_efficient] < c, axis=1) | \
                                                    np.all(costs_min[pareto_efficient] == c, axis=1)
                pareto_efficient[i] = True  #keep current point
                
        return pareto_efficient
    
    def calculate_hypervolume(self, points, reference_point=None):
        points = np.array(points)
        if reference_point is None:
            reference_point = np.min(points, axis=0) - 0.1
        
        if points.shape[1] == 2:
            sorted_indices = np.argsort(points[:, 0])
            sorted_points = points[sorted_indices]
            
            volume = 0
            prev_x = reference_point[0]
            
            for point in sorted_points:
                volume += (point[0] - prev_x) * (point[1] - reference_point[1])
                prev_x = point[0]
                
            return volume
        else:
            pareto_points = points[self.is_pareto_efficient(points)]
            volumes = []
            
            for point in pareto_points:
                vol = 1.0
                for i in range(len(point)):
                    vol *= max(0, point[i] - reference_point[i])
                volumes.append(vol)
                
            return np.sum(volumes)
    
    def plot_pareto_front(self, points, save_path=None, title="Circuit Design Pareto Front"):
        #plot the 2d projection of the pareto front
        points = np.array(points)
        pareto_efficient = self.is_pareto_efficient(points)
        pareto_points = points[pareto_efficient]
        
        if points.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            
            plt.scatter(points[:, 0], points[:, 1], alpha=0.5, c='blue', label='All Solutions')
            
            plt.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                       c='red', s=100, label='Pareto Front', marker='*')
            
            plt.xlabel(self.objective_names[0])
            plt.ylabel(self.objective_names[1])
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        return pareto_points
    
    def evaluate_solution_diversity(self, points):
        points = np.array(points)
        if len(points) < 2:
            return 0.0
        
        distances = euclidean_distances(points)
        np.fill_diagonal(distances, np.inf)
        
        min_distances = np.min(distances, axis=1)
        
        spacing = np.std(min_distances)
        
        return spacing

def interpolate_pareto_front(pareto_points, preference_vectors):
    """
    Interpolate Pareto front to predict objective values for given preferences
    Used for dynamic preference adjustment during training
    """
    pareto_points = np.array(pareto_points)
    preference_vectors = np.array(preference_vectors)
    
    if len(pareto_points) < 3:
        return None
    
    try:
        interpolator = RBFInterpolator(preference_vectors, pareto_points, kernel='linear')
        return interpolator
    except:
        return None

def suggest_preference_from_constraints(constraints, objective_names=None):
    if objective_names is None:
        objective_names = ['ugbw', 'gain', 'phm', 'ibias_max']
    
    num_objectives = len(objective_names)
    preference = np.ones(num_objectives) / num_objectives  #equal weights
    
    constraint_weights = {
        'bandwidth': ['ugbw'],
        'gain': ['gain'],  
        'stability': ['phm'],
        'power': ['ibias_max']
    }
    
    for constraint_type, obj_names in constraint_weights.items():
        if any(c in constraints for c in [f'min_{constraint_type}', f'max_{constraint_type}']):
            for obj_name in obj_names:
                if obj_name in objective_names:
                    idx = objective_names.index(obj_name)
                    preference[idx] *= 2.0
    
    preference = preference / np.sum(preference)
    
    return preference
