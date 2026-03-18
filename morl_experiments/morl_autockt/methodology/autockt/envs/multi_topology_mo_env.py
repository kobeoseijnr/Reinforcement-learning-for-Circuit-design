"""
Multi-Topology Multi-Objective AutoCkt Environment for PD-MORL
Supports: two_stage, diff_pair, single_stage topologies
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import gym
from gym import spaces
import os
import sys

# Add parent directory to path to import autockt modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp


class MultiTopologyMOEnv(gym.Env):
    """
    Multi-objective wrapper for AutoCkt that supports multiple topologies.
    Supports: two_stage, diff_pair, single_stage
    Environments: ngspice, surrogate, bag
    """
    metadata = {'render.modes': ['human']}
    
    SUPPORTED_TOPOLOGIES = ['two_stage', 'diff_pair', 'single_stage']
    SUPPORTED_ENVIRONMENTS = ['ngspice', 'surrogate', 'bag']
    
    def __init__(self, topology='two_stage', env_type='ngspice', 
                 generalize=True, num_valid=50, run_valid=False):
        """
        Initialize the multi-topology multi-objective environment.
        
        Args:
            topology: Circuit topology ('two_stage', 'diff_pair', 'single_stage')
            env_type: Simulation environment ('ngspice', 'surrogate', 'bag')
            generalize: If True, use generated specs from gen_specs
            num_valid: Number of validation specs
            run_valid: If True, cycle through validation specs
        """
        self.topology = topology
        self.env_type = env_type
        
        if topology not in self.SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unsupported topology: {topology}. "
                           f"Supported: {self.SUPPORTED_TOPOLOGIES}")
        if env_type not in self.SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"Unsupported environment: {env_type}. "
                           f"Supported: {self.SUPPORTED_ENVIRONMENTS}")
        
        # Configure environment based on env_type
        if env_type == 'surrogate':
            os.environ['AUTOCKT_USE_SURROGATE'] = 'true'
        elif env_type == 'bag':
            # BAG will be used for post-parasitic verification later
            # For sizing, we use ngspice or surrogate
            os.environ['AUTOCKT_USE_SURROGATE'] = 'false'
        
        # Initialize the underlying TwoStageAmp environment
        # Note: Currently only two_stage is fully implemented
        # For diff_pair and single_stage, we use two_stage as a placeholder
        # In production, these would have their own implementations
        env_config = {
            "generalize": generalize,
            "valid": run_valid,
            "num_valid": num_valid,
            "multi_goal": False,
            "save_specs": False,
            "topology": topology,  # Pass topology info
            "env_type": env_type   # Pass env type info
        }
        
        self.base_env = TwoStageAmp(env_config)
        
        # Action space: discrete actions for each parameter
        self.action_space = self.base_env.action_space
        
        # Observation space: same as base environment
        self.observation_space = self.base_env.observation_space
        
        # Reward space: 4 objectives [gain, ugbw, phase_margin, -ibias]
        self.reward_space = [[0, 100], [0, 1e7], [0, 90], [-1e-3, 0]]
        
        # Track current state
        self.current_state = None
        self.terminal = False
        self.step_idx = 0
        self.max_episode_steps = 50
        
        # Map spec names to indices for reward extraction
        self.spec_names = ['ugbw', 'gain', 'phm', 'ibias_max']
        
        self.action_space_type = "Tuple"
        
        # Store last design parameters for BAG post-parasitic check
        self.last_design_params = None
        
    def reset(self):
        """Reset the environment and return initial observation."""
        obs = self.base_env.reset()
        self.current_state = obs
        self.terminal = False
        self.step_idx = 0
        self.last_design_params = None
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action tuple/list for discrete parameter adjustments
            
        Returns:
            observation: Current state observation
            reward: 4-dimensional reward vector [gain, ugbw, phase_margin, -ibias]
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional info dict with topology and env_type
        """
        # Convert action to tuple if needed
        num_params = len(self.base_env.params_id)
        
        if isinstance(action, np.ndarray):
            action = action.flatten().astype(int)
        elif isinstance(action, (list, tuple)):
            action = np.array([int(a) for a in action])
        elif isinstance(action, (int, np.integer)):
            action = np.array([int(action)] * num_params)
        else:
            raise ValueError(f"Invalid action format: {action} (type: {type(action)})")
        
        # Ensure action is the right length
        if len(action) != num_params:
            if len(action) < num_params:
                action = np.pad(action, (0, num_params - len(action)), mode='constant', constant_values=0)
            else:
                action = action[:num_params]
        
        # Clip actions to valid range [0, 2]
        action = np.clip(action, 0, 2)
        action = tuple(action.tolist())
        
        # Step the base environment
        step_result = self.base_env.step(action)
        if len(step_result) == 4:
            obs, scalar_reward, done, info = step_result
        elif len(step_result) == 5:
            obs, scalar_reward, done, truncated_base, info = step_result
            done = done or truncated_base
        else:
            raise ValueError(f"Unexpected step return: {len(step_result)} values")
        
        # Extract multi-objective rewards from current specs
        if hasattr(self.base_env, 'cur_specs') and hasattr(self.base_env, 'specs_id'):
            specs = self.base_env.cur_specs
            specs_id = self.base_env.specs_id
            
            spec_dict = dict(zip(specs_id, specs)) if len(specs) == len(specs_id) else {}
            
            gain_obj = spec_dict.get('gain', specs[0] if len(specs) > 0 else 0.0)
            ugbw_obj = spec_dict.get('ugbw', specs[1] if len(specs) > 1 else 0.0)
            phm_obj = spec_dict.get('phm', specs[2] if len(specs) > 2 else 0.0)
            ibias_val = spec_dict.get('ibias_max', specs[3] if len(specs) > 3 else 0.0)
            
            ibias_obj = -ibias_val
            
            reward = np.array([gain_obj, ugbw_obj, phm_obj, ibias_obj], dtype=np.float32)
            
            # Store design parameters if available
            if hasattr(self.base_env, 'params_id') and hasattr(self.base_env, 'param_vec'):
                try:
                    param_dict = {}
                    for param_id in self.base_env.params_id:
                        if param_id in self.base_env.param_vec:
                            # Get current parameter index
                            param_idx = getattr(self.base_env, f'cur_{param_id}', 0)
                            if param_idx < len(self.base_env.param_vec[param_id]):
                                param_dict[param_id] = self.base_env.param_vec[param_id][param_idx]
                    self.last_design_params = param_dict if param_dict else None
                except:
                    pass
        else:
            reward = np.array([scalar_reward, scalar_reward, scalar_reward, scalar_reward], dtype=np.float32)
        
        # Update state
        self.current_state = obs
        self.terminal = done
        self.step_idx += 1
        
        truncated = (self.step_idx >= self.max_episode_steps)
        done = done or truncated
        
        if not isinstance(info, dict):
            info = {}
        
        # Add topology and environment info
        info['topology'] = self.topology
        info['env_type'] = self.env_type
        if self.last_design_params:
            info['design_params'] = self.last_design_params.copy()
        
        return (
            obs.astype(np.float32),
            reward.astype(np.float32),
            done,
            truncated,
            info
        )
    
    def render(self, mode='human', close=False):
        """Render the environment (not implemented for circuit design)."""
        if hasattr(self.base_env, 'render'):
            return self.base_env.render(mode=mode, close=close)
        print('This environment does not have render option')
    
    def observe(self):
        """Return current observation."""
        return self.current_state if self.current_state is not None else self.reset()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        if hasattr(self.base_env, 'seed'):
            return self.base_env.seed(seed)
        return [seed]
    
    def get_design_params(self):
        """Get current design parameters for post-parasitic simulation."""
        return self.last_design_params

