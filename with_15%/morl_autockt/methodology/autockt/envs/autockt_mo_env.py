"""
Multi-Objective AutoCkt Environment for PD-MORL
Wraps the TwoStageAmp environment to provide multi-objective rewards compatible with PD-MORL
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

class AutoCktMOEnv(gym.Env):
    """
    Multi-objective wrapper for AutoCkt TwoStageAmp environment.
    Returns 4-dimensional reward: [gain, ugbw, phase_margin, -ibias]
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, generalize=True, num_valid=50, run_valid=False):
        """
        Initialize the AutoCkt multi-objective environment.
        
        Args:
            generalize: If True, use generated specs from gen_specs
            num_valid: Number of validation specs
            run_valid: If True, cycle through validation specs
        """
        # Initialize the underlying TwoStageAmp environment
        env_config = {
            "generalize": generalize,
            "valid": run_valid,
            "num_valid": num_valid,
            "multi_goal": False,  # We'll handle multi-objective at this level
            "save_specs": False
        }
        
        self.base_env = TwoStageAmp(env_config)
        
        # Action space: discrete actions for each parameter
        # TwoStageAmp uses Tuple of Discrete spaces, we'll flatten it
        self.action_space = self.base_env.action_space
        
        # Observation space: same as base environment
        self.observation_space = self.base_env.observation_space
        
        # Reward space: 4 objectives [gain, ugbw, phase_margin, -ibias]
        # Note: ibias is negated because lower is better (minimize power)
        self.reward_space = [[0, 100], [0, 1e7], [0, 90], [-1e-3, 0]]  # Approximate ranges
        
        # Track current state
        self.current_state = None
        self.terminal = False
        self.step_idx = 0
        self.max_episode_steps = 50  # Default max steps for circuit design
        
        # Map spec names to indices for reward extraction
        # From TwoStageClass: ugbw, gain, phm, ibias
        self.spec_names = ['ugbw', 'gain', 'phm', 'ibias_max']
        
        self.action_space_type = "Tuple"  # For compatibility with PD-MORL
        
    def reset(self):
        """
        Reset the environment and return initial observation.
        """
        obs = self.base_env.reset()
        self.current_state = obs
        self.terminal = False
        self.step_idx = 0
        
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
            info: Additional info dict
        """
        # Convert action to tuple if needed
        # The base env expects actions as indices into action_meaning = [-1, 0, 2]
        # So valid action indices are 0, 1, 2 (for -1, 0, 2 respectively)
        num_params = len(self.base_env.params_id)
        
        if isinstance(action, np.ndarray):
            action = action.flatten().astype(int)
        elif isinstance(action, (list, tuple)):
            action = np.array([int(a) for a in action])
        elif isinstance(action, (int, np.integer)):
            # Single integer: apply to all parameters
            action = np.array([int(action)] * num_params)
        else:
            raise ValueError(f"Invalid action format: {action} (type: {type(action)})")
        
        # Ensure action is the right length
        if len(action) != num_params:
            # Pad or truncate to match number of parameters
            if len(action) < num_params:
                action = np.pad(action, (0, num_params - len(action)), mode='constant', constant_values=0)
            else:
                action = action[:num_params]
        
        # Clip actions to valid range [0, 2] for action_meaning indices
        action = np.clip(action, 0, 2)
        
        # Convert to tuple for base environment
        action = tuple(action.tolist())
        
        # Step the base environment (returns 4 values: obs, reward, done, info)
        step_result = self.base_env.step(action)
        if len(step_result) == 4:
            obs, scalar_reward, done, info = step_result
        elif len(step_result) == 5:
            obs, scalar_reward, done, truncated_base, info = step_result
            done = done or truncated_base
        else:
            raise ValueError(f"Unexpected step return: {len(step_result)} values")
        
        # Extract multi-objective rewards from current specs
        # The base env stores cur_specs as OrderedDict values sorted by key
        # spec_id order determines the mapping: ['gain', 'ibias_max', 'phm', 'ugbw'] (alphabetical)
        if hasattr(self.base_env, 'cur_specs') and hasattr(self.base_env, 'specs_id'):
            specs = self.base_env.cur_specs
            specs_id = self.base_env.specs_id
            
            # Map specs to objectives: [gain, ugbw, phm, -ibias]
            spec_dict = dict(zip(specs_id, specs)) if len(specs) == len(specs_id) else {}
            
            # Extract objectives in desired order
            gain_obj = spec_dict.get('gain', specs[0] if len(specs) > 0 else 0.0)
            ugbw_obj = spec_dict.get('ugbw', specs[1] if len(specs) > 1 else 0.0)
            phm_obj = spec_dict.get('phm', specs[2] if len(specs) > 2 else 0.0)
            ibias_val = spec_dict.get('ibias_max', specs[3] if len(specs) > 3 else 0.0)
            
            # Bias current: lower is better, so negate (in Amperes)
            ibias_obj = -ibias_val
            
            # Create multi-objective reward vector: [gain, ugbw, phm, -ibias]
            reward = np.array([gain_obj, ugbw_obj, phm_obj, ibias_obj], dtype=np.float32)
        else:
            # Fallback: use scalar reward converted to vector
            reward = np.array([scalar_reward, scalar_reward, scalar_reward, scalar_reward], dtype=np.float32)
        
        # Update state
        self.current_state = obs
        self.terminal = done
        self.step_idx += 1
        
        # Check for truncation (max steps)
        truncated = (self.step_idx >= self.max_episode_steps)
        done = done or truncated
        
        # Ensure info is a dict
        if not isinstance(info, dict):
            info = {}
        
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

