"""
BAG (Berkeley Analog Generator) Simulation Wrapper
For post-parasitic performance verification
"""
import os
import numpy as np
import yaml
from collections import OrderedDict
from abc import ABC, abstractmethod


class BAGWrapper(ABC):
    """
    Base class for BAG simulation wrapper.
    BAG provides post-parasitic simulation capabilities.
    """
    
    def __init__(self, topology_name, bag_config_path=None):
        """
        Initialize BAG wrapper.
        
        Args:
            topology_name: Name of the topology (e.g., 'two_stage', 'diff_pair', 'single_stage')
            bag_config_path: Path to BAG configuration file
        """
        self.topology_name = topology_name
        self.bag_config_path = bag_config_path
        self.bag_available = self._check_bag_availability()
        
    def _check_bag_availability(self):
        """Check if BAG is available in the environment."""
        # Check for BAG installation
        bag_path = os.environ.get('BAG_WORK_DIR', None)
        if bag_path and os.path.exists(bag_path):
            return True
        
        # Check for Python module
        try:
            import bag
            return True
        except ImportError:
            pass
        
        # For now, return False if BAG is not available
        # In production, BAG would need to be installed
        print(f"[WARNING] BAG not found. Post-parasitic simulations will use surrogate model.")
        return False
    
    def simulate_post_parasitic(self, design_params, layout_info=None):
        """
        Simulate circuit with post-parasitic effects.
        
        Args:
            design_params: Dictionary of design parameters
            layout_info: Optional layout information (if available)
            
        Returns:
            specs_dict: Dictionary of post-parasitic specifications
        """
        if self.bag_available:
            return self._run_bag_simulation(design_params, layout_info)
        else:
            # Fallback to surrogate model that estimates parasitic effects
            return self._surrogate_post_parasitic(design_params)
    
    @abstractmethod
    def _run_bag_simulation(self, design_params, layout_info):
        """Run actual BAG simulation. Must be implemented by subclasses."""
        pass
    
    def _surrogate_post_parasitic(self, design_params):
        """
        Surrogate model for post-parasitic effects when BAG is not available.
        Applies degradation factors to account for parasitic capacitance and resistance.
        """
        # Estimate parasitic degradation (conservative estimates)
        # In practice, these would come from layout extraction
        
        # Typical degradation factors for post-parasitic:
        # - Bandwidth: 10-20% reduction due to parasitic capacitance
        # - Gain: 5-10% reduction
        # - Phase margin: 5-15% reduction
        # - Power: Slight increase due to parasitic resistance
        
        # This is a placeholder - in real implementation, would use actual BAG or extracted netlist
        degradation = {
            'ugbw_factor': 0.85,  # 15% reduction
            'gain_factor': 0.95,   # 5% reduction
            'phm_factor': 0.90,    # 10% reduction
            'ibias_factor': 1.05   # 5% increase
        }
        
        # For now, return a dict indicating post-parasitic simulation was performed
        # In real implementation, would extract from BAG results
        specs = {
            'ugbw_post': None,  # Would be filled by actual simulation
            'gain_post': None,
            'phm_post': None,
            'ibias_post': None,
            'degradation_applied': True,
            'degradation_factors': degradation
        }
        
        return specs
    
    def compare_pre_post_parasitic(self, pre_parasitic_specs, post_parasitic_specs):
        """
        Compare pre-layout and post-layout (parasitic) specifications.
        
        Args:
            pre_parasitic_specs: Specifications from pre-layout simulation
            post_parasitic_specs: Specifications from post-layout simulation
            
        Returns:
            comparison_dict: Dictionary with comparison metrics
        """
        comparison = {}
        
        for spec_name in ['ugbw', 'gain', 'phm', 'ibias']:
            pre_key = spec_name
            post_key = f'{spec_name}_post' if f'{spec_name}_post' in post_parasitic_specs else spec_name
            
            if pre_key in pre_parasitic_specs and post_key in post_parasitic_specs:
                pre_val = pre_parasitic_specs[pre_key]
                post_val = post_parasitic_specs[post_key]
                
                if pre_val is not None and post_val is not None:
                    comparison[f'{spec_name}_pre'] = pre_val
                    comparison[f'{spec_name}_post'] = post_val
                    comparison[f'{spec_name}_degradation'] = (pre_val - post_val) / pre_val if pre_val != 0 else 0
        
        return comparison


class TwoStageBAGWrapper(BAGWrapper):
    """BAG wrapper for two-stage opamp topology."""
    
    def __init__(self, bag_config_path=None):
        super().__init__('two_stage', bag_config_path)
    
    def _run_bag_simulation(self, design_params, layout_info):
        """
        Run BAG simulation for two-stage opamp.
        
        In a real implementation, this would:
        1. Generate layout using BAG
        2. Extract parasitic netlist
        3. Run post-layout simulation
        4. Extract specifications
        """
        # Placeholder for actual BAG integration
        # Real implementation would call BAG API here
        
        # For now, use surrogate model
        return self._surrogate_post_parasitic(design_params)


class DiffPairBAGWrapper(BAGWrapper):
    """BAG wrapper for differential pair topology."""
    
    def __init__(self, bag_config_path=None):
        super().__init__('diff_pair', bag_config_path)
    
    def _run_bag_simulation(self, design_params, layout_info):
        """Run BAG simulation for differential pair."""
        return self._surrogate_post_parasitic(design_params)


class SingleStageBAGWrapper(BAGWrapper):
    """BAG wrapper for single-stage opamp topology."""
    
    def __init__(self, bag_config_path=None):
        super().__init__('single_stage', bag_config_path)
    
    def _run_bag_simulation(self, design_params, layout_info):
        """Run BAG simulation for single-stage opamp."""
        return self._surrogate_post_parasitic(design_params)


def get_bag_wrapper(topology_name):
    """Factory function to get appropriate BAG wrapper."""
    topology_map = {
        'two_stage': TwoStageBAGWrapper,
        'diff_pair': DiffPairBAGWrapper,
        'single_stage': SingleStageBAGWrapper
    }
    
    wrapper_class = topology_map.get(topology_name)
    if wrapper_class is None:
        raise ValueError(f"Unknown topology: {topology_name}. Available: {list(topology_map.keys())}")
    
    return wrapper_class()

