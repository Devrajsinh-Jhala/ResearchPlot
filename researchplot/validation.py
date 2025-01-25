import inspect
import numpy as np

def validate_plot(func):
    """Decorator for plot validation"""
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        
        # Get parameter values
        params = bound_args.arguments
        
        # General validations
        if 'x' in params and 'y' in params:
            if len(params['x']) != len(params['y']):
                raise ValueError("x and y must have same length")
                
        if 'data' in params:
            if not isinstance(params['data'], (list, np.ndarray)):
                raise TypeError("data must be list or numpy array")
        
        # Specific plot validations
        if func.__name__ == 'heatmap':
            if params['data'].ndim != 2:
                raise ValueError("Heatmap requires 2D data")
                
        if func.__name__ == 'pie':
            if abs(sum(params['data']) - 1) > 0.01:
                raise ValueError("Pie chart data must sum to ~1")
        
        return func(*args, **kwargs)
    return wrapper