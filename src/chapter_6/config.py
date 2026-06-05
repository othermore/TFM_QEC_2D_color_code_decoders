import numpy as np

# Global Defaults
DEFAULT_CONFIG = {
    "distances": [5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
    "p_rates": np.linspace(0.02, 0.15, 12),
    "max_shots": 40000000,
    "max_errors": 400,
    
    # Zoom Simulation Defaults
    "zoom_max_shots": 60000000,
    "zoom_max_errors": 16000,
    "zoom_range_factor": (0.75, 1.25), # Multiplier around the calculated threshold (e.g., 0.75 * th to 1.25 * th)
    "zoom_points": 9                   # Number of points to simulate within the zoom range
}

# Decoder-specific overrides
DECODER_CONFIGS = {
    "bposd": {
        "p_rates": np.linspace(0.02, 0.20, 15),
    },
    "projection": {},
    "restriction": {},
    "chromobius": {}, # Möbius decoder
    "concat_mwpm": {}, # Concatenated MWPM decoder
    "correlated": {}  # Correlated matching decoder
}

def get_config(decoder_name: str, param: str):
    """
    Retrieves a configuration parameter, prioritizing decoder-specific overrides.
    
    Args:
        decoder_name (str): The name of the decoder (e.g., 'bposd', 'chromobius').
        param (str): The configuration parameter to retrieve.
        
    Returns:
        The value of the configuration parameter.
    """
    decoder_name = decoder_name.lower()
    
    if decoder_name in DECODER_CONFIGS and param in DECODER_CONFIGS[decoder_name]:
        return DECODER_CONFIGS[decoder_name][param]
        
    if param in DEFAULT_CONFIG:
        return DEFAULT_CONFIG[param]
        
    raise ValueError(f"Unknown configuration parameter: {param}")
