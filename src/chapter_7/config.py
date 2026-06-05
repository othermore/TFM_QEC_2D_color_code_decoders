import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CH6_DIR = os.path.join(PROJECT_ROOT, 'data', 'chapter_6')
DATA_CH7_DIR = os.path.join(PROJECT_ROOT, 'data', 'chapter_7')
FIGURES_CH7_DIR = os.path.join(PROJECT_ROOT, 'figures', 'chapter_7')

DECODERS = [
    'bposd',
    'projection',
    'restriction',
    'chromobius',
    'concat_mwpm',
    'correlated'
]

DECODER_PREFIXES = {
    'bposd': 's2_bposd',
    'projection': 's3_projection',
    'restriction': 's4_restriction',
    'chromobius': 's5_mobius',
    'concat_mwpm': 's6_concatenated',
    'correlated': 's7_correlated'
}

DECODER_LABELS = {
    'bposd': 'BP+OSD',
    'projection': 'Projection',
    'restriction': 'Restriction',
    'chromobius': 'Möbius',
    'concat_mwpm': 'Concat',
    'correlated': 'Correlated'
}

COLORS = {
    'bposd': '#1f77b4',       # blue
    'projection': '#ff7f0e',  # orange
    'restriction': '#2ca02c', # green
    'chromobius': '#d62728',  # red
    'concat_mwpm': '#9467bd', # purple
    'correlated': '#8c564b'   # brown
}

# Distances for analysis 2 (Mitigation Capacity)
DISTANCES_MITIGATION = [9, 23]

# Multiplier to ensure we use sub-threshold data in regressions
SUB_THRESHOLD_RATIO = 0.7  # Only use data where p_phys < 0.7 * p_threshold

# Fixed targets for extrapolation (Analysis 5 and 6)
P_PHYS_TARGETS = [1e-3, 1e-4]
P_LOGICAL_TARGETS = [1e-6, 1e-12]

# Minimum r^2 threshold to trust regressions and show warnings
MIN_R2_THRESHOLD = 0.90
