"""
Configuration file for LLM embeddings.
Add new models here to easily extend the corpus with additional embeddings.
"""

# Global configuration flags
FORCE_RECOMPUTE_EMBEDDINGS = False  # Set to True to force recomputing existing embeddings
FORCE_RERUN_EXPERIMENTS = False     # Set to True to force rerunning existing experiments

# Model configurations for embedding
# Add new models by appending to this list
MODEL_CONFIGS = [
    {
        'name': 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base',
        'display_name': 'deepseek_coder'
    },
    {
        'name': 'answerdotai/ModernBERT-base',
        'display_name': 'modernbert'
    },
    # Add more models here as needed
    {
        'name': 'microsoft/codebert-base',
        'display_name': 'codebert'
    },
    {
        'name': 'codellama/CodeLlama-34b-hf',
        'display_name': 'codellama'
    },
    {
        'name': 'codellama/CodeLlama-34b-Python-hf',
        'display_name': 'codellama_python'
    },
    {
        'name': 'nomic-ai/CodeRankEmbed',
        'display_name': 'coderankembed'
    },
]

def get_model_config(display_name):
    """Get model configuration by display name."""
    for config in MODEL_CONFIGS:
        if config['display_name'] == display_name:
            return config
    return None

def get_all_model_names():
    """Get list of all model display names."""
    return [config['display_name'] for config in MODEL_CONFIGS]
