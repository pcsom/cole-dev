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
        'display_name': 'deepseek_coder',
        'sentence_transformer': False
    },
    {
        'name': 'answerdotai/ModernBERT-base',
        'display_name': 'modernbert',
        'sentence_transformer': False
    },
    {
        'name': 'answerdotai/ModernBERT-large',
        'display_name': 'modernbert_large',
        'sentence_transformer': False
    },
    # Add more models here as needed
    {
        'name': 'microsoft/codebert-base',
        'display_name': 'codebert',
        'sentence_transformer': False
    },
    {
        'name': 'codellama/CodeLlama-34b-hf',
        'display_name': 'codellama',
        'sentence_transformer': False
    },
    {
        'name': 'codellama/CodeLlama-34b-Python-hf',
        'display_name': 'codellama_python',
        'sentence_transformer': False
    },
    {
        'name': 'codellama/CodeLlama-7b-Python-hf',
        'display_name': 'codellama_python_7b',
        'sentence_transformer': False
    },
    {
        'name': 'Qwen/Qwen2.5-Coder-0.5B',
        'display_name': 'qwen_coder_0_5b',
        'sentence_transformer': False
    },
    {
        'name': 'mistralai/Codestral-22B-v0.1',
        'display_name': 'codestral_22b',
        'sentence_transformer': False
    },
    {
        'name': 'nomic-ai/CodeRankEmbed',
        'display_name': 'coderankembed',
        'sentence_transformer': True
    },
    {
        'name': 'jinaai/jina-embeddings-v2-base-code',
        'display_name': 'jina_embed_code',
        'sentence_transformer': True
    },
    {
        'name': 'jinaai/jina-code-embeddings-1.5b',
        'display_name': 'jina_embed_code_1_5b',
        'sentence_transformer': True
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
