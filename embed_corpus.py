import torch
import pandas as pd
import numpy as np
import gc
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from generate_corpus import load_corpus
from embedding_config import MODEL_CONFIGS, get_model_config, FORCE_RECOMPUTE_EMBEDDINGS

# Try to import sentence-transformers (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to get sentence embeddings.
    
    Args:
        model_output: Model output containing hidden states
        attention_mask: Attention mask to ignore padding tokens
    
    Returns:
        Mean-pooled embeddings
    """
    token_embeddings = model_output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(texts, model, tokenizer, device, batch_size=16, max_length=2048):
    """
    Extract embeddings for a list of texts using mean pooling.
    
    Args:
        texts: List of text strings
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: torch device
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    
    Returns:
        numpy array of embeddings [num_texts, embedding_dim]
    """
    all_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Prepare basic inputs
            model_inputs = {
                'input_ids': input_ids, 
                'attention_mask': attention_mask
            }
            
            # Only pass use_cache=False if the model config has this attribute (e.g. DeepSeek).
            # ModernBERT (encoder) does not have this arg and will error if it is passed.
            if getattr(model.config, 'use_cache', False):
                model_inputs['use_cache'] = False
            
            # Get model output
            outputs = model(**model_inputs)
            
            # Mean pooling
            embeddings = mean_pooling(outputs, attention_mask)
            
            # Move to CPU and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_embeddings_sentence_transformers(texts, model, batch_size=16):
    """
    Extract embeddings using sentence-transformers library.
    
    Args:
        texts: List of text strings
        model: SentenceTransformer model
        batch_size: Batch size for processing
    
    Returns:
        numpy array of embeddings [num_texts, embedding_dim]
    """
    # sentence-transformers handles batching and device management internally
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

def embed_with_model(df, model_name, model_display_name, device='cuda', force=None, pytorch_only=False, use_sentence_transformers=False):
    """
    Add embeddings from a specific model to the dataframe.
    
    Args:
        df: DataFrame with code representations
        model_name: HuggingFace model name
        model_display_name: Display name for column naming
        device: Device to run model on
        force: If True, force recompute even if embeddings exist. 
               If None, uses FORCE_RECOMPUTE_EMBEDDINGS from config.
        pytorch_only: If True, only embed pytorch_code
        use_sentence_transformers: If True, use sentence-transformers library instead of transformers
    
    Returns:
        DataFrame with added embedding columns
    """
    if force is None:
        force = FORCE_RECOMPUTE_EMBEDDINGS
    
    # Check if embeddings already exist
    code_types = ['pytorch_code', 'onnx_code', 'grammar_code']
    expected_cols = [f'{model_display_name}_{ct}_embedding' for ct in code_types]
    existing_cols = [col for col in expected_cols if col in df.columns]
    
    if existing_cols and not force:
        print(f"\n{'='*80}")
        print(f"Embeddings for {model_display_name} already exist")
        print(f"{'='*80}")
        print(f"Found existing columns: {existing_cols}")
        print(f"Skipping embedding computation (set FORCE_RECOMPUTE_EMBEDDINGS=True to override)")
        print(f"{'='*80}\n")
        return df
    
    if existing_cols and force:
        print(f"\nWARNING: Overwriting existing embeddings for {model_display_name}")
        df = df.drop(columns=existing_cols)
    
    print(f"\n{'='*80}")
    print(f"Processing with {model_display_name}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")
    
    if use_sentence_transformers:
        print(f"Detected sentence-transformers model. Using SentenceTransformer API...")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                f"Model {model_name} requires sentence-transformers library but it's not installed. "
                f"Install with: pip install sentence-transformers"
            )
        
        # Load using sentence-transformers
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        tokenizer = None  # sentence-transformers handles tokenization internally
        print(f"Model loaded on {device}")
        print(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")
    else:
        print(f"Using transformers library with custom mean pooling...")
        # Load model and tokenizer
        print(f"Loading {model_display_name} model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token if not present (common for decoder-only models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Use 8-bit quantization for memory efficiency while maintaining good quality
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',
            trust_remote_code=True,
            use_safetensors=True  # Use safetensors format to avoid torch.load security issues
        )
        
        print(f"Model loaded with 8-bit quantization on {device}")
        print(f"Model hidden size: {model.config.hidden_size}")
    
    # Process each representation type
    code_types = ['pytorch_code', 'onnx_code', 'grammar_code']
    if pytorch_only:
        code_types = ['pytorch_code']
    for code_type in code_types:
        print(f"\nProcessing {code_type}...")
        texts = df[code_type].tolist()
        
        # Get embeddings using appropriate method
        if use_sentence_transformers:
            embeddings = get_embeddings_sentence_transformers(texts, model, batch_size=16)
        else:
            embeddings = get_embeddings(texts, model, tokenizer, device)
        
        # Add to dataframe as a column (store as list for each row)
        column_name = f'{model_display_name}_{code_type}_embedding'
        df[column_name] = embeddings.tolist()
        
        print(f"Added column: {column_name}")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Aggressively free memory after each representation to avoid OOM
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Thorough cleanup to free VRAM and disk space for next model
    print(f"\nCleaning up {model_display_name} from memory...")
    
    # Step 1: Move model to CPU and delete objects
    try:
        model.cpu()
        print("Moved model to CPU")
    except:
        pass  # Model might not support .cpu() if quantized
    
    del model
    if tokenizer is not None:  # tokenizer is None for sentence-transformers
        del tokenizer
    print("Deleted model and tokenizer objects")
    
    # Force garbage collection
    gc.collect()
    
    # Step 3: Clear CUDA cache thoroughly
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()  # Additional cleanup for inter-process memory
    
    print(f"GPU memory cleared")
    
    # Step 4: Delete model from HuggingFace cache to free disk space
    print(f"Deleting {model_display_name} from HuggingFace cache to free disk space...")
    try:
        # Get HuggingFace cache directory
        cache_dir = os.environ.get('HF_HOME', 
                    os.environ.get('TRANSFORMERS_CACHE', 
                    os.path.join(Path.home(), '.cache', 'huggingface')))
        
        # Model cache folder name format: models--org--model-name
        model_cache_name = model_name.replace('/', '--')
        model_cache_path = os.path.join(cache_dir, 'hub', f'models--{model_cache_name}')
        
        if os.path.exists(model_cache_path):
            shutil.rmtree(model_cache_path)
            print(f"Deleted cache directory: {model_cache_path}")
        else:
            print(f"Cache directory not found (may have been cleaned already): {model_cache_path}")
    except Exception as e:
        print(f"Warning: Could not delete model cache: {e}")
    
    print(f"Memory and disk cleaned. Ready for next model.\n")
    
    return df

def embed_corpus(
    input_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus.pkl',
    output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
    device='cuda',
    use_half=True,
    model_names=None
):
    """
    Embed code representations in the corpus using multiple LLMs.
    
    Args:
        input_path: Path to input corpus
        output_path: Path to save embedded corpus
        device: Device to run models on
        use_half: If True, only process first half of corpus
        model_names: List of model display names to use. If None, uses all models from config.
    
    Returns:
        DataFrame with embeddings
    """
    # Load corpus
    print("Loading corpus...")
    df = load_corpus(input_path)
    print(f"Loaded {len(df)} architectures")
    
    # Use only half if specified
    if use_half:
        half_size = len(df) // 2
        df = df.iloc[:half_size].copy()
        print(f"Using first half: {len(df)} architectures")
    
    print(f"Original columns: {list(df.columns)}\n")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Model configurations
    if model_names is None:
        # Use all models from config
        models = MODEL_CONFIGS
    else:
        # Use specified models only
        models = [get_model_config(name) for name in model_names]
        models = [m for m in models if m is not None]  # Filter out None
    
    if not models:
        raise ValueError("No valid models specified")
    
    print(f"\nWill process {len(models)} model(s): {[m['display_name'] for m in models]}\n")
    
    # Process with each model
    for model_config in models:
        df = embed_with_model(
            df,
            model_config['name'],
            model_config['display_name'],
            device=device,
            use_sentence_transformers=model_config.get('sentence_transformer', False)
        )
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving embedded corpus...")
    print(f"{'='*80}\n")
    
    # Save as pickle
    print(f"Saving to {output_path}")
    df.to_pickle(output_path)
    
    # Also save as CSV (embeddings will be stored as strings)
    csv_path = output_path.replace('.pkl', '.csv')
    print(f"Saving CSV version to {csv_path}")
    df.to_csv(csv_path, index=False)
    
    print(f"\nEmbedding complete!")
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
    # Print embedding column info
    print(f"\nEmbedding columns added:")
    embedding_cols = [col for col in df.columns if 'embedding' in col]
    for col in embedding_cols:
        sample_embedding = df[col].iloc[0]
        print(f"  {col}: dimension {len(sample_embedding)}")
    
    return df

def add_embeddings_to_corpus(
    corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
    model_name='codebert',
    output_path=None,
    device='cuda',
    force=None,
    pytorch_only=False
):
    """
    Add embeddings from a new model to an existing corpus.
    
    Args:
        corpus_path: Path to existing corpus (with or without embeddings)
        model_name: Display name of model to add (must be in embedding_config.py)
        output_path: Path to save updated corpus. If None, overwrites input.
        device: Device to run model on
        force: If True, force recompute even if embeddings exist.
               If None, uses FORCE_RECOMPUTE_EMBEDDINGS from config.
    
    Returns:
        DataFrame with added embeddings
    """
    if force is None:
        force = FORCE_RECOMPUTE_EMBEDDINGS
    
    print(f"{'='*80}")
    print(f"Adding embeddings from {model_name} to existing corpus")
    print(f"{'='*80}\n")
    
    # Load existing corpus
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_pickle(corpus_path)
    print(f"Loaded {len(df)} architectures")
    print(f"Existing columns: {list(df.columns)}\n")
    
    # Check if embeddings already exist
    embedding_cols = [col for col in df.columns if f'{model_name}_' in col and 'embedding' in col]
    if embedding_cols and not force:
        print(f"INFO: Embeddings for {model_name} already exist: {embedding_cols}")
        print(f"Skipping embedding computation (set force=True or FORCE_RECOMPUTE_EMBEDDINGS=True to override)")
        return df
    
    if embedding_cols and force:
        print(f"WARNING: Overwriting existing embeddings for {model_name}: {embedding_cols}")
        df = df.drop(columns=embedding_cols)
        print(f"Dropped existing embedding columns")
    
    # Get model config
    model_config = get_model_config(model_name)
    if model_config is None:
        raise ValueError(f"Model '{model_name}' not found in embedding_config.py")
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Add embeddings
    df = embed_with_model(
        df,
        model_config['name'],
        model_config['display_name'],
        device=device,
        force=force,
        use_sentence_transformers=model_config.get('sentence_transformer', False),
        pytorch_only=pytorch_only
    )
    
    # Save
    if output_path is None:
        output_path = corpus_path
    
    print(f"\n{'='*80}")
    print("Saving updated corpus...")
    print(f"{'='*80}\n")
    
    print(f"Saving to {output_path}")
    df.to_pickle(output_path)
    
    csv_path = output_path.replace('.pkl', '.csv')
    print(f"Saving CSV version to {csv_path}")
    df.to_csv(csv_path, index=False)
    
    print(f"\nUpdate complete!")
    print(f"Final DataFrame shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("No GPU available, will use CPU\n")
    
    # Run embedding
    df = embed_corpus()
    
    # Show example
    print("\n" + "="*80)
    print("Example embedded entry:")
    print("="*80)
    example = df.iloc[0]
    print(f"\nArchitecture Index: {example['arch_index']}")
    print(f"Architecture String: {example['arch_string']}")
    
    embedding_cols = [col for col in df.columns if 'embedding' in col]
    print(f"\nEmbeddings:")
    for col in embedding_cols[:3]:  # Show first 3
        emb = example[col]
        print(f"  {col}:")
        print(f"    Dimension: {len(emb)}")
        print(f"    First 5 values: {emb[:5]}")
