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
from stringify_utils import generate_dependency_classes, generate_network_class, generate_context_docstring

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

            tokenized_lengths = encoded['attention_mask'].sum(dim=1)
            # if any length exceeds max_length, print a warning
            if (tokenized_lengths > max_length).any():
                print(f"WARNING: Some sequences in batch exceed max_length={max_length} after tokenization.")
            
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

def get_echo_embeddings(texts, model, tokenizer, device, batch_size=8, max_length=2048):
    """
    Extract Echo Embeddings (Springer et al., 2024).
    Input: [x, x] (text repeated twice with \n\n separator)
    Pooling: Mean pooling over the SECOND occurrence only.
    
    This method is designed for causal decoder models (CodeLlama, Qwen, etc.)
    and may not be appropriate for encoder-only models (BERT, ModernBERT).
    
    Args:
        texts: List of text strings
        model: Transformer model
        tokenizer: Tokenizer
        device: Device to run on
        batch_size: Batch size for processing (smaller due to doubled sequence length)
        max_length: Maximum sequence length
    
    Returns:
        numpy array of embeddings [num_texts, embedding_dim]
    """
    all_embeddings = []
    
    # Check model type
    is_encoder = model.config.is_encoder_decoder if hasattr(model.config, 'is_encoder_decoder') else False
    if is_encoder or 'bert' in model.config.model_type.lower():
        print("WARNING: Echo embeddings are designed for causal decoder models (CodeLlama/Qwen).")
        print("         Using on encoder models (BERT/ModernBERT) is experimental and unproven.")
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Echo Embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Create echo strings: text + "\n\n" + text
            echo_texts = [t + "\n\n" + t for t in batch_texts]
            
            # Tokenize the full echo strings
            encoded = tokenizer(
                echo_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Create pooling mask that selects only the second repetition
            pooling_mask = torch.zeros_like(attention_mask)
            
            for b_idx, text in enumerate(batch_texts):
                # Tokenize just the first part to find the boundary
                first_part_tokens = tokenizer(text + "\n\n", add_special_tokens=False)['input_ids']
                first_part_len = len(first_part_tokens)
                
                # Get total valid length (excluding padding)
                total_len = attention_mask[b_idx].sum().item()
                
                # The echo part is from first_part_len to end
                start_idx = min(first_part_len, total_len - 1)
                end_idx = total_len
                
                # Set mask to 1 only for the second occurrence
                pooling_mask[b_idx, start_idx:end_idx] = 1
            
            # Prepare model inputs
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            if getattr(model.config, 'use_cache', False):
                model_inputs['use_cache'] = False
            
            # Forward pass
            outputs = model(**model_inputs)
            
            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                token_embeddings = outputs.last_hidden_state
            else:
                token_embeddings = outputs[0]
            
            # Weighted mean pooling using the POOLING MASK (not attention mask)
            input_mask_expanded = pooling_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embeddings = sum_embeddings / sum_mask
            
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

def embed_with_model(df, model_name, model_display_name, device='cuda', force=None, pytorch_only=False, 
                     use_sentence_transformers=False, use_quantization=True, use_echo_embeddings=False,
                     pytorch_context_mode=None, max_length=2048):
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
        use_quantization: If True, use 8-bit quantization for transformers models
        use_echo_embeddings: If True, use echo embeddings (repeat text twice, pool second half)
        pytorch_context_mode: None (default), "network" (add full Network code), or "comment" (add docstring).
                             Affects pytorch_code embedding by appending context information.
    
    Returns:
        DataFrame with added embedding columns
    """
    if force is None:
        force = FORCE_RECOMPUTE_EMBEDDINGS
    
    # Determine code types and expected columns
    code_types = ['pytorch_code', 'onnx_code', 'grammar_code']
    if pytorch_only:
        code_types = ['pytorch_code']
    
    # Build expected column names based on context mode
    expected_cols = []
    for ct in code_types:
        # Build suffix based on echo and quantization settings
        suffix_parts = []
        if use_echo_embeddings:
            suffix_parts.append('echo')
        if not use_quantization:
            suffix_parts.append('noquant')
        suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
        
        if ct == 'pytorch_code' and pytorch_context_mode == 'network':
            col_name = f'{model_display_name}_pytorch_code_with_network{suffix}_embedding'
        elif ct == 'pytorch_code' and pytorch_context_mode == 'comment':
            col_name = f'{model_display_name}_pytorch_code_with_comment{suffix}_embedding'
        else:
            col_name = f'{model_display_name}_{ct}{suffix}_embedding'
        expected_cols.append(col_name)
    
    # Check if embeddings already exist
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
        pooling_method = "echo embeddings" if use_echo_embeddings else "mean pooling"
        print(f"Using transformers library with custom {pooling_method}...")
        # Load model and tokenizer
        print(f"Loading {model_display_name} model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token if not present (common for decoder-only models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with or without quantization
        if use_quantization:
            print(f"Using 8-bit quantization for memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
                # load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto',
                trust_remote_code=True,
                use_safetensors=True
            )
            print(f"Model loaded with 8-bit quantization on {device}")
        else:
            print(f"Loading model without quantization (full precision)")
            model = AutoModel.from_pretrained(
                model_name,
                device_map='auto',
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.float16  # Use fp16 for memory efficiency
            )
            print(f"Model loaded in fp16 precision on {device}")
        
        print(f"Model hidden size: {model.config.hidden_size}")
    
    # Process each representation type
    for i, code_type in enumerate(code_types):
        print(f"\nProcessing {code_type}...")
        
        # Check if column exists in dataframe
        if code_type not in df.columns:
            print(f"WARNING: Column '{code_type}' not found in dataframe. Skipping.")
            continue
        
        texts = df[code_type].tolist()
        
        # Apply context mode for pytorch_code
        if code_type == 'pytorch_code' and pytorch_context_mode is not None:
            print(f"Applying pytorch_context_mode='{pytorch_context_mode}'...")
            modified_texts = []
            for text in texts:
                if pytorch_context_mode == 'network':
                    # Append full network code
                    dependency_code = generate_dependency_classes()
                    network_code = generate_network_class()
                    modified_text = f"{dependency_code}\n\n{text}\n\n{network_code}"
                elif pytorch_context_mode == 'comment':
                    # Prepend docstring description
                    docstring = generate_context_docstring()
                    modified_text = f"{docstring}\n\n{text}"
                else:
                    modified_text = text
                modified_texts.append(modified_text)
            texts = modified_texts
            print(f"Modified {len(texts)} texts with context")
        
        # Get embeddings using appropriate method
        if use_sentence_transformers:
            embeddings = get_embeddings_sentence_transformers(texts, model, batch_size=16)
        elif use_echo_embeddings:
            embeddings = get_echo_embeddings(texts, model, tokenizer, device)
        else:
            embeddings = get_embeddings(texts, model, tokenizer, device, max_length=max_length)
        
        # Add to dataframe as a column (store as list for each row)
        column_name = expected_cols[i]
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
    model_names=None,
    use_echo_embeddings=False,
    use_quantization=True,
    max_length=2048
):
    """
    Embed code representations in the corpus using multiple LLMs.
    
    Args:
        input_path: Path to input corpus
        output_path: Path to save embedded corpus
        device: Device to run models on
        use_half: If True, only process first half of corpus
        model_names: List of model display names to use. If None, uses all models from config.
        use_echo_embeddings: If True, use echo embeddings (repeat text twice, pool second half)
        use_quantization: If True, use 8-bit quantization for transformer models (ignored for sentence-transformers)
    
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
            use_sentence_transformers=model_config.get('sentence_transformer', False),
            use_quantization=use_quantization,
            use_echo_embeddings=use_echo_embeddings,
            max_length=max_length
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
    pytorch_only=False,
    use_echo_embeddings=False,
    use_quantization=True,
    pytorch_context_mode=None,
    max_length=2048
):
    """
    Add embeddings from a new model to an existing corpus.
    
    This function:
    1. Loads source corpus from corpus_path (for code text)
    2. Loads output corpus from output_path (if exists) to preserve existing embeddings
    3. Adds only the new embedding columns
    4. Saves to output_path
    
    Args:
        corpus_path: Path to source corpus (must have code columns like pytorch_code)
        model_name: Display name of model to add (must be in embedding_config.py)
        output_path: Path to save updated corpus. If None, overwrites corpus_path.
        device: Device to run model on
        force: If True, force recompute even if embeddings exist.
               If None, uses FORCE_RECOMPUTE_EMBEDDINGS from config.
        pytorch_only: If True, only embed pytorch_code
        use_echo_embeddings: If True, use echo embeddings (repeat text twice, pool second half)
        use_quantization: If True, use 8-bit quantization for transformer models (ignored for sentence-transformers)
        pytorch_context_mode: None (default), "network" (add full Network code), or "comment" (add docstring).
                             Affects pytorch_code embedding by appending context information.
    
    Returns:
        DataFrame with added embeddings
    """
    if force is None:
        force = FORCE_RECOMPUTE_EMBEDDINGS
    
    print(f"{'='*80}")
    print(f"Adding embeddings from {model_name}")
    print(f"{'='*80}\n")
    
    # Determine output path
    if output_path is None:
        output_path = corpus_path
    
    # Load source corpus for code text
    print(f"Loading source corpus (for code text) from {corpus_path}...")
    df_source = pd.read_pickle(corpus_path)
    print(f"  Loaded {len(df_source)} architectures")
    
    # Load or initialize output corpus
    if os.path.exists(output_path) and output_path != corpus_path:
        print(f"Loading existing output corpus from {output_path}...")
        df_output = pd.read_pickle(output_path)
        print(f"  Loaded {len(df_output)} architectures")
        print(f"  Existing columns: {list(df_output.columns)}")
        
        # Verify same architectures
        if len(df_source) != len(df_output):
            raise ValueError(f"Source and output corpus sizes don't match: {len(df_source)} vs {len(df_output)}")
    else:
        print(f"Output corpus doesn't exist or same as source, will create new")
        df_output = df_source.copy()
    
    # Build expected embedding column names based on context mode
    code_types = ['pytorch_code', 'onnx_code', 'grammar_code']
    if pytorch_only:
        code_types = ['pytorch_code']
    
    expected_embedding_cols = []
    for ct in code_types:
        # Build suffix based on echo and quantization settings
        suffix_parts = []
        if use_echo_embeddings:
            suffix_parts.append('echo')
        if not use_quantization:
            suffix_parts.append('noquant')
        suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
        
        if ct == 'pytorch_code' and pytorch_context_mode == 'network':
            col_name = f'{model_name}_pytorch_code_with_network{suffix}_embedding'
        elif ct == 'pytorch_code' and pytorch_context_mode == 'comment':
            col_name = f'{model_name}_pytorch_code_with_comment{suffix}_embedding'
        else:
            col_name = f'{model_name}_{ct}{suffix}_embedding'
        expected_embedding_cols.append(col_name)
    
    # Check if the SPECIFIC embeddings we want to create already exist in OUTPUT corpus
    existing_expected_cols = [col for col in expected_embedding_cols if col in df_output.columns]
    
    if existing_expected_cols and not force:
        print(f"\nINFO: Embeddings already exist in output: {existing_expected_cols}")
        print(f"Skipping embedding computation (set force=True or FORCE_RECOMPUTE_EMBEDDINGS=True to override)")
        return df_output
    
    if existing_expected_cols and force:
        print(f"\nWARNING: Overwriting existing embeddings: {existing_expected_cols}")
        df_output = df_output.drop(columns=existing_expected_cols)
        print(f"Dropped existing embedding columns from output")
    
    # Get model config
    model_config = get_model_config(model_name)
    if model_config is None:
        raise ValueError(f"Model '{model_name}' not found in embedding_config.py")
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"\nEmbedding with {model_name}...")
    
    # Add embeddings (use SOURCE corpus for code text)
    df_with_new_embeddings = embed_with_model(
        df_source,
        model_config['name'],
        model_config['display_name'],
        device=device,
        force=force,
        use_sentence_transformers=model_config.get('sentence_transformer', False),
        use_quantization=use_quantization,
        pytorch_only=pytorch_only,
        use_echo_embeddings=use_echo_embeddings,
        pytorch_context_mode=pytorch_context_mode,
        max_length=max_length
    )
    
    # The new columns should match the expected_embedding_cols we computed earlier
    # Verify they exist in the returned dataframe
    new_embedding_cols = [col for col in expected_embedding_cols if col in df_with_new_embeddings.columns]
    
    if not new_embedding_cols:
        print("\nWARNING: No new embedding columns found in returned dataframe!")
        print(f"Expected: {expected_embedding_cols}")
        print(f"Available: {[c for c in df_with_new_embeddings.columns if 'embedding' in c]}")
    
    print(f"\nNew embedding columns to add: {new_embedding_cols}")
    
    # Add new columns to output corpus
    for col in new_embedding_cols:
        df_output[col] = df_with_new_embeddings[col]
    
    # Save
    print(f"\n{'='*80}")
    print("Saving updated corpus...")
    print(f"{'='*80}\n")
    
    print(f"Saving to {output_path}")
    df_output.to_pickle(output_path)
    
    csv_path = output_path.replace('.pkl', '.csv')
    print(f"Saving CSV version to {csv_path}")
    df_output.to_csv(csv_path, index=False)
    
    print(f"\nUpdate complete!")
    print(f"Final DataFrame shape: {df_output.shape}")
    print(f"Final columns: {list(df_output.columns)}")
    
    return df_output

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
