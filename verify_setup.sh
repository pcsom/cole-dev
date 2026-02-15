#!/bin/bash
# verify coder-nas setup

set -e

module load anaconda3/2023.03 2>/dev/null || true

ENV_NAME="codenas"

echo "checking conda environment..."
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: ${ENV_NAME} environment not found"
    echo "run: conda env create -f environment.yml"
    exit 1
fi
echo "found ${ENV_NAME}"

echo ""
echo "checking packages..."
conda run -n ${ENV_NAME} python -c "
import sys
import torch
import transformers
import nas_201_api
import umap
import xgboost
import sentence_transformers
import sklearn
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('xgboost:', xgboost.__version__)
print('umap:', umap.__version__)
print('sentence_transformers:', sentence_transformers.__version__)
print('sklearn:', sklearn.__version__)
"

echo ""
echo "checking project imports..."
conda run -n ${ENV_NAME} python -c "
from embed_corpus import add_embeddings_to_corpus
from robust_surrogate_predict import run_comparison
from stringify_utils import get_api
from embedding_config import MODEL_CONFIGS
print('models configured:', len(MODEL_CONFIGS))
"

echo ""
echo "checking corpus..."
# shared corpus
CORPUS="/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_corrected.pkl"
if [ -f "$CORPUS" ]; then
    echo "corpus found: $(du -h $CORPUS | cut -f1)"
else
    echo "WARNING: corpus not found at $CORPUS"
fi

echo ""
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "created logs/"
fi

echo ""
echo "============================================="
echo "✓ SETUP VERIFIED - READY TO RUN"
echo "============================================="
echo ""
echo "run: sbatch run_paper_llm.sh <model1> <model2>"
