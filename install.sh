#!/bin/bash
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip --no-cache-dir

pip install -e .
