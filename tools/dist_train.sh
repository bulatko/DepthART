#!/usr/bin/bash
CONFIG=${1:-launch/train/models/var_train}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
accelerate launch tools/train.py --config-name=$CONFIG