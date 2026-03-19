#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Default variables
TRIGGER=${1:-"t1"}      # First argument is the trigger, defaults to "t1"
TOYS=${2:-100000}       # Second argument is number of toys, defaults to 100000
MIN_LEN="0.15"          # Validated GP Length Scale Bound

echo "======================================================"
echo " Starting $TOYS GP Pseudo-Experiments for Trigger: ${TRIGGER^^}"
echo "======================================================"

# Loop through all 3 toy generation methods
for METHOD in naive linear copula; do
    echo ""
    echo ">>> Running method: $METHOD (Length Scale > 15%)"

    python3 python/run_toys_gp.py \
        --trigger "$TRIGGER" \
        --method "$METHOD" \
        --toys "$TOYS" \
        --min-len "$MIN_LEN"

    echo ">>> Completed $METHOD"
done

echo ""
echo "======================================================"
echo " All methods completed successfully for ${TRIGGER^^}!"
echo " GP Results are stored in the results/ directory."
echo "======================================================"
