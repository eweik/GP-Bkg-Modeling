#!/bin/bash
set -e

# 1. Setup Environment (Required for uproot/sklearn on lxplus)
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh"
if [ -f "$LCG_VIEW" ]; then
    source "$LCG_VIEW"
else
    echo "ERROR: Could not find LCG view. Are you on a node with CVMFS?"
    exit 1
fi

# 2. Directories
ROOT_DIR="/afs/cern.ch/user/e/edweik/private/new_ad_files"
FITS_DIR="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--root-dir) ROOT_DIR="$2"; shift ;;
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -r, --root-dir <path>   Path to ROOT files (default: $ROOT_DIR)"
            echo "  -f, --fits-dir <path>   Path to JSON fit files (default: $FITS_DIR)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================"
echo " Running Advanced Pull Diagnostics (KS Test & Q-Q Plots)"
echo "======================================================"
echo " ROOT: $ROOT_DIR"
echo " FITS: $FITS_DIR"
echo "------------------------------------------------------"

# Execute the python script
python3 python/plot_pull_diagnostics.py \
    --root-dir "$ROOT_DIR" \
    --fits-dir "$FITS_DIR"

echo "======================================================"
echo " Done."
echo "======================================================"

