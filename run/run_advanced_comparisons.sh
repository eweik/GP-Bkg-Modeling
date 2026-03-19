#!/bin/bash
set -e

# 1. Setup Environment
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh"
if [ -f "$LCG_VIEW" ]; then
    source "$LCG_VIEW"
else
    echo "ERROR: Could not find LCG view at $LCG_VIEW."
    exit 1
fi

# 2. Directories and Parameters
ROOT_DIR="/afs/cern.ch/user/e/edweik/private/new_ad_files"
FITS_DIR="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits"
MIN_LEN="0.12"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--root-dir) ROOT_DIR="$2"; shift ;;
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -m|--min-len) MIN_LEN="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -r, --root-dir <path>   Path to ROOT files"
            echo "  -f, --fits-dir <path>   Path to JSON fit files"
            echo "  -m, --min-len <float>   Log-space length scale bound (default: 0.12)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================================"
echo " Generating Advanced GP Spectral Comparisons"
echo "======================================================"
echo " ROOT: $ROOT_DIR"
echo " FITS: $FITS_DIR"
echo " Min Log Length Scale: $MIN_LEN"
echo "------------------------------------------------------"

# Run the python script
python3 python/compare_advanced_fits.py \
    --root-dir "$ROOT_DIR" \
    --fits-dir "$FITS_DIR" \
    --min-len "$MIN_LEN"

echo "======================================================"
echo " Done."
echo "======================================================"
