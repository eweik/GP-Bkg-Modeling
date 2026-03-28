#!/bin/bash
set -e

# Setup LCG Environment
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh"
if [ -f "$LCG_VIEW" ]; then
    source "$LCG_VIEW"
else
    echo "ERROR: Could not find LCG view."
    exit 1
fi

FITS_DIR="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits"
MIN_LEN="0.15"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -m|--min-len) MIN_LEN="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -f, --fits-dir <path>   Path to JSON fit files"
            echo "  -m, --min-len <float>   Log-space length scale bound (default: 0.12)"
            exit 0
            ;;
    esac
    shift
done

echo "======================================================"
echo " Running Advanced GP Signal Injection (Asimov)"
echo "======================================================"
echo " FITS: $FITS_DIR"
echo " Length Scale Bound: $MIN_LEN"
echo "------------------------------------------------------"

python3 python/run_signal_injection.py \
    --fits-dir "$FITS_DIR" \
    --min-len "$MIN_LEN"

echo "======================================================"
echo " Done."
echo "======================================================"
