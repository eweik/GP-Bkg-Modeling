#!/bin/bash
set -e

LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh"
if [ -f "$LCG_VIEW" ]; then
    source "$LCG_VIEW"
else
    echo "ERROR: Could not find LCG view."
    exit 1
fi

ROOT_DIR="/afs/cern.ch/user/e/edweik/private/new_ad_files"
FITS_DIR="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits"
MIN_LEN="0.15" # 15% mass resolution constraint

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--root-dir) ROOT_DIR="$2"; shift ;;
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -m|--min-len) MIN_LEN="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            exit 0
            ;;
    esac
    shift
done

echo "======================================================"
echo " Running Advanced GP Diagnostics (Log-X & Mean Prior)"
echo "======================================================"

python3 python/plot_advanced_gp_diagnostics.py \
    --root-dir "$ROOT_DIR" \
    --fits-dir "$FITS_DIR" \
    --min-len "$MIN_LEN"

echo "======================================================"
echo " Done."
echo "======================================================"
