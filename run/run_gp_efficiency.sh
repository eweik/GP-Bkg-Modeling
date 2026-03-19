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

ROOT_DIR="/afs/cern.ch/user/e/edweik/private/new_ad_files"
FITS_DIR="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits"
TRIGGER="all"
CHANNEL="all"
MIN_LEN="0.15"
TOYS="10"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--trigger) TRIGGER="$2"; shift ;;
        -c|--channel) CHANNEL="$2"; shift ;;
        -r|--root-dir) ROOT_DIR="$2"; shift ;;
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -m|--min-len) MIN_LEN="$2"; shift ;;
        --toys) TOYS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -t, --trigger   Trigger (e.g., t1) or 'all' (default: all)"
            echo "  -c, --channel   Channel (e.g., jj) or 'all' (default: all)"
            echo "  -m, --min-len   GP log-space length scale bound (default: 0.15)"
            echo "  --toys          Number of Poisson toys per mass point (default: 10)"
            exit 0
            ;;
    esac
    shift
done

echo "======================================================"
echo " Generating GP vs 5-Param Efficiency Plots (with Toys)"
echo "======================================================"
echo " Trigger:  ${TRIGGER^^}"
echo " Channel:  ${CHANNEL^^}"
echo " Min-Len:  $MIN_LEN"
echo " Toys:     $TOYS per mass point"
echo "------------------------------------------------------"

python3 python/plot_efficiency_comparison.py \
    --trigger "$TRIGGER" \
    --channel "$CHANNEL" \
    --root-dir "$ROOT_DIR" \
    --fits-dir "$FITS_DIR" \
    --min-len "$MIN_LEN" \
    --toys "$TOYS"
