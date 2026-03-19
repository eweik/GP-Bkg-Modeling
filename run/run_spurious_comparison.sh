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
TRIGGER="all"
CHANNEL="all"
METHOD="toys"
MIN_LEN="0.15"
TOYS="10"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--trigger) TRIGGER="$2"; shift ;;
        -c|--channel) CHANNEL="$2"; shift ;;
        -M|--method) METHOD="$2"; shift ;;
        -r|--root-dir) ROOT_DIR="$2"; shift ;;
        -f|--fits-dir) FITS_DIR="$2"; shift ;;
        -m|--min-len) MIN_LEN="$2"; shift ;;
        --toys) TOYS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            exit 0
            ;;
    esac
    shift
done

echo "======================================================"
echo " Generating Spurious Signal Plots"
echo "======================================================"
echo " Trigger:  ${TRIGGER^^}"
echo " Channel:  ${CHANNEL^^}"
echo " Method:   ${METHOD^^}"
if [ "$METHOD" == "toys" ]; then
    echo " Toys:     $TOYS per mass point"
fi
echo "------------------------------------------------------"

python3 python/plot_spurious_comparison.py \
    --trigger "$TRIGGER" \
    --channel "$CHANNEL" \
    --method "$METHOD" \
    --root-dir "$ROOT_DIR" \
    --fits-dir "$FITS_DIR" \
    --min-len "$MIN_LEN" \
    --toys "$TOYS"
