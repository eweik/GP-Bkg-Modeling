#!/bin/bash
TRIGGER=$1
METHOD=$2
TOYS=$3
JOBID=$4
MIN_LEN=${5:-0.15}

echo "Starting GP job $JOBID on $(hostname)"

# Load ROOT/Python environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Move into Condor's temporary worker directory if it exists
if [ -n "$_CONDOR_SCRATCH_DIR" ]; then
    cd $_CONDOR_SCRATCH_DIR
fi

# Find where run_toys_gp.py actually ended up
if [ -f "python/run_toys_gp.py" ]; then
    PY_PATH="python/run_toys_gp.py"
elif [ -f "run_toys_gp.py" ]; then
    PY_PATH="run_toys_gp.py"
else
    echo "ERROR: run_toys_gp.py not found in root or python/ directory!"
    ls -R
    exit 1
fi

# Run the python script (Legacy MINUIT args removed, MIN_LEN added)
python3 $PY_PATH \
    --trigger "$TRIGGER" \
    --method "$METHOD" \
    --toys "$TOYS" \
    --min-len "$MIN_LEN" \
    --jobid "$JOBID"

echo "Job $JOBID finished."
