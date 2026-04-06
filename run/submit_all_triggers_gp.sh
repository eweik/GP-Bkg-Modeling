#!/bin/bash

# Default values for massive generation
TOTAL_TOYS=${1:-100000}
TOYS_PER_JOB=${2:-20000}
MIN_LEN="0.15"

# Calculate jobs per method
N_JOBS=$((TOTAL_TOYS / TOYS_PER_JOB))
TOTAL_SUBMISSIONS=$((7 * 4 * N_JOBS))

# Ensure local directories exist before submission
mkdir -p run/logs results

echo "================================================="
echo " MASSIVE GP BATCH SUBMISSION"
echo "================================================="
echo " Triggers:          t1 through t7"
echo " Total Toys/Method: $TOTAL_TOYS"
echo " Toys per Job:      $TOYS_PER_JOB"
echo " Total Jobs:        $TOTAL_SUBMISSIONS"
echo " GP Min Length:     $MIN_LEN"
echo "================================================="

# Loop over all 7 triggers
for TRIGGER in t1 t2 t3 t4 t5 t6 t7; do
    echo ">>> Queuing Trigger: $TRIGGER"
    
    # Loop over all 3 methods
    for METHOD in naive copula poisson_event decorrelated_bootstrap; do
    # for METHOD in naive copula; do
    # for METHOD in linear; do
        condor_submit run/submit_toys_gp.sub \
            trigger=$TRIGGER \
            method=$METHOD \
            toys_per_job=$TOYS_PER_JOB \
            min_len=$MIN_LEN \
            n_jobs=$N_JOBS > /dev/null
            
        echo "    - $METHOD submitted."
    done
done

echo "================================================="
echo "All $TOTAL_SUBMISSIONS GP jobs submitted! Type 'condor_q' to monitor."
