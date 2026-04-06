#!/bin/bash

# Default values if no arguments are provided
TRIGGER=${1:-"t1"}
TOTAL_TOYS=${2:-100000}
TOYS_PER_JOB=${3:-1000}
MIN_LEN=${4:-0.15}

# Calculate how many jobs are needed per method
N_JOBS=$((TOTAL_TOYS / TOYS_PER_JOB))

# Ensure local directories exist before submission
mkdir -p run/logs results

echo "Directories verified. Commencing submission..."

echo "================================================="
echo " GP BATCH SUBMISSION (SINGLE TRIGGER)"
echo "================================================="
echo " Trigger:           ${TRIGGER^^}"
echo " Total Toys/Method: $TOTAL_TOYS"
echo " Toys per Job:      $TOYS_PER_JOB"
echo " Jobs per Method:   $N_JOBS"
echo " GP Min Length:     $MIN_LEN"
echo "================================================="

# Loop through all three methods and submit them
# for METHOD in naive linear copula; do
# for METHOD in naive ; do
for METHOD in naive copula poisson_bootstrap decorrelated_bootstrap; do
    echo "Submitting $METHOD..."
    condor_submit run/submit_toys_gp.sub \
        trigger=$TRIGGER \
        method=$METHOD \
        toys_per_job=$TOYS_PER_JOB \
        min_len=$MIN_LEN \
        n_jobs=$N_JOBS
done

echo "================================================="
echo "All jobs submitted! Type 'condor_q' to check status."
