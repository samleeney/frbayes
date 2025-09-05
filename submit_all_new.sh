#!/bin/bash

# Parse command line arguments
RUN_MODE="sbatch"  # Default to sbatch mode
while [[ $# -gt 0 ]]; do
    case $1 in
        --terminal)
            RUN_MODE="terminal"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--terminal]"
            exit 1
            ;;
    esac
done

# Define environment variables
SBATCH_NTASKS=72   # Used in both modes
SBATCH_ARRAY="12"  # Used in both modes as it's needed for main.py
export SBATCH_ARRAY  # Export for direct terminal use

# SLURM-specific settings (only used in sbatch mode)
if [ "$RUN_MODE" = "sbatch" ]; then
    MAX_REQUEUES=1
    REPEAT_TIMES=1
else
    MAX_REQUEUES=0
    REPEAT_TIMES=1
fi

# Array of main files to run
main_files=(
    "main_exponential_default.py"
    "main_exponential_raw.py"
    "main_exponential_paper.py"
    "main_emg_default.py"
    "main_emg_raw.py"
    "main_emg_paper.py"
    "main_emg_with_baseline_default.py"
    "main_emg_with_baseline_raw.py"
    "main_emg_with_baseline_paper.py"
    "main_exponential_with_baseline_default.py"
    "main_exponential_with_baseline_raw.py"
    "main_exponential_with_baseline_paper.py"
)

# Time limits for each job (can be adjusted as needed)
time_limits=(
    "12:00:00" "12:00:00" "12:00:00"
    "12:00:00" "12:00:00" "12:00:00"
    "12:00:00" "12:00:00" "12:00:00"
    "12:00:00" "12:00:00" "12:00:00"
)

# Convert the SBATCH_ARRAY to an actual number of tasks
array_start=$(echo $SBATCH_ARRAY | cut -d'-' -f1)
array_end=$(echo $SBATCH_ARRAY | cut -d'-' -f2)
num_tasks=$((array_end - array_start + 1))

# Calculate total potential hours (only relevant for sbatch mode)
if [ "$RUN_MODE" = "sbatch" ]; then
    total_hours=0
    for time_limit in "${time_limits[@]}"; do
        hours=$(echo $time_limit | cut -d':' -f1)
        total_hours=$((total_hours + hours * SBATCH_NTASKS * num_tasks * MAX_REQUEUES * REPEAT_TIMES))
    done

    echo "Total potential hours: $total_hours"
    echo "Do you want to proceed with these settings? (yes/no)"
    read proceed

    if [[ "$proceed" != "yes" ]]; then
        echo "Exiting script."
        exit 1
    fi
fi

# Function to run the job
run_job() {
    local main_file=$1
    local time_limit=$2
    local repeat_number=$3
    local job_name=$(basename "$main_file" .py) # Extract job name from file name

    export JOB_NAME="${job_name}_${repeat_number}"
    export TIME_LIMIT=$time_limit
    export SBATCH_NTASKS

    if [ "$RUN_MODE" = "sbatch" ]; then
        # Submit via SLURM
        jobid=$(sbatch --job-name="$JOB_NAME" --time="$TIME_LIMIT" --ntasks="$SBATCH_NTASKS" --array="$SBATCH_ARRAY" submit.slurm | awk '{print $4}')
        echo "Submitted job $jobid for $main_file"
        sleep 3

        # Submit requeue attempts
        for ((i=2; i<=MAX_REQUEUES; i++)); do
            echo "Submitting requeue attempt $i for job $jobid"
            jobid=$(sbatch --dependency=afternotok:$jobid --job-name="$JOB_NAME" --time="$TIME_LIMIT" --ntasks="$SBATCH_NTASKS" --array="$SBATCH_ARRAY" submit.slurm | awk '{print $4}')
            echo "Requeue job ID: $jobid"
            sleep 1
        done
    else
        # Run directly in terminal
        echo "Running $main_file directly in terminal"
        mpirun -n $SBATCH_NTASKS python "$main_file"
    fi
}

# Main execution loop
for ((n=0; n<REPEAT_TIMES; n++)); do
    for index in "${!main_files[@]}"; do
        run_job "${main_files[$index]}" "${time_limits[$index]}" $((n + 1))
    done
done