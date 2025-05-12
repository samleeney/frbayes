#!/usr/bin/env python3

import os
import glob
from pathlib import Path

def get_base_name(resume_file):
    """Extract the base name from a resume file path."""
    return Path(resume_file).stem

def has_results(chain_dir, base_name):
    """Check if there are corresponding result files for this base name."""
    results_dir = os.path.join(chain_dir, "results")
    if not os.path.exists(results_dir):
        return False
    
    # Look for PNG files in results that match the base name
    result_files = glob.glob(os.path.join(results_dir, f"{base_name}*.png"))
    return len(result_files) > 0

def main():
    # Find all chain directories
    chain_dirs = glob.glob("chains_*")
    files_to_delete = []

    # Process each chain directory
    for chain_dir in chain_dirs:
        # Find all resume files in this directory
        resume_files = glob.glob(os.path.join(chain_dir, "*.resume"))
        
        for resume_file in resume_files:
            base_name = get_base_name(resume_file)
            if has_results(chain_dir, base_name):
                files_to_delete.append(resume_file)

    # Print summary
    if not files_to_delete:
        print("No resume files found that can be safely deleted.")
        return

    print("The following resume files will be deleted:")
    total_size = 0
    for file in files_to_delete:
        size_bytes = os.path.getsize(file)
        size_gb = size_bytes / (1024**3)  # Convert to GB
        print(f"  {file} ({size_gb:.2f} GB)")
        total_size += size_bytes

    print(f"\nTotal space to be freed: {total_size / (1024**3):.2f} GB")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with deletion? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    # Perform deletion
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

if __name__ == "__main__":
    main() 