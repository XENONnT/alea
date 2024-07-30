#!/bin/bash

set -e

# Extract the arguments
wf_id=$1

# Sanity check: these are the files in the current directory
ls -lh

# Make output filename
# This file will be used to store the output of the workflow
output_filename=$wf_id-combined_output.tar.gz

# Tar all the .h5 files into the output file
tar -czf $output_filename *.h5

# Check the output
echo "Checking the output"
ls -lh $output_filename

# Goodbye
echo "Done. Exiting."
exit 0
