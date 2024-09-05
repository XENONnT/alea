#!/bin/bash

set -e

# Extract the arguments
workflow_id=$1

# Sanity check: these are the files in the current directory
ls -lh

# Make input filename
# This file will be used to store the input of the workflow
input_filename=$workflow_id-combined_output.tar.gz

# Untar the output file into the .h5 files
tar -xzf $input_filename

# Check the output
echo "Checking the output"
ls -lh

# Goodbye
echo "Done. Exiting."
exit 0
