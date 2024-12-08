#!/bin/bash

set -e

# Extract the arguments
workflow_id=$1
outputfolder=$2

# Sanity check: these are the files in the current directory
ls -lh

# Make input filename
# This file will be used to store the input of the workflow
input_filename=$workflow_id-combined_output.tar.gz

# Make a temporary directory for decompressed files
mkdir decompressed

# Untar the output file into the .h5 files
tar -xzf $input_filename -C decompressed

# Check the output
echo "Checking the output"
ls -lh

# Move the outputs
mv decompressed/* $outputfolder/

# Goodbye
echo "Done. Exiting."
exit 0
