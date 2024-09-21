#!/bin/bash

set -e

# Check if number of arguments passed is correct
if [ $# -ne 21 ]; then
    echo "Error: You need to provide required number of arguments."
    exit 1
fi

# Extract the arguments
statistical_model=$1
poi=$2
hypotheses=$3
n_mc=$4
common_hypothesis=$5
generate_values=$6
nominal_values=$7
statistical_model_config=$8
parameter_definition=$9
statistical_model_args=${10}
likelihood_config=${11}
compute_confidence_interval=${12}
confidence_level=${13}
confidence_interval_kind=${14}
fit_strategy=${15}
toydata_mode=${16}
toydata_filename=${17}
only_toydata=${18}
output_filename=${19}
seed=${20}
metadata=${21}
echo "statistical_model: $statistical_model"
echo "poi: $poi"
echo "hypotheses: $hypotheses"
echo "n_mc: $n_mc"
echo "common_hypothesis: $common_hypothesis"
echo "generate_values: $generate_values"
echo "nominal_values: $nominal_values"
echo "statistical_model_config: $statistical_model_config"
echo "parameter_definition: $parameter_definition"
echo "statistical_model_args: $statistical_model_args"
echo "likelihood_config: $likelihood_config"
echo "compute_confidence_interval: $compute_confidence_interval"
echo "confidence_level: $confidence_level"
echo "confidence_interval_kind: $confidence_interval_kind"
echo "fit_strategy: $fit_strategy"
echo "toydata_mode: $toydata_mode"
echo "toydata_filename: $toydata_filename"
echo "only_toydata: $only_toydata"
echo "output_filename: $output_filename"
echo "seed: $seed"
echo "metadata: $metadata"

# Overwrite the template path
# Check if template_path exists in statistical_model_args
if echo $statistical_model_args | jq -e .template_path > /dev/null; then
    # Overwrite the template path
    echo "Original statistical_model_args: $statistical_model_args"
    statistical_model_args=$(echo $statistical_model_args | jq --arg pwd "$PWD/" --compact-output '.template_path = $pwd + .template_path')
    echo "Modified statistical_model_args: $statistical_model_args"
fi
# Check if limit_threshold exists in statistical_model_args
if echo $statistical_model_args | jq -e .limit_threshold > /dev/null; then
    # Overwrite the limit_threshold
    echo "Original statistical_model_args: $statistical_model_args"
    statistical_model_args=$(echo $statistical_model_args | jq --arg pwd "$PWD/" --compact-output '.limit_threshold = $pwd + .limit_threshold')
    echo "Modified statistical_model_args: $statistical_model_args"
fi

# Escaped strings
STATISTICAL_MODEL=$(echo "$statistical_model" | sed "s/'/\"/g")
POI=$(echo "$poi" | sed "s/'/\"/g")
HYPOTHESES=$(echo "$hypotheses" | sed "s/'/\"/g")
N_MC=$(echo "$n_mc" | sed "s/'/\"/g")
COMMON_HYPOTHESIS=$(echo "$common_hypothesis" | sed "s/'/\"/g")
GENERATE_VALUES=$(echo "$generate_values" | sed "s/'/\"/g")
NOMINAL_VALUES=$(echo "$nominal_values" | sed "s/'/\"/g")
STATISTICAL_MODEL_CONFIG=$(echo "$statistical_model_config" | sed "s/'/\"/g")
PARAMETER_DEFINITION=$(echo "$parameter_definition" | sed "s/'/\"/g")
STATISTICAL_MODEL_ARGS=$(echo "$statistical_model_args" | sed "s/'/\"/g")
LIKELIHOOD_CONFIG=$(echo "$likelihood_config" | sed "s/'/\"/g")
COMPUTE_CONFIDENCE_INTERVAL=$(echo "$compute_confidence_interval" | sed "s/'/\"/g")
CONFIDENCE_LEVEL=$(echo "$confidence_level" | sed "s/'/\"/g")
CONFIDENCE_INTERVAL_KIND=$(echo "$confidence_interval_kind" | sed "s/'/\"/g")
FIT_STRATEGY=$(echo "$fit_strategy" | sed "s/'/\"/g")
TOYDATA_MODE=$(echo "$toydata_mode" | sed "s/'/\"/g")
TOYDATA_FILENAME=$(echo "$toydata_filename" | sed "s/'/\"/g")
ONLY_TOYDATA=$(echo "$only_toydata" | sed "s/'/\"/g")
OUTPUT_FILENAME=$(echo "$output_filename" | sed "s/'/\"/g")
SEED=$(echo "$seed" | sed "s/'/\"/g")
METADATA=$(echo "$metadata" | sed "s/'/\"/g")

# Installing customized packages
. install.sh alea

# Extract tarballs input
mkdir -p templates
START=$(date +%s)
for TAR in `ls *.tar.gz`; do
    tar -xzf $TAR -C templates --strip-components=1
done
rm *.tar.gz
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Untarring took $DIFF seconds."

# Source the environment
. /opt/XENONnT/setup.sh
echo "Before running the toy MC, the work directory contains:"
ls -lh
echo "These are the contents of templates/:"
ls -lh templates/

# Print the command
echo "Running command: python alea_run_toymc.py \\
    --statistical_model $STATISTICAL_MODEL \\
    --poi $POI \\
    --hypotheses $HYPOTHESES \\
    --n_mc $N_MC \\
    --common_hypothesis $COMMON_HYPOTHESIS \\
    --generate_values $GENERATE_VALUES \\
    --nominal_values $NOMINAL_VALUES \\
    --statistical_model_config $STATISTICAL_MODEL_CONFIG \\
    --parameter_definition $PARAMETER_DEFINITION \\
    --statistical_model_args $STATISTICAL_MODEL_ARGS \\
    --likelihood_config $LIKELIHOOD_CONFIG \\
    --compute_confidence_interval $COMPUTE_CONFIDENCE_INTERVAL \\
    --confidence_level $CONFIDENCE_LEVEL \\
    --confidence_interval_kind $CONFIDENCE_INTERVAL_KIND \\
    --fit_strategy $FIT_STRATEGY \\
    --toydata_mode $TOYDATA_MODE \\
    --toydata_filename $TOYDATA_FILENAME \\
    --only_toydata $ONLY_TOYDATA \\
    --output_filename $OUTPUT_FILENAME \\
    --seed $SEED \\
    --metadata $METADATA"

# Run the toy MC
time python alea_run_toymc.py \
    --statistical_model $STATISTICAL_MODEL \
    --poi $POI \
    --hypotheses $HYPOTHESES \
    --n_mc $N_MC \
    --common_hypothesis $COMMON_HYPOTHESIS \
    --generate_values $GENERATE_VALUES \
    --nominal_values $NOMINAL_VALUES \
    --statistical_model_config $STATISTICAL_MODEL_CONFIG \
    --parameter_definition $PARAMETER_DEFINITION \
    --statistical_model_args $STATISTICAL_MODEL_ARGS \
    --likelihood_config $LIKELIHOOD_CONFIG \
    --compute_confidence_interval $COMPUTE_CONFIDENCE_INTERVAL \
    --confidence_level $CONFIDENCE_LEVEL \
    --confidence_interval_kind $CONFIDENCE_INTERVAL_KIND \
    --fit_strategy $FIT_STRATEGY \
    --toydata_mode $TOYDATA_MODE \
    --toydata_filename $TOYDATA_FILENAME \
    --only_toydata $ONLY_TOYDATA \
    --output_filename $OUTPUT_FILENAME \
    --seed $SEED \
    --metadata $METADATA

# Check the output
echo "Checking the files after processing:"
ls -lh

# Goodbye
echo "Done. Exiting."
exit 0
