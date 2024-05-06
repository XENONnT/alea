#!/bin/bash

set -e

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
toydata_mode=${15}
toydata_filename=${16}
only_toydata=${17}
output_filename=${18}
seed=${19}
metadata=${20}
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
echo "toydata_mode: $toydata_mode"
echo "toydata_filename: $toydata_filename"
echo "only_toydata: $only_toydata"
echo "output_filename: $output_filename"
echo "seed: $seed"
echo "metadata: $metadata"

# Extract tarballs input
START=$(date +%s)
for TAR in `ls *.tar.gz`; do
    tar xzf $TAR
done
rm *.tar.gz
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Untarring took $DIFF seconds."

# Move all untarred files to templates/
mkdir -p templates
mv *.h5 templates/

# Source the environment
. /opt/XENONnT/setup.sh
chmod +x alea-run_toymc
echo "Before running the toy MC, the work directory contains:"
ls -lh
echo "These are the contents of templates/:"
ls -lh templates/

# Run the toy MC
time python3 ./alea-run_toymc \
    --statistical_model $statistical_model \
    --poi $poi \
    --hypotheses $hypotheses \
    --n_mc $n_mc \
    --common_hypothesis $common_hypothesis \
    --generate_values $generate_values \
    --nominal_values $nominal_values \
    --statistical_model_config $statistical_model_config \
    --parameter_definition $parameter_definition \
    --statistical_model_args $statistical_model_args \
    --likelihood_config $likelihood_config \
    --compute_confidence_interval $compute_confidence_interval \
    --confidence_level $confidence_level \
    --confidence_interval_kind $confidence_interval_kind \
    --toydata_mode $toydata_mode \
    --toydata_filename $toydata_filename \
    --only_toydata $only_toydata \
    --output_filename $output_filename \
    --seed $seed \
    --metadata $metadata

# Check the output
echo "Checking the output"
ls -lh $output_filename
ls -lh $toydata_filename

# Goodbye
echo "Done. Exiting."
exit 0
