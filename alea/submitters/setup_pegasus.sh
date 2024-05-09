#!/bin/bash
# This script only works for XENONnT on ap23

. /cvmfs/xenon.opensciencegrid.org/releases/nT/development/setup.sh

cp /ospool/uc-shared/project/xenon/grid_proxy/xenon_service_proxy $HOME/.xenon_service_proxy
chmod 600 $HOME/.xenon_service_proxy

export X509_USER_PROXY=$HOME/.xenon_service_proxy
export PATH=/opt/pegasus/current/bin:$PATH
export PYTHONPATH=`pegasus-config --python`:$PYTHONPATH
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
