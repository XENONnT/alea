#!/bin/bash

. /cvmfs/xenon.opensciencegrid.org/releases/nT/development/setup.sh

export X509_USER_PROXY=/ospool/uc-shared/project/xenon/grid_proxy/xenon_service_proxy
export PATH=/opt/pegasus/current/bin:$PATH
export PYTHONPATH=`pegasus-config --python`:$PYTHONPATH
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"