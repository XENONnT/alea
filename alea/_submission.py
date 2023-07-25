import logging
logging.basicConfig(level=logging.INFO, force=True)
import subprocess
import time
from alea.toymc_running import toymc_to_sbatch_call_array, compute_neyman_thresholds, toymc_to_sbatch_call_array_update, compute_neyman_thresholds_update
from alea import toymc_running
import argparse
import os
import sys
from pydoc import locate
import re
import glob
import copy
import pkg_resources
from tqdm import tqdm
import json
import itertools
import inspect
from alea.utils import read_config
import alea

def create_alea_tarball(targetname, targetpath):
    alea_data_path = pkg_resources.resource_filename(
        "alea", "data")
    alea_root_path = os.path.join(alea_data_path, "../..")

    if os.path.exists(targetname):
        os.remove(targetname)

    #  cmd = "tar cvfz {target} {input} --exclude=*.tar.gz --exclude=*/.git/* --exclude=*/sbatch_submission/*".format(
        #  target=targetname, input=alea_root_path)
    cmd = "tar cvfz {target} {input} --exclude=*.tar.gz --exclude=*.sbatch".format(
        target=targetname, input=alea_root_path)
    os.system(cmd)

    #  full_targetpath = os.path.join(targetpath, targetname)
    #  if os.path.exists(full_targetpath):
    #      os.remove(full_targetpath)

    #  cmd = "mv {targetname} {full_targetpath}".format(
    #      targetname=targetname, full_targetpath=full_targetpath)
    #  os.system(cmd)


def OSG_template(
        joblist,
        inputfiles,
        singularity_container="/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest",
        request_memory="1000Mb"):
    submission_template = """#!/bin/bash
executable = $(job)
universe = vanilla
Error  = $(job).err
Output = $(job).out
Log    = $(job).log

#  Requirements = (HAS_CVMFS_xenon_opensciencegrid_org) && (OSGVO_OS_STRING == "RHEL 7") && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName1) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName2) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName3) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName4) &&  (GLIDEIN_SITE =!= "SU-ITS" )

Requirements = HAS_SINGULARITY && HAS_CVMFS_singularity_opensciencegrid_org && HAS_CVMFS_xenon_opensciencegrid_org && (OSGVO_OS_STRING == "RHEL 7") && (GLIDEIN_Site =!= "IIT" && GLIDEIN_Site =!= "NotreDame" && GLIDEIN_Site =!= "OSG_US_NMSU_AGGIE_GRID") && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName1) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName2) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName3) && (TARGET.GLIDEIN_ResourceName =!= MY.MachineAttrGLIDEIN_ResourceName4)
request_cpus = 1
request_memory = {request_memory}

max_retries = 5
periodic_release =  (NumJobStarts < JobMaxRetries) && ((CurrentTime - EnteredCurrentStatus) > (10*60))
transfer_output_files = $(job).tar.gz
transfer_input_files = {inputfiles}
+WANT_RCC_ciconnect = True
+ProjectName = "xenon1t"
+AccountingGroup = "group_opportunistic.xenon1t.processing"
+SingularityImage = "{singularity_container}"
when_to_transfer_output = ON_EXIT
transfer_executable = True
# x509userproxy = /home/ershockley/user_cert

queue job from(
{joblist}
)
"""
    return submission_template.format(
        joblist="\n".join(joblist),
        inputfiles=", ".join(inputfiles),
        singularity_container=singularity_container,
        request_memory=request_memory)


def write_OSG_job(jobname, arguments, output_filename, toydata_file, inputfiles, targetfiles,
                  config_path):

    jobname_with_out_dir = os.path.basename(jobname)
    merged_args = "'" + "' '".join(arguments) + "'"

    copy_template = "cp {inputfile} alea/data/{target_path}\n"
    # path_from_template refers to to a prepending path in the source of config
    # e.g. ER/template_XENONnT_ER_field_20.h5 --> the ER is not matched correctly
    # without this

    copy_input_files_to_where_they_belong = ""
    for inputfile, target_path in zip(inputfiles, targetfiles):
        if inputfile == "alea.tar.gz":
            continue
        #  elif inputfile == os.path.basename(config_path):
        #      #  mkdirs = "mkdir -p {config_dir}\n".format(config_dir=os.path.dirname(config_path))
        #      #  copy_input_files_to_where_they_belong += mkdirs
        #      copy_input_files_to_where_they_belong += "cp {inputfile} {config_path}\n".format(
        #          inputfile=os.path.basename(config_path),
        #          config_path=config_path)
        else:
            copy_input_files_to_where_they_belong += copy_template.format(
                inputfile=inputfile, target_path=target_path)

    create_paths_in_alea = ""
    for target_path in targetfiles:
        dirname = os.path.dirname(target_path)
        if dirname != "":
            create_paths_in_alea += "mkdir -p alea/data/{dirname}\n".format(
                dirname=dirname)

    osg_job = """#!/bin/bash

# set -e is needed so a job "can" fail and submission knows about it
set -e
echo "---Check environment---"
which python
which pip
git --version
echo "hostname:"
hostname
echo "GLIDEIN_Site = $GLIDEIN_Site"
echo "GLIDEIN_ResourceName = $GLIDEIN_ResourceName"

tar xvfz alea.tar.gz

# inside the OSG job the input files need to be copied into alea
{create_paths_in_alea}
{copy_input_files_to_where_they_belong}
pip install -r requirements.txt --user
pip install -e . --user

toymc_script=$(python -c 'import pkg_resources; print(pkg_resources.resource_filename("alea","/scripts/run_toymc.py"))')
python $toymc_script {arguments}

ls -ltrha
#  echo $execution_cmd
#  echo "-------------------"

#  $execution_cmd

if [[ -f {toydata_file} ]]
then
    files_to_tar="{output_filename} {toydata_file}"
else
    files_to_tar="{output_filename}"
fi


# tar the output
echo "tar cvfz {jobname}.tar.gz $files_to_tar"
if ! tar cvfz {jobname}.tar.gz $files_to_tar &> /dev/null; then
    rm {jobname}.tar.gz
    echo "TARALL CREATION FAILED"
else
    echo "TARBALL SUCCESFULLY CREATED"
fi
ls -ltrha
mkdir -p jobs
pwd
cp {jobname}.tar.gz jobs
ls -ltrha jobs/
""".format(arguments=merged_args,
           output_filename=output_filename,
           toydata_file=toydata_file,
           jobname=jobname_with_out_dir,
           copy_input_files_to_where_they_belong=
           copy_input_files_to_where_they_belong,
           create_paths_in_alea=create_paths_in_alea)
    with open(jobname, "w") as f:
        f.write(osg_job)


def parse_args(args):
    parser = argparse.ArgumentParser("Submission script for toyMCs in alea")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--local',
                       action='store_true',
                       help="Executes the defined jobs locally")
    group.add_argument('--midway',
                       action='store_true',
                       help="Prepare submission for Midway")
    group.add_argument('--OSG',
                       action='store_true',
                       help="Write out files for submission to OSG")
    parser.add_argument("--submit",
                        action='store_true',
                        help="Submit to OSG/Midway")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Only 1 job will be prepared")
    parser.add_argument("--unpack",
                        action='store_true',
                        help="Unpacks the results from OSG submission")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="YAML config file specifying alea details")
    parser.add_argument("--outputfolder_overwrite",
                        type=str,
                        required=False,
                        default=None,
                        help="Overwriting the outputfolder, usually defined in config.")
    parser.add_argument(
        "--computation",
        type=str,
        required=True,
        choices=["discovery_power", "threshold", "sensitivity", "unpack", "unpack_sensi"],
        help="Type of computation, defined in YAML config file")
    args = parser.parse_args(args)
    return args


def submit_jobs(argv):
    parsed_args = parse_args(argv)
    config_data = read_config(parsed_args.config)
    outputfolder = config_data["outputfolder"]
    if parsed_args.outputfolder_overwrite is not None:
        outputfolder = parsed_args.outputfolder_overwrite
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    abs_config_path = os.path.abspath(parsed_args.config)
    cwd = os.getcwd()
    os.chdir(outputfolder)
    if ((parsed_args.computation == "discovery_power")
        or (parsed_args.computation == "threshold")
        or (parsed_args.computation == "sensitivity")):
        computation = config_data["computation"][parsed_args.computation]

        parameters_to_vary = computation.get("parameters_to_vary", {})
        parameters_in_common = computation.get("parameters_in_common", {})
        if parsed_args.local or parsed_args.midway:
            parameters_in_common["inference_object_args"].update(
                {"config_path": abs_config_path})
        elif parsed_args.OSG:
            parameters_in_common["inference_object_args"].update(
                {"config_path": os.path.basename(parsed_args.config)})
        parameters_to_zip = computation.get("parameters_to_zip", {})
        parameters_as_wildcards = computation.get("parameters_as_wildcards", {})

        if computation.get("use_conveniencevariation", False):
            generate_args_parameter = computation.get("generate_args_parameter")
            InferenceObjectClass = locate(parameters_in_common.get("inference_class_name"))
            signature = inspect.signature(InferenceObjectClass)
            varcon = alea.utils.VariationConvenience(parameters_to_vary=parameters_to_vary,
                                        parameters_to_zip=parameters_to_zip,
                                        parameters_as_wildcards=parameters_as_wildcards,
                                        parameters_in_common=parameters_in_common,
                                        generate_args_parameters=generate_args_parameter,
                                        signature=signature
                                        )
            propagate_guess = computation.get("propagate_guess", False)
            varcon.combined_zip_input(propagate_guess=propagate_guess)
            parameters_to_vary = {}
            parameters_to_zip = varcon.zip_input
            parameters_in_common = parameters_in_common


    if parsed_args.computation == "discovery_power" or parsed_args.computation == "sensitivity":

        if parsed_args.computation == "sensitivity":
            limit_threshold = parameters_in_common.get("limit_threshold", None)
            if limit_threshold is None:
                print("You are running with asymptotics!")
                time.sleep(5)

        fnames, calls = toymc_to_sbatch_call_array_update(
            parameters_to_vary=parameters_to_vary,
            parameters_to_zip=parameters_to_zip,
            parameters_in_common=parameters_in_common)

        if parsed_args.debug:
            calls = [calls[0]]
            fnames = [fnames[0]]
            n_mc_index = calls[0].index("--n_mc") + 1
            calls[0][n_mc_index] = "1"
        print(len(calls), "calls could be submitted.")

        joblist = []
        first = True
        inputfiles, targetfiles = ["alea.tar.gz"], ["alea.tar.gz"]
        for number, c in enumerate(tqdm(calls)):
            if parsed_args.local:
                local_path = pkg_resources.resource_filename(
                    "alea", "/scripts/run_toymc.py")
                local_call = ["python", local_path]
                local_call += c[2:]
                print(local_call)
                subprocess.call(local_call)
                break
            elif parsed_args.midway:
                if parsed_args.debug:
                    print(c)
                continue
            elif parsed_args.OSG:
                parsed_args_toy = toymc_running.parse_cl_arguments(c[2:])
                # this path is absolute to be able to read it locally below
                parsed_args_toy.inference_object_args.update(
                    {"config_path": abs_config_path})
                logging.debug("parsed_args_toy: " + str(parsed_args_toy))

                # here we create a dummy InferenceObject to read the templates that we need to copy
                if first:
                    InferenceObject = locate(
                        parsed_args_toy.inference_class_name)
                    generate_args_index = calls[0].index("--generate_args") + 1
                    generate_args = json.loads(calls[0][generate_args_index])
                    threshold_key_index = calls[0].index("--threshold_key") + 1
                    threshold_key = calls[0][threshold_key_index]
                    logging.debug("initializing a dummy InferenceObject with parameters:")
                    logging.debug("wimp_mass: " + str(parsed_args_toy.wimp_mass))
                    logging.debug("i_batch: " + str(parsed_args_toy.i_batch))
                    logging.debug("livetime: " + str(parsed_args_toy.livetime))
                    logging.debug("limit_threshold: " + str(parsed_args_toy.limit_threshold))
                    logging.debug("toydata_file: " + str(parsed_args_toy.toydata_file))
                    logging.debug("toydata_mode: " + str(parsed_args_toy.toydata_mode))
                    logging.debug("signal_component_name: " + str(parsed_args_toy.signal_component_name))
                    logging.debug("generate_args: " + str(parsed_args_toy.generate_args))
                    logging.debug("threshold_key: " + str(parsed_args_toy.threshold_key))
                    logging.debug("inference_object_args: " + str(parsed_args_toy.inference_object_args))
                    statistical_model = InferenceObject(
                        wimp_mass=parsed_args_toy.wimp_mass,
                        i_batch=parsed_args_toy.i_batch,
                        livetime=parsed_args_toy.livetime,
                        limit_threshold=parsed_args_toy.limit_threshold,
                        toydata_file=parsed_args_toy.toydata_file,
                        toydata_mode=parsed_args_toy.toydata_mode,
                        signal_component_name=parsed_args_toy.
                        signal_component_name,
                        generate_args=generate_args,
                        threshold_key=threshold_key,
                        **parsed_args_toy.inference_object_args)

                    # collect templates to copy as input-files on OSG
                    # We assume that formatting strings are indicated with { }
                    templates_to_copy = []
                    templates_to_copy.append(statistical_model.config_path)
                    templates_to_target = []
                    templates_to_target.append(
                        os.path.basename(statistical_model.config_path))
                    logging.debug("statistical_model.config_path: " + str(statistical_model.config_path))
                    logging.debug("statistical_model.input_files: " + str(statistical_model.input_files))
                    for input_file in statistical_model.input_files:
                        p = re.compile(r"{(.*?)}")
                        result = p.findall(input_file)
                        for res in result:
                            input_file = input_file.replace(
                                "{" + res + "}", "*")
                        else:
                            matches = glob.glob(input_file)
                            for match in matches:
                                templates_to_copy.append(match)
                                #  target_path = os.path.relpath(
                                #      match, config_data["OSG_path"])
                                templates_to_target.append(".")

                    # copy template files to top dir of submission to prepare for input transfer
                    logging.debug("templates_to_copy: " + str(templates_to_copy))
                    # Check whether the basenames of two templates are the same and raise a warning if so
                    check_basename_sanity(templates_to_copy)
                    for template in templates_to_copy:
                        target = os.path.basename(template)
                        if target.endswith(".yaml"):
                            cmd = "cp {source} {target}".format(
                                source=template, target=target)
                            os.system(cmd)
                            continue
                        else:
                            if os.path.exists(target):
                                cmd = "rm {filename}".format(filename=target)
                                os.system(cmd)
                            cmd = "ln -s {source} {filename}".format(
                                source=template,
                                filename=os.path.basename(template))
                            os.system(cmd)

                    templates_to_copy = [
                        os.path.basename(template)
                        for template in templates_to_copy
                    ]

                    # check if threshols-file is specified
                    threshold_index = calls[0].index("--limit_threshold")
                    if calls[0][threshold_index + 1] != "None":
                        threshold_file = calls[0][threshold_index + 1]
                        inputfiles.append(threshold_file)
                        # This assumes that there is no subdirectory specified for the
                        # threshold file...
                        targetfiles.append(".")

                    inputfiles += templates_to_copy
                    targetfiles += templates_to_target
                    logging.debug("inputfiles: " + str(inputfiles))
                    logging.debug("targetfiles: " + str(targetfiles))
                    first = False

                output_filename = parsed_args_toy.output_filename
                toydata_file = parsed_args_toy.toydata_file
                if parsed_args.debug:
                    jobname = "jobs/debug_{computation}_{number}".format(
                        computation=parsed_args.computation, number=number)
                else:
                    jobname = "jobs/{computation}_{number}".format(
                        computation=parsed_args.computation, number=number)
                jobdir = "jobs"
                if parsed_args_toy.toydata_mode == 'read':
                    inputfiles.append(parsed_args_toy.toydata_file)
                    targetfiles.append(".")
                logging.debug('inputfiles: ' + str(inputfiles))
                logging.debug('targetfiles: ' + str(targetfiles))

                if not os.path.exists(jobdir):
                    os.makedirs(jobdir)
                this_call = c[2:]
                logging.debug("this_call: " + str(this_call))
                write_OSG_job(jobname,
                              this_call,
                              output_filename=output_filename,
                              toydata_file=toydata_file,
                              inputfiles=inputfiles,
                              targetfiles=targetfiles,
                              config_path=parsed_args.config)
                joblist.append(jobname)
                if parsed_args.debug:
                    break

        if parsed_args.OSG:
            OSG_parameters = config_data.get("OSG_parameters", {})
            singularity_container = OSG_parameters.get(
                "singularity_container",
                "/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest"
            )
            print(f"Using singularity_container {singularity_container}")
            request_memory = OSG_parameters.get("request_memory", "1000Mb")
            print(f"Using request_memory {request_memory}")
            submission_script = OSG_template(
                joblist=joblist,
                inputfiles=inputfiles,
                singularity_container=singularity_container,
                request_memory=request_memory)
            with open("submit.sub", "w") as f:
                f.write(submission_script)

        if parsed_args.submit:
            if parsed_args.OSG:
                print("CREATNG alea-tarball")
                # The tar ball will exclude all *.tar.gz files
                create_alea_tarball(targetname=inputfiles[0],
                                          targetpath=os.getcwd())
                cmd = "condor_submit submit.sub"
                os.system(cmd)
            elif parsed_args.midway:
                for c in calls:
                    subprocess.call(c)
                    time.sleep(0.2)
            elif parsed_args.local:
                print("Cannot submit local submission!")
                raise SystemExit
        os.chdir(cwd)
    elif parsed_args.computation == "threshold":
        print(
            "This is done after computation of discovery_power and is executed locally"
        )

        if parsed_args.unpack:
            filelist = glob.glob("*.tar.gz")
            for file in filelist:
                if "alea.tar.gz" in file:
                    continue
                cmd = "tar xvfz {file}".format(file=file)
                os.system(cmd)

        threshold_file = computation.get("limit_threshold", "thresholds.hdf5")
        output_filename = parameters_in_common.get("output_filename")
        file_name_pattern = output_filename.split(
            ".hdf5")[0] + "_{n_batch:d}.hdf5"
        print("Writing threshold_file:", threshold_file)
        compute_neyman_thresholds_update(
            file_name_pattern,
            threshold_file,
            parameters_in_common=parameters_in_common,
            parameters_to_vary=parameters_to_vary,
            parameters_as_wildcards=parameters_as_wildcards,
            parameters_to_zip=parameters_to_zip
            )
    elif parsed_args.computation == "unpack":
        print(
            "This is done after computation of discovery_power and is executed locally"
        )
        filelist = glob.glob("*.tar.gz")
        for file in filelist:
            if "alea.tar.gz" in file:
                continue
            cmd = "tar xvfz {file}".format(file=file)
            os.system(cmd)
    elif parsed_args.computation == "unpack_sensi":
        print(
            "This is done after computation of sensitivity and is executed locally"
        )
        filelist = glob.glob("*sensi*.tar.gz")
        for file in filelist:
            if "alea.tar.gz" in file:
                continue
            cmd = "tar xvfz {file}".format(file=file)
            os.system(cmd)


def check_basename_sanity(templates_to_copy):
    basenames = [os.path.basename(template) for template in templates_to_copy]
    # Create a dictionary to store the basenames and corresponding directories
    basename_dirs = {}

    # Iterate over the templates and check for duplicate basenames
    for template, basename in zip(templates_to_copy, basenames):
        if basename not in basename_dirs:
            basename_dirs[basename] = [template]
        else:
            basename_dirs[basename].append(template)

    # Iterate over the dictionary and print the directories with duplicate basenames
    duplicates = False
    for basename, dirs in basename_dirs.items():
        if (len(dirs) > 1) & (len(set(dirs)) > 1):
            print("Duplicate basename:", basename)
            print("Directories:")
            for directory in dirs:
                print(directory)
            print()
            duplicates = True
    if duplicates:
        raise ValueError("Duplicate basenames found!")
