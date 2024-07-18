import os
import getpass
import tarfile
import shlex
import json
import tempfile
import time
import threading
import yaml
import subprocess
from alea.submitter import Submitter
import logging
from pathlib import Path
from Pegasus.api import (
    Workflow,
    Job,
    File,
    Site,
    Directory,
    FileServer,
    Operation,
    Namespace,
    Arch,
    Properties,
    SiteCatalog,
    Transformation,
    TransformationCatalog,
    ReplicaCatalog,
)

from datetime import datetime

DEFAULT_IMAGE = "/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest"
WORK_DIR = f"/scratch/{getpass.getuser()}/workflows"
TOP_DIR = Path(__file__).resolve().parent.parent.parent


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster."""

    def __init__(self, *args, **kwargs):
        # General start
        self.name = self.__class__.__name__
        self.htcondor_configurations = kwargs.get("htcondor_configurations", {})
        self.singularity_image = self.htcondor_configurations.pop(
            "singularity_image", DEFAULT_IMAGE
        )
        self._initial_dir = os.getcwd()
        self.work_dir = WORK_DIR
        self.runs_dir = os.path.join(self.work_dir, "runs")
        self.top_dir = TOP_DIR
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        self.max_jobs_to_combine = self.htcondor_configurations.pop("max_jobs_to_combine", 100)

        # A flag to check if limit_threshold is added to the rc
        self.added_limit_threshold = False

        # Cluster size for toymc jobs
        self.cluster_size = self.htcondor_configurations.pop("cluster_size", 1)

        # Resources configurations
        self.request_cpus = self.htcondor_configurations.pop("request_cpus", 1)
        self.request_memory = self.htcondor_configurations.pop("request_memory", 2000)
        self.request_disk = self.htcondor_configurations.pop("request_disk", 2000000)
        self.combine_disk = self.htcondor_configurations.pop("combine_disk", 20000000)

        # Dagman configurations
        self.dagman_maxidle = self.htcondor_configurations.pop("dagman_maxidle", 100000)
        self.dagman_retry = self.htcondor_configurations.pop("dagman_retry", 2)
        self.dagman_maxjobs = self.htcondor_configurations.pop("dagman_maxjobs", 100000)

        # Pegasus configurations
        self._make_pegasus_config()

        super().__init__(*args, **kwargs)

        # Job input configurations
        self.config_file_path = os.path.abspath(self.config_file_path)
        self._get_statistical_model_config_filename(kwargs)

        # User can provide a name for the workflow, otherwise it will be the current time
        self._setup_wf_id()
        # Pegasus workflow directory
        self.wf_dir = os.path.join(self.runs_dir, self.wf_id)

    def _get_statistical_model_config_filename(self, kwargs):
        """Get the statistical model config file name from the kwargs."""
        _statistical_model_config_filename = kwargs.get("statistical_model_config")
        if os.path.isabs(_statistical_model_config_filename):
            self.statistical_model_config_filename = _statistical_model_config_filename
        else:
            _directory = os.path.dirname(self.config_file_path)
            self.statistical_model_config_filename = os.path.join(
                _directory, _statistical_model_config_filename
            )

    def _validate_x509_proxy(self, min_valid_hours=20):
        """Ensure $X509_USER_PROXY exists and has enough time left.

        This is necessary only if you are going to use Rucio.

        """
        self.x509_user_proxy = os.getenv("X509_USER_PROXY")
        assert self.x509_user_proxy, "Please provide a valid X509_USER_PROXY environment variable."

        logger.debug("Verifying that the X509_USER_PROXY proxy has enough lifetime")
        shell = Shell("grid-proxy-info -timeleft -file %s" % (self.x509_user_proxy))
        shell.run()
        valid_hours = int(shell.get_outerr()) / 60 / 60
        if valid_hours < min_valid_hours:
            raise RuntimeError(
                "User proxy is only valid for %d hours. Minimum required is %d hours."
                % (valid_hours, min_valid_hours)
            )

    def _validate_template_path(self):
        """Validate the template path."""
        assert self.template_path, "Please provide a template path."
        # This path must exists locally, and it will be used to stage the input files
        assert os.path.exists(self.template_path), f"Path {self.template_path} does not exist."

        # Printout the template path file structure
        logger.info("Template path file structure:")
        for dirpath, dirnames, filenames in os.walk(self.template_path):
            for filename in filenames:
                logger.info(f"File: {filename} in {dirpath}")
        if self._contains_subdirectories(self.template_path):
            logger.warning(
                "The template path contains subdirectories. All .h5 files will be tarred."
            )

    def _tar_h5_files(self, directory, output_filename="templates.tar.gz"):
        """Tar all .h5 templates in the directory and its subdirectories into a tarball."""
        # Create a tar.gz archive
        with tarfile.open(output_filename, "w:gz") as tar:
            # Walk through the directory
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith(".h5"):
                        # Get the full path to the file
                        filepath = os.path.join(dirpath, filename)
                        # Add the file to the tar
                        # Specify the arcname to store relative path within the tar
                        tar.add(filepath, arcname=os.path.basename(filename))

    def _make_template_tarball(self):
        """Make tarball of the templates if not exists."""
        self.template_tarball_filename = os.path.join(self._generated_dir(), "templates.tar.gz")
        self._tar_h5_files(self.template_path, self.template_tarball_filename)

    def _modify_yaml(self):
        """Modify the statistical model config file to correct the 'template_filename' fields.

        We will use the modified one to upload to OSG. This modification is necessary because the
        templates on the grid will have different path compared to the local ones, and the
        statistical model config file must reflect that.

        """
        input_file = self.statistical_model_config_filename
        # Output file will have the same name as input file but with '_modified' appended
        _output_file = self._get_file_name(input_file)
        _output_file = _output_file.replace(".yaml", "_modified.yaml")
        output_file = os.path.join(self._generated_dir(), _output_file)
        self.modified_statistical_model_config_filename = output_file

        # Load the YAML data from the original file
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)

        # Recursive function to update 'template_filename' fields
        def update_template_filenames(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "template_filename":
                        filename = value.split("/")[-1]
                        node[key] = filename
                    else:
                        update_template_filenames(value)
            elif isinstance(node, list):
                for item in node:
                    update_template_filenames(item)

        # Update the data
        update_template_filenames(data)

        # Write the updated YAML data to the new file
        # Overwrite if the file already exists
        with open(output_file, "w") as file:
            yaml.safe_dump(data, file)
        logger.info(f"Modified statistical model config file written to {output_file}")

    def _generated_dir(self, work_dir=WORK_DIR):
        """Directory for generated files."""
        return os.path.join(work_dir, "generated", self.wf_id)

    def _contains_subdirectories(self, directory):
        """Check if the specified directory contains any subdirectories.

        Args:
        directory (str): The path to the directory to check.

        Returns:
        bool: True if there are subdirectories inside the given directory, False otherwise.

        """
        # List all entries in the directory
        try:
            for entry in os.listdir(directory):
                # Check if the entry is a directory
                if os.path.isdir(os.path.join(directory, entry)):
                    return True
        except FileNotFoundError:
            print("The specified directory does not exist.")
            return False
        except PermissionError:
            print("Permission denied for accessing the directory.")
            return False

        # If no subdirectories are found
        return False

    def _setup_wf_id(self):
        """Set up the workflow ID."""
        # If you have named the workflow, use that name. Otherwise, use the current time as name.
        self._wf_id = self.htcondor_configurations.pop("wf_id")
        if self._wf_id:
            self.wf_id = (
                self._wf_id + "-" + self.computation + "-" + datetime.now().strftime("%Y%m%d%H%M")
            )
        else:
            self.wf_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def _pegasus_properties(self):
        """Writes the file pegasus.properties.

        This file contains configuration settings used by Pegasus and HTCondor.

        """
        props = Properties()

        # Don't ask why, but this is necessary for the Pegasus API to work
        props["pegasus.metrics.app"] = "XENON"
        props["pegasus.data.configuration"] = "nonsharedfs"

        # Give jobs a total of 1+{retry} tries
        props["dagman.retry"] = self.dagman_retry
        # Make sure we do start too many jobs at the same time
        props["dagman.maxidle"] = self.dagman_maxidle
        # Total number of jobs cap
        props["dagman.maxjobs"] = self.dagman_maxjobs

        # Help Pegasus developers by sharing performance data
        props["pegasus.monitord.encoding"] = "json"
        props["pegasus.catalog.workflow.amqp.url"] = (
            "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"
        )

        # write properties file to ./pegasus.properties
        props.write()
        self.pegasus_properties = props

    def _make_pegasus_config(self):
        """Make the Pegasus configuration into a dict to pass as keywords argument later."""
        pconfig = {}

        pconfig["pegasus.metrics.app"] = "XENON"
        pconfig["pegasus.data.configuration"] = "nonsharedfs"
        pconfig["dagman.retry"] = self.dagman_retry
        pconfig["dagman.maxidle"] = self.dagman_maxidle
        pconfig["dagman.maxjobs"] = self.dagman_maxjobs
        pconfig["pegasus.transfer.threads"] = 4

        # Help Pegasus developers by sharing performance data (optional)
        pconfig["pegasus.monitord.encoding"] = "json"
        pconfig["pegasus.catalog.workflow.amqp.url"] = (
            "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"
        )

        self.pegasus_config = pconfig

    def _generate_sc(self):
        """Generates the SiteCatalog for the workflow."""
        sc = SiteCatalog()

        # Local site: this is the submit host
        logger.debug("Defining local site")
        local = Site("local")
        # Logs and pegasus output goes here. This place is called stash in OSG jargon.
        scratch_dir = Directory(
            Directory.SHARED_SCRATCH, path="{}/scratch/{}".format(self.work_dir, self.wf_id)
        )
        scratch_dir.add_file_servers(
            FileServer("file:///{}/scratch/{}".format(self.work_dir, self.wf_id), Operation.ALL)
        )
        # Jobs outputs goes here, but note that it is in scratch so it only stays for short term
        # This place is called stash in OSG jargon.
        storage_dir = Directory(
            Directory.LOCAL_STORAGE, path="{}/outputs/{}".format(self.work_dir, self.wf_id)
        )
        storage_dir.add_file_servers(
            FileServer("file:///{}/outputs/{}".format(self.work_dir, self.wf_id), Operation.ALL)
        )
        # Add scratch and storage directories to the local site
        local.add_directories(scratch_dir, storage_dir)
        # Add profiles to the local site
        local.add_profiles(Namespace.ENV, HOME=os.environ["HOME"])
        local.add_profiles(Namespace.ENV, GLOBUS_LOCATION="")
        local.add_profiles(
            Namespace.ENV,
            PATH=(
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/bin:"  # noqa
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/condabin:/usr/bin:/bin"  # noqa
            ),
        )
        local.add_profiles(
            Namespace.ENV,
            LD_LIBRARY_PATH=(
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/lib64:"  # noqa
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/lib"  # noqa
            ),
        )
        local.add_profiles(Namespace.ENV, PEGASUS_SUBMITTING_USER=os.environ["USER"])
        local.add_profiles(Namespace.ENV, X509_USER_PROXY=os.environ["X509_USER_PROXY"])

        # Staging sites: for XENON it is physically at dCache in UChicago
        # You will be able to download results from there via gfal commands
        logger.debug("Defining stagging site")
        staging_davs = Site("staging-davs")
        scratch_dir = Directory(
            Directory.SHARED_SCRATCH, path="/xenon/scratch/{}".format(getpass.getuser())
        )
        scratch_dir.add_file_servers(
            FileServer(
                "gsidavs://xenon-gridftp.grid.uchicago.edu:2880/xenon/scratch/{}".format(
                    getpass.getuser()
                ),
                Operation.ALL,
            )
        )
        staging_davs.add_directories(scratch_dir)

        # Condorpool: These are the job nodes on grid
        logger.debug("Defining condorpool")
        condorpool = Site("condorpool")
        condorpool.add_profiles(Namespace.PEGASUS, style="condor")
        condorpool.add_profiles(Namespace.CONDOR, universe="vanilla")
        condorpool.add_profiles(
            Namespace.CONDOR, key="+SingularityImage", value=f'"{self.singularity_image}"'
        )
        # Ignore the site settings, since the container will set all this up inside
        condorpool.add_profiles(Namespace.ENV, OSG_LOCATION="")
        condorpool.add_profiles(Namespace.ENV, GLOBUS_LOCATION="")
        condorpool.add_profiles(Namespace.ENV, PYTHONPATH="")
        condorpool.add_profiles(Namespace.ENV, PERL5LIB="")
        condorpool.add_profiles(Namespace.ENV, LD_LIBRARY_PATH="")
        condorpool.add_profiles(Namespace.ENV, PEGASUS_SUBMITTING_USER=os.environ["USER"])
        condorpool.add_profiles(
            Namespace.CONDOR, key="x509userproxy", value=os.environ["X509_USER_PROXY"]
        )

        # Add the sites to the SiteCatalog
        sc.add_sites(local, staging_davs, condorpool)
        return sc

    def _generate_tc(self, cluster_size=1):
        """Generates the TransformationCatalog for the workflow.

        Every executable that is used in the workflow should be here.

        """
        # Wrappers that runs alea-run_toymc
        run_toymc_wrapper = Transformation(
            name="run_toymc_wrapper",
            site="local",
            pfn=self.top_dir / "alea/submitters/run_toymc_wrapper.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        ).add_pegasus_profile(clusters_size=cluster_size)

        # Wrappers that collect outputs
        combine = Transformation(
            name="combine",
            site="local",
            pfn=self.top_dir / "alea/submitters/combine.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        )

        tc = TransformationCatalog()
        tc.add_transformations(run_toymc_wrapper, combine)

        return tc

    def _generate_rc(self):
        """Generate the ReplicaCatalog for the workflow.

        1. The input files for the job, which are the templates in tarball,
            the yaml files and alea-run_toymc.
        2. The output files for the job, which are the toydata and the output files.
        Since the outputs are not known in advance, we will add them in the job definition.

        """
        rc = ReplicaCatalog()

        # Add the templates
        self.f_template_tarball = File(str(self._get_file_name(self.template_tarball_filename)))
        rc.add_replica(
            "local",
            str(self._get_file_name(self.template_tarball_filename)),
            "file://{}".format(self.template_tarball_filename),
        )
        # Add the yaml files
        self.f_running_configuration = File(str(self._get_file_name(self.config_file_path)))
        rc.add_replica(
            "local",
            str(self._get_file_name(self.config_file_path)),
            "file://{}".format(self.config_file_path),
        )
        self.f_statistical_model_config = File(
            str(self._get_file_name(self.modified_statistical_model_config_filename))
        )
        rc.add_replica(
            "local",
            str(self._get_file_name(self.modified_statistical_model_config_filename)),
            "file://{}".format(self.modified_statistical_model_config_filename),
        )
        # Add run_toymc_wrapper
        self.f_run_toymc_wrapper = File("run_toymc_wrapper.sh")
        rc.add_replica(
            "local",
            "run_toymc_wrapper.sh",
            "file://{}".format(self.top_dir / "alea/submitters/run_toymc_wrapper.sh"),
        )
        # Add alea-run_toymc
        self.f_alea_run_toymc = File("alea-run_toymc")
        rc.add_replica(
            "local",
            "alea-run_toymc",
            "file://{}".format(self.top_dir / "bin/alea-run_toymc"),
        )
        #  Add combine executable
        self.f_combine = File("combine.sh")
        rc.add_replica(
            "local",
            "combine.sh",
            "file://{}".format(self.top_dir / "alea/submitters/combine.sh"),
        )

        return rc

    def _add_combine_job(self, combine_i):
        """Add a combine job to the workflow."""
        logger.info(f"Adding combine job {combine_i} to the workflow")
        combine_name = "combine"
        combine_job = self._initialize_job(
            name=combine_name,
            cores=self.request_cpus,
            memory=self.request_memory * 2,
            disk=self.combine_disk,
        )
        combine_job.add_profiles(Namespace.CONDOR, "requirements", self.requirements)

        # Combine job configuration: all toymc results and files will be combined into one tarball
        combine_job.add_outputs(
            File("%s-%s-combined_output.tar.gz" % (self.wf_id, combine_i)), stage_out=True
        )
        combine_job.add_args(self.wf_id + "-" + str(combine_i))
        self.wf.add_jobs(combine_job)

        return combine_job

    def _generate_workflow(self, name="run_toymc_wrapper"):
        """Generate the workflow.

        1. Define catalogs
        2. Generate jobs by iterating over the path-modified tickets
        3. Add jobs to the workflow

        """
        # Initialize the workflow
        self.wf = Workflow("alea-workflow")
        self.sc = self._generate_sc()
        self.tc = self._generate_tc(cluster_size=self.cluster_size)
        self.rc = self._generate_rc()

        # Job requirements
        self.requirements = self._make_requirements()

        # Iterate over the tickets and generate jobs
        combine_i = 0
        new_to_combine = True
        # Generate jobstring and output names from tickets generator
        # _script for example:
        # alea-submission lq_b8_cevns_running.yaml --computation discovery_power --local --debug
        # _last_output_filename for example: /project/lgrandi/yuanlq/alea_outputs/b8mini/toymc_power_cevns_livetime_1.22_0.50_b8_rate_1.00_0.h5  # noqa
        # _script for example: python3 /home/yuanlq/.local/bin/alea-run_toymc --statistical_model alea.models.BlueiceExtendedModel --poi b8_rate_multiplier --hypotheses '["free","zero","true"]' --n_mc 50 --common_hypothesis None --generate_values '{"b8_rate_multiplier":1.0}' --nominal_values '{"livetime_sr0":1.221,"livetime_sr1":0.5}' --statistical_model_config lq_b8_cevns_statistical_model.yaml --parameter_definition None --statistical_model_args '{"template_path":"/project2/lgrandi/binference_common/nt_cevns_templates"}' --likelihood_config None --compute_confidence_interval False --confidence_level 0.9000 --confidence_interval_kind central --toydata_mode generate_and_store --toydata_filename /project/lgrandi/yuanlq/alea_outputs/b8mini/toyfile_cevns_livetime_1.22_0.50_b8_rate_1.00_0.h5 --only_toydata False --output_filename /project/lgrandi/yuanlq/alea_outputs/b8mini/toymc_power_cevns_livetime_1.22_0.50_b8_rate_1.00_0.h5 --seed None --metadata None  # noqa
        for jobid, (_script, _) in enumerate(self.combined_tickets_generator()):
            # If the number of jobs to combine is reached, add a new combine job
            if new_to_combine:
                combine_job = self._add_combine_job(combine_i)

            # Reorganize the script to get the executable and arguments,
            # in which the paths are corrected
            executable, args_dict = self._reorganize_script(_script)
            if not (args_dict["toydata_mode"] in ["generate_and_store", "generate"]):
                raise NotImplementedError(
                    "Only generate_and_store toydata mode is supported on OSG."
                )

            logger.info(f"Adding job {jobid} to the workflow")
            logger.debug(f"Naked Script: {_script}")
            logger.debug(f"Output: {args_dict['output_filename']}")
            logger.debug(f"Executable: {executable}")
            logger.debug(f"Toydata: {args_dict['toydata_filename']}")
            logger.debug(f"Arguments: {args_dict}")

            # Create a job with base requirements
            job = self._initialize_job(
                name=name,
                cores=self.request_cpus,
                memory=self.request_memory,
                disk=self.request_disk,
            )
            job.add_profiles(Namespace.CONDOR, "requirements", self.requirements)

            # Add the inputs and outputs
            job.add_inputs(
                self.f_template_tarball,
                self.f_running_configuration,
                self.f_statistical_model_config,
                self.f_run_toymc_wrapper,
                self.f_alea_run_toymc,
            )
            if self.added_limit_threshold:
                job.add_inputs(self.f_limit_threshold)

            job.add_outputs(File(args_dict["output_filename"]), stage_out=False)
            combine_job.add_inputs(File(args_dict["output_filename"]))

            # Only add the toydata file if instructed to do so
            if args_dict["toydata_mode"] == "generate_and_store":
                job.add_outputs(File(args_dict["toydata_filename"]), stage_out=False)
                combine_job.add_inputs(File(args_dict["toydata_filename"]))

            # Add the arguments into the job
            # Using escaped argument to avoid the shell syntax error
            def _extract_all_to_tuple(d):
                return tuple(
                    f"{json.dumps(str(d[key])).replace(' ', '')}".replace("'", '\\"')
                    for key in d.keys()
                )

            args_tuple = _extract_all_to_tuple(args_dict)
            job.add_args(*args_tuple)

            # Add the job to the workflow
            self.wf.add_jobs(job)

            # If the number of jobs to combine is reached, add a new combine job
            if (jobid + 1) % self.max_jobs_to_combine == 0:
                new_to_combine = True
                combine_i += 1
            else:
                new_to_combine = False

        # Finalize the workflow
        os.chdir(self._generated_dir())
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.add_site_catalog(self.sc)
        self.wf.write()

    def _initialize_job(
        self,
        name="run_toymc_wrapper",
        run_on_submit_node=False,
        cores=1,
        memory=1_700,
        disk=1_000_000,
    ):
        """Initilize a Pegasus job, also sets resource profiles.

        Memory in unit of MB, and disk in unit of MB.

        """
        job = Job(name)
        job.add_profiles(Namespace.CONDOR, "request_cpus", str(cores))

        if run_on_submit_node:
            job.add_selector_profile(execution_site="local")
            # no other attributes on a local job
            return job

        # Set memory and disk requirements
        # If the job fails, retry with more memory and disk
        memory_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {memory}, (DAGNodeRetry + 1)*{memory})"
        )
        disk_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {disk}, (DAGNodeRetry + 1)*{disk})"
        )
        job.add_profiles(Namespace.CONDOR, "request_disk", disk_str)
        job.add_profiles(Namespace.CONDOR, "request_memory", memory_str)

        return job

    def _make_requirements(self):
        """Make the requirements for the job."""
        # Minimal requirements on singularity/cvmfs/ports/microarchitecture
        requirements_base = (
            "HAS_SINGULARITY && HAS_CVMFS_xenon_opensciencegrid_org"
            + " && PORT_2880 && PORT_8000 && PORT_27017"
            + ' && (Microarch >= "x86_64-v3")'
        )

        # If in debug mode, use the MWT2 site because we own it
        if self.debug:
            requirements_base = requirements_base + ' && GLIDEIN_ResourceName == "MWT2" '

        requirements = requirements_base
        return requirements

    def _reorganize_script(self, _script):
        """Extract executable and arguments from the naked scripts.

        Correct the paths on the fly.

        """
        script_fragments = _script.split(" ")
        _executable = script_fragments[1]
        # Remove the directory and only keep the file name
        executable = self._get_file_name(_executable)

        args_dict = self._parse_command_args(_script)

        # Add the limit_threshold to the replica catalog if not added
        if (
            not self.added_limit_threshold
            and "limit_threshold" in args_dict["statistical_model_args"].keys()
        ):
            self.limit_threshold_filename = args_dict["statistical_model_args"]["limit_threshold"]
            self._add_limit_threshold()

        # Correct the paths in the arguments
        args_dict = self._correct_paths_args_dict(args_dict)

        return executable, args_dict

    def _add_limit_threshold(self):
        """Add the Neyman thresholds limit_threshold to the replica catalog."""
        self.f_limit_threshold = File(str(self._get_file_name(self.limit_threshold_filename)))
        self.rc.add_replica(
            "local",
            str(self._get_file_name(self.limit_threshold_filename)),
            "file://{}".format(self.limit_threshold_filename),
        )
        self.added_limit_threshold = True

    def _correct_paths_args_dict(self, args_dict):
        """Correct the paths in the arguments dictionary in a hardcoding way."""
        args_dict["statistical_model_args"]["template_path"] = "templates/"

        if "limit_threshold" in args_dict["statistical_model_args"].keys():
            limit_threshold_filename = self._get_file_name(
                args_dict["statistical_model_args"]["limit_threshold"]
            )
            args_dict["statistical_model_args"]["limit_threshold"] = limit_threshold_filename

        toydata_filename = self._get_file_name(args_dict["toydata_filename"])
        args_dict["toydata_filename"] = toydata_filename

        output_filename = self._get_file_name(args_dict["output_filename"])
        args_dict["output_filename"] = output_filename

        statistical_model_config = self._get_file_name(
            self.modified_statistical_model_config_filename
        )
        args_dict["statistical_model_config"] = statistical_model_config

        return args_dict

    def _parse_command_args(self, command):
        """Parse the command line arguments and return a dictionary with the flags and their values.

        Example command: python3 /home/yuanlq/.local/bin/alea-run_toymc --statistical_model alea.models.BlueiceExtendedModel --poi b8_rate_multiplier --hypotheses '["free","zero","true"]' --n_mc 50 --common_hypothesis None --generate_values '{"b8_rate_multiplier":1.0}' --nominal_values '{"livetime_sr0":1.221,"livetime_sr1":0.5}' --statistical_model_config lq_b8_cevns_statistical_model.yaml --parameter_definition None --statistical_model_args '{"template_path":"/project2/lgrandi/binference_common/nt_cevns_templates"}' --likelihood_config None --compute_confidence_interval False --confidence_level 0.9000 --confidence_interval_kind central --toydata_mode generate_and_store --toydata_filename /project/lgrandi/yuanlq/alea_outputs/b8mini/toyfile_cevns_livetime_1.22_0.50_b8_rate_1.00_0.h5 --only_toydata False --output_filename /project/lgrandi/yuanlq/alea_outputs/b8mini/toymc_power_cevns_livetime_1.22_0.50_b8_rate_1.00_0.h5 --seed None --metadata None  # noqa

        """
        # Use shlex to handle spaces within quotes correctly
        parts = shlex.split(command)

        # Dictionary to store command flags and their values
        args_dict = {}

        # Iterator to go through all parts
        it = iter(parts)
        for part in it:
            if part.startswith("--"):
                flag = part[2:]  # Remove '--' from the flag name
                next_part = next(it, None)
                # Check if the next part is also a flag or end of the command
                if next_part is None or next_part.startswith("--"):
                    args_dict[flag] = True  # No value means boolean flag set to True
                    continue
                # Try to handle JSON values correctly
                if next_part.startswith("{"):
                    # Reconstruct the complete JSON by consuming parts until the closing '}'
                    json_value = next_part
                    while not json_value.strip().endswith("}"):
                        json_value += " " + next(it, "")
                    try:
                        args_dict[flag] = json.loads(json_value)
                    except json.JSONDecodeError:
                        args_dict[flag] = json_value  # Store raw string if JSON parsing fails
                else:
                    args_dict[flag] = next_part

        return args_dict

    def _get_file_name(self, file_path):
        """Get the filename from the file path."""
        return os.path.basename(file_path)

    def _us_sites_only(self):
        raise NotImplementedError

    def _exclude_sites(self):
        raise NotImplementedError

    def _this_site_only(self):
        raise NotImplementedError

    def _check_workflow_exists(self):
        """Check if the workflow already exists."""
        if os.path.exists(self.wf_dir):
            logger.error(f"Workflow already exists at {self.wf_dir}. Exiting.")
            return

    def _plan_and_submit(self):
        """Plan and submit the workflow."""
        os.chdir(self._generated_dir())
        self.wf.plan(
            submit=not self.debug,
            cluster=["horizontal"],
            cleanup="none",
            sites=["condorpool"],
            verbose=3 if self.debug else 0,
            staging_sites={"condorpool": "staging-davs"},
            output_sites=["local"],
            dir=os.path.dirname(self.wf_dir),
            relative_dir=self.wf_id,
            **self.pegasus_config,
        )

        print(f"Worfklow written to \n\n\t{self.wf_dir}\n\n")

    def _warn_outputfolder(self):
        """Warn users about the outputfolder in running config won't be really used."""
        logger.warning(
            "The outputfolder in the running configuration %s won't be used in this submission."
            % (self.outputfolder)
        )
        logger.warning(
            "Instead, you should find your outputs at %s"
            % (self.work_dir + "/outputs/" + self.wf_id)
        )

    def _check_filename_unique(self):
        """Check if all the files in the template path are unique."""
        all_files = []
        for _, _, filenames in os.walk(self.template_path):
            for filename in filenames:
                all_files.append(filename)
        if len(all_files) != len(set(all_files)):
            raise RuntimeError("All files in the template path must have unique names.")

    def submit(self, **kwargs):
        """Serve as the main function to submit the workflow."""
        self._check_workflow_exists()
        self._validate_x509_proxy()

        #  0o755 means read/write/execute for owner, read/execute for everyone else
        try:
            os.makedirs(self._generated_dir(), 0o755)
        except FileExistsError:
            logger.error(f"Workflow directory {self._generated_dir()} already exists. Exiting.")
        os.makedirs(self.runs_dir, 0o755, exist_ok=True)

        # Modify the statistical model config file to correct the 'template_filename' fields
        self._modify_yaml()

        # Handling templates as part of the inputs
        self._validate_template_path()
        self._check_filename_unique()
        self._make_template_tarball()

        self._generate_workflow()
        self._plan_and_submit()

        # Return to initial dir, as we are done.
        logger.info("We are done. Returning to initial directory.")
        self._warn_outputfolder()
        os.chdir(self._initial_dir)


class Shell(object):
    """Provides a shell callout with buffered stdout/stderr, error handling and timeout."""

    def __init__(self, cmd, timeout_secs=1 * 60 * 60, log_cmd=False, log_outerr=False):
        self._cmd = cmd
        self._timeout_secs = timeout_secs
        self._log_cmd = log_cmd
        self._log_outerr = log_outerr
        self._process = None
        self._out_file = None
        self._outerr = ""
        self._duration = 0.0

    def run(self):
        def target():

            self._process = subprocess.Popen(
                self._cmd,
                shell=True,
                stdout=self._out_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            self._process.communicate()

        if self._log_cmd:
            print(self._cmd)

        # temp file for the stdout/stderr
        self._out_file = tempfile.TemporaryFile(prefix="outsource-", suffix=".out")

        ts_start = time.time()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self._timeout_secs)
        if thread.is_alive():
            # do our best to kill the whole process group
            try:
                kill_cmd = "kill -TERM -%d" % (os.getpgid(self._process.pid))
                kp = subprocess.Popen(kill_cmd, shell=True)
                kp.communicate()
                self._process.terminate()
            except Exception:
                pass
            thread.join()
            # log the output
            self._out_file.seek(0)
            stdout = self._out_file.read().decode("utf-8").strip()
            if self._log_outerr and len(stdout) > 0:
                print(stdout)
            self._out_file.close()
            raise RuntimeError(
                "Command timed out after %d seconds: %s" % (self._timeout_secs, self._cmd)
            )

        self._duration = time.time() - ts_start

        # log the output
        self._out_file.seek(0)
        self._outerr = self._out_file.read().decode("utf-8").strip()
        if self._log_outerr and len(self._outerr) > 0:
            print(self._outerr)
        self._out_file.close()

        if self._process.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code (%d): %s\n%s"
                % (self._process.returncode, self._cmd, self._outerr)
            )

    def get_outerr(self):
        """Returns the combined stdout and stderr from the command."""
        return self._outerr

    def get_exit_code(self):
        """Returns the exit code from the process."""
        return self._process.returncode

    def get_duration(self):
        """Returns the timing of the command (seconds)"""
        return self._duration
