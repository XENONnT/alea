import os
import getpass
import tarfile
import shlex
from datetime import datetime
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
import utilix
from utilix.x509 import _validate_x509_proxy
from utilix.tarball import Tarball
from Pegasus.api import (
    Arch,
    Operation,
    Namespace,
    Workflow,
    File,
    Directory,
    FileServer,
    Job,
    Site,
    SiteCatalog,
    Transformation,
    TransformationCatalog,
    ReplicaCatalog,
)
from alea.runner import Runner
from alea.submitter import Submitter
from alea.utils import TEMPLATE_RECORDS, load_yaml, dump_yaml


DEFAULT_IMAGE = "/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest"
WORK_DIR = f"/scratch/{getpass.getuser()}/workflows"
TOP_DIR = Path(__file__).resolve().parents[2]


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster."""

    def __init__(self, *args, **kwargs):
        # General start
        self.htcondor_configurations = kwargs.get("htcondor_configurations", {})
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        self.singularity_image = self.htcondor_configurations.pop(
            "singularity_image", DEFAULT_IMAGE
        )
        self.top_dir = TOP_DIR
        self.work_dir = WORK_DIR
        self.combine_n_outputs = self.htcondor_configurations.pop("combine_n_outputs", 100)

        # A flag to check if limit_threshold is added to the rc
        self.added_limit_threshold = False

        # Cluster size for toymc jobs
        self.cluster_size = self.htcondor_configurations.pop("cluster_size", 1)

        # Resources configurations
        self.request_cpus = self.htcondor_configurations.pop("request_cpus", 1)
        self.request_memory = self.htcondor_configurations.pop("request_memory", 2000)
        self.request_disk = self.htcondor_configurations.pop("request_disk", 2_000)
        self.combine_disk = self.htcondor_configurations.pop("combine_disk", 20_000)

        # Dagman configurations
        self.dagman_retry = self.htcondor_configurations.pop("dagman_retry", 2)
        self.dagman_maxidle = self.htcondor_configurations.pop("dagman_maxidle", 100_000)
        self.dagman_maxjobs = self.htcondor_configurations.pop("dagman_maxjobs", 100_000)

        super().__init__(*args, **kwargs)
        TEMPLATE_RECORDS.lock()

        # Job input configurations
        self.config_file_path = os.path.abspath(self.config_file_path)

        # User can provide a name for the workflow, otherwise it will be the current time
        self._setup_workflow_id()
        # Pegasus workflow directory
        self.workflow_dir = os.path.join(self.work_dir, self.workflow_id)
        self.generated_dir = os.path.join(self.workflow_dir, "generated")
        self.runs_dir = os.path.join(self.workflow_dir, "runs")
        self.outputs_dir = os.path.join(self.workflow_dir, "outputs")
        self.scratch_dir = os.path.join(self.workflow_dir, "scratch")
        self.templates_tarball_dir = os.path.join(self.generated_dir, "templates")

    @property
    def template_tarball(self):
        return os.path.join(self.generated_dir, "templates.tar.gz")

    @property
    def workflow(self):
        return os.path.join(self.generated_dir, "workflow.yml")

    @property
    def pegasus_config(self):
        """Pegasus configurations."""
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
        return pconfig

    @property
    def requirements(self):
        """Make the requirements for the job."""
        # Minimal requirements on singularity/cvmfs/ports/microarchitecture
        _requirements = (
            "HAS_SINGULARITY && HAS_CVMFS_xenon_opensciencegrid_org"
            + " && PORT_2880 && PORT_8000 && PORT_27017"
            + ' && (Microarch >= "x86_64-v3")'
        )

        # If in debug mode, use the MWT2 site because we own it
        if self.debug:
            _requirements += ' && GLIDEIN_ResourceName == "MWT2" '

        return _requirements

    def _tar_h5_files(self, directory, template_tarball="templates.tar.gz"):
        """Tar all needed templates into a flat tarball."""
        # Create a tar.gz archive
        with tarfile.open(template_tarball, "w:gz") as tar:
            tar.add(directory, arcname=os.path.basename(directory))

    def _make_template_tarball(self):
        """Make tarball of the templates if not exists."""
        if not TEMPLATE_RECORDS.uniqueness:
            raise RuntimeError("All files in the template path must have unique basenames.")
        os.makedirs(self.templates_tarball_dir, exist_ok=True)
        if os.listdir(self.templates_tarball_dir):
            raise RuntimeError(
                f"Directory {self.templates_tarball_dir} is not empty. "
                "Please remove it before running the script."
            )

        logger.info(f"Copying templates into {self.templates_tarball_dir}")
        for record in tqdm(TEMPLATE_RECORDS):
            # Copy each file to the destination folder
            shutil.copy(record, self.templates_tarball_dir)
        self._tar_h5_files(self.templates_tarball_dir, self.template_tarball)
        logger.info(f"Tarbal made at {self.template_tarball}")

    def _modify_yaml(self):
        """Modify the statistical model config file to correct the 'template_filename' fields.

        We will use the modified one to upload to OSG. This modification is necessary because the
        templates on the grid will have different path compared to the local ones, and the
        statistical model config file must reflect that.

        """
        # Output file will have the same name as input file but with '_modified' appended
        _output_file = os.path.basename(self.statistical_model_config).replace(
            ".yaml", "_modified.yaml"
        )
        self.modified_statistical_model_config = os.path.join(self.generated_dir, _output_file)

        # Load the YAML data from the original file
        data = load_yaml(self.statistical_model_config)

        # Recursive function to update 'template_filename' fields
        def update_template_filenames(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ["template_filename", "spectrum_name"]:
                        filename = os.path.basename(value)
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
        dump_yaml(self.modified_statistical_model_config, data)
        logger.info(
            "Modified statistical model config file "
            f"written to {self.modified_statistical_model_config}"
        )

    def _setup_workflow_id(self):
        """Set up the workflow ID."""
        # If you have named the workflow, use that name. Otherwise, use the current time as name.
        _workflow_id = self.htcondor_configurations.pop("workflow_id", None)
        now = datetime.now().strftime("%Y%m%d%H%M")
        if _workflow_id:
            workflow_id = (_workflow_id, self.computation, now)
        else:
            workflow_id = (self.computation, now)
        self.workflow_id = "-".join(workflow_id)

    def _generate_sc(self):
        """Generates the SiteCatalog for the workflow."""
        sc = SiteCatalog()

        # Local site: this is the submit host
        logger.debug("Defining local site")
        local = Site("local")
        # Logs and pegasus output goes here. This place is called stash in OSG jargon.
        scratch_dir = Directory(Directory.SHARED_SCRATCH, path=self.scratch_dir)
        scratch_dir.add_file_servers(FileServer(f"file:///{self.scratch_dir}", Operation.ALL))
        # Jobs outputs goes here, but note that it is in scratch so it only stays for short term
        # This place is called stash in OSG jargon.
        storage_dir = Directory(Directory.LOCAL_STORAGE, path=self.outputs_dir)
        storage_dir.add_file_servers(FileServer(f"file:///{self.outputs_dir}", Operation.ALL))
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
            Directory.SHARED_SCRATCH, path=f"/xenon/scratch/{getpass.getuser()}"
        )
        scratch_dir.add_file_servers(
            FileServer(
                "gsidavs://xenon-gridftp.grid.uchicago.edu:2880"
                f"/xenon/scratch/{getpass.getuser()}",
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

    def _generate_tc(self):
        """Generates the TransformationCatalog for the workflow.

        Every executable that is used in the workflow should be here.

        """
        # Wrappers that runs alea_run_toymc
        run_toymc_wrapper = Transformation(
            name="run_toymc_wrapper",
            site="local",
            pfn=self.top_dir / "alea/submitters/run_toymc_wrapper.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        ).add_pegasus_profile(clusters_size=self.cluster_size)

        # Wrappers that collect outputs
        combine = Transformation(
            name="combine",
            site="local",
            pfn=self.top_dir / "alea/submitters/combine.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        )

        # Wrappers that untar outputs
        separate = Transformation(
            name="separate",
            site="local",
            pfn=self.top_dir / "alea/submitters/separate.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        )

        tc = TransformationCatalog()
        tc.add_transformations(run_toymc_wrapper, combine, separate)

        return tc

    def _generate_rc(self):
        """Generate the ReplicaCatalog for the workflow.

        1. The input files for the job, which are the templates in tarball,
            the yaml files, toydata files, alea_run_toymc.py and install.sh.
        2. The output files for the job, which are the toydata and the output files.
        Since the outputs are not known in advance, we will add them in the job definition.

        """
        rc = ReplicaCatalog()

        # Add the templates
        self.f_template_tarball = File(os.path.basename(self.template_tarball))
        rc.add_replica(
            "local",
            os.path.basename(self.template_tarball),
            f"file://{self.template_tarball}",
        )
        # Add the yaml files
        self.f_running_configuration = File(os.path.basename(self.config_file_path))
        rc.add_replica(
            "local",
            os.path.basename(self.config_file_path),
            f"file://{self.config_file_path}",
        )
        self.f_statistical_model_config = File(
            os.path.basename(self.modified_statistical_model_config)
        )
        rc.add_replica(
            "local",
            os.path.basename(self.modified_statistical_model_config),
            f"file://{self.modified_statistical_model_config}",
        )
        # Add run_toymc_wrapper
        self.f_run_toymc_wrapper = File("run_toymc_wrapper.sh")
        rc.add_replica(
            "local",
            "run_toymc_wrapper.sh",
            "file://{}".format(self.top_dir / "alea/submitters/run_toymc_wrapper.sh"),
        )
        # Add alea_run_toymc
        self.f_alea_run_toymc = File("alea_run_toymc.py")
        rc.add_replica(
            "local",
            "alea_run_toymc.py",
            "file://{}".format(self.top_dir / "alea/scripts/alea_run_toymc.py"),
        )
        # Add combine executable
        self.f_combine = File("combine.sh")
        rc.add_replica(
            "local",
            "combine.sh",
            "file://{}".format(self.top_dir / "alea/submitters/combine.sh"),
        )
        # Add separate executable
        self.f_separate = File("separate.sh")
        rc.add_replica(
            "local",
            "separate.sh",
            "file://{}".format(self.top_dir / "alea/submitters/separate.sh"),
        )
        # Untar and install the packages
        self.f_install = File("install.sh")
        rc.add_replica(
            "local",
            "install.sh",
            f"file://{os.path.join(os.path.dirname(utilix.__file__), 'install.sh')}",
        )

        return rc

    def make_tarballs(self):
        """Make tarballs of Ax-based packages if they are in editable user-installed mode."""
        tarballs = []
        tarball_paths = []
        for package_name in ["alea"]:
            _tarball = Tarball(self.generated_dir, package_name)
            if not Tarball.get_installed_git_repo(package_name):
                # Packages should not be non-editable user-installed
                if Tarball.is_user_installed(package_name):
                    raise RuntimeError(
                        f"You should install {package_name} in non-editable user-installed mode."
                    )
            else:
                _tarball.create_tarball()
                tarball = File(_tarball.tarball_name)
                tarball_path = _tarball.tarball_path
                logger.warning(
                    f"Using tarball of user installed package {package_name} at {tarball_path}."
                )
            tarballs.append(tarball)
            tarball_paths.append(tarball_path)
        return tarballs, tarball_paths

    def _initialize_job(
        self,
        name="run_toymc_wrapper",
        cores=1,
        memory=1_700,
        disk=1_000,
        run_on_submit_node=False,
    ):
        """Initilize a Pegasus job, also sets resource profiles.

        Memory and disk in unit of MB.

        """
        job = Job(name)

        if run_on_submit_node:
            job.add_selector_profile(execution_site="local")
            # no other attributes on a local job
            return job

        job.add_profiles(Namespace.CONDOR, "request_cpus", f"{cores}")

        # Set memory and disk requirements
        # If the job fails, retry with more memory and disk
        # Somehow we need to write memory in MB and disk in kB
        memory_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {memory}, (DAGNodeRetry + 1) * {memory})"
        )
        disk_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {disk * 1_000}, (DAGNodeRetry + 1) * {disk * 1_000})"
        )
        job.add_profiles(Namespace.CONDOR, "request_disk", disk_str)
        job.add_profiles(Namespace.CONDOR, "request_memory", memory_str)

        return job

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
            File(f"{self.workflow_id}-{combine_i}-combined_output.tar.gz"), stage_out=True
        )
        combine_job.add_args(f"{self.workflow_id}-{combine_i}")
        self.wf.add_jobs(combine_job)

        return combine_job

    def _add_separate_job(self, combine_i):
        """Add a separate job to the workflow."""
        logger.info(f"Adding separate job {combine_i} to the workflow")
        separate_name = "separate"
        separate_job = self._initialize_job(name=separate_name, run_on_submit_node=True)

        # Separate job configuration: all toymc results and files will be combined into one tarball
        separate_job.add_inputs(File(f"{self.workflow_id}-{combine_i}-combined_output.tar.gz"))
        separate_job.add_args(f"{self.workflow_id}-{combine_i}", self.outputfolder)
        self.wf.add_jobs(separate_job)

        return separate_job

    def _correct_paths_args_dict(self, args_dict):
        """Correct the paths in the arguments dictionary in a hardcoding way."""
        args_dict["statistical_model_args"]["template_path"] = "templates/"

        if "limit_threshold" in args_dict["statistical_model_args"].keys():
            limit_threshold = os.path.basename(
                args_dict["statistical_model_args"]["limit_threshold"]
            )
            args_dict["statistical_model_args"]["limit_threshold"] = limit_threshold

        args_dict["toydata_filename"] = os.path.basename(args_dict["toydata_filename"])
        args_dict["output_filename"] = os.path.basename(args_dict["output_filename"])
        args_dict["statistical_model_config"] = os.path.basename(
            self.modified_statistical_model_config
        )

        return args_dict

    def _reorganize_script(self, script):
        """Extract executable and arguments from the naked scripts.

        Correct the paths on the fly.

        """
        executable = os.path.basename(script.split()[1])
        args_dict = Submitter.runner_kwargs_from_script(shlex.split(script)[2:])

        return executable, args_dict

    def _generate_workflow(self, name="run_toymc_wrapper"):
        """Generate the workflow.

        1. Define catalogs
        2. Generate jobs by iterating over the path-modified tickets
        3. Add jobs to the workflow

        """
        if self.combine_n_jobs != 1:
            raise ValueError(
                f"{self.__class__.__name__} can not combine jobs "
                f"but can only combine outputs so please set {self.combine_n_jobs} to 1."
            )

        # Initialize the workflow
        self.wf = Workflow("alea_workflow")
        self.sc = self._generate_sc()
        self.tc = self._generate_tc()
        self.rc = self._generate_rc()

        # Tarball the editable self-installed packages
        tarballs, tarball_paths = self.make_tarballs()
        for tarball, tarball_path in zip(tarballs, tarball_paths):
            self.rc.add_replica("local", tarball, tarball_path)

        # Iterate over the tickets and generate jobs
        combine_i = 0
        new_to_combine = True

        # Prepare for argument conversion
        _, _, annotations = Runner.runner_arguments()

        # Generate jobstring and output names from tickets generator
        for job_id, (script, _) in enumerate(self.combined_tickets_generator()):
            # If the number of jobs to combine is reached, add a new combine job
            if new_to_combine:
                combine_job = self._add_combine_job(combine_i)
                self._add_separate_job(combine_i)

            # Reorganize the script to get the executable and arguments,
            # in which the paths are corrected
            executable, args_dict = self._reorganize_script(script)
            if not (
                args_dict["toydata_mode"]
                in ["read", "generate_and_store", "generate", "no_toydata"]
            ):
                raise NotImplementedError(
                    f"{args_dict['toydata_mode']} toydata mode is not supported on OSG."
                )

            # Create a job with base requirements
            job = self._initialize_job(
                name=name,
                cores=self.request_cpus,
                memory=self.request_memory,
                disk=self.request_disk,
            )
            job.add_profiles(Namespace.CONDOR, "requirements", self.requirements)

            # Add the limit_threshold to the replica catalog if not added
            if "limit_threshold" in args_dict["statistical_model_args"]:
                limit_threshold = args_dict["statistical_model_args"]["limit_threshold"]
                self.rc.add_replica(
                    "local",
                    os.path.basename(limit_threshold),
                    f"file://{limit_threshold}",
                )
                job.add_inputs(File(os.path.basename(limit_threshold)))

            # Add the inputs and outputs
            job.add_inputs(
                self.f_template_tarball,
                self.f_running_configuration,
                self.f_statistical_model_config,
                self.f_run_toymc_wrapper,
                self.f_alea_run_toymc,
                self.f_install,
                *tarballs,
            )

            if not args_dict["only_toydata"]:
                output_filename = args_dict["output_filename"]
                output_filename_base = os.path.basename(output_filename)
                job.add_outputs(File(output_filename_base), stage_out=False)
                job.set_stdout(File(f"{output_filename_base}.log"), stage_out=False)
                combine_job.add_inputs(File(output_filename_base))
                combine_job.add_inputs(File(f"{output_filename_base}.log"))

            toydata_filename = args_dict["toydata_filename"]
            toydata_filename_base = os.path.basename(toydata_filename)
            if args_dict["toydata_mode"] == "read":
                if not os.path.exists(toydata_filename):
                    raise ValueError(f"Can not find {toydata_filename} containing toydata.")
                # Add toydata as input if needed
                self.rc.add_replica(
                    "local",
                    toydata_filename_base,
                    f"file://{toydata_filename}",
                )
                job.add_inputs(File(toydata_filename_base))
            elif args_dict["toydata_mode"] == "generate_and_store":
                # Only add the toydata file if instructed to do so
                job.add_outputs(File(toydata_filename_base), stage_out=False)
                combine_job.add_inputs(File(toydata_filename_base))

            # Add the arguments into the job
            # Using escaped argument to avoid the shell syntax error
            def _extract_all_to_tuple(kwargs) -> tuple:
                """Generate the submission script from the runner arguments."""
                return tuple(
                    shlex.quote(Submitter.arg_to_str(kwargs[arg], annotation))
                    for arg, annotation in annotations.items()
                )

            # Correct the paths in the arguments
            args_dict = self._correct_paths_args_dict(args_dict)
            args_tuple = _extract_all_to_tuple(args_dict)
            job.add_args(*args_tuple)

            # Add the job to the workflow
            self.wf.add_jobs(job)

            # If the number of jobs to combine is reached, add a new combine job
            if (job_id + 1) % self.combine_n_outputs == 0:
                new_to_combine = True
                combine_i += 1
            else:
                new_to_combine = False

            logger.info(f"Adding job {job_id} to the workflow")
            logger.debug(f"Naked Script: {script}")
            logger.debug(f"Executable: {executable}")
            logger.debug(f"Output: {args_dict['output_filename']}")
            logger.debug(f"Toydata: {args_dict['toydata_filename']}")
            logger.debug(f"Arguments: {args_dict}")

        # Finalize the workflow
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.add_site_catalog(self.sc)
        self.wf.write(file=self.workflow)

    def _us_sites_only(self):
        raise NotImplementedError

    def _exclude_sites(self):
        raise NotImplementedError

    def _this_site_only(self):
        raise NotImplementedError

    def _plan_and_submit(self):
        """Plan and submit the workflow."""
        self.wf.plan(
            submit=not self.debug,
            cluster=["horizontal"],
            cleanup="none",
            sites=["condorpool"],
            verbose=3 if self.debug else 0,
            staging_sites={"condorpool": "staging-davs"},
            output_sites=["local"],
            dir=os.path.dirname(self.runs_dir),
            relative_dir=os.path.basename(self.runs_dir),
            **self.pegasus_config,
        )

    def submit(self, **kwargs):
        """Serve as the main function to submit the workflow."""
        if os.path.exists(self.workflow_dir):
            raise RuntimeError(f"Workflow already exists at {self.workflow_dir}.")

        # ensure we have a proxy with enough time left
        _validate_x509_proxy()

        # 0o755 means read/write/execute for owner, read/execute for everyone else
        os.makedirs(self.generated_dir, 0o755, exist_ok=True)
        os.makedirs(self.runs_dir, 0o755, exist_ok=True)
        os.makedirs(self.outputs_dir, 0o755, exist_ok=True)

        # Modify the statistical model config file to correct the 'template_filename' fields
        self._modify_yaml()

        # Handling templates as part of the inputs
        self._make_template_tarball()

        self._generate_workflow()
        if len(self.wf.jobs):
            self._plan_and_submit()
        if self.debug:
            self.wf.graph(
                output=os.path.join(self.generated_dir, "workflow_graph.dot"), label="xform-id"
            )
            # self.wf.graph(
            #     output=os.path.join(self.generated_dir, "workflow_graph.svg"), label="xform-id"
            # )
