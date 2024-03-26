import subprocess
import os
import threading
import tempfile
import time
from alea.submitter import Submitter
import logging
from Pegasus.api import *
from datetime import datetime

DEFAULT_IMAGE = "/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest"
WORK_DIR = "/scratch/$USER/workflows"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SubmitterHTCondor(Submitter):
    """
    Submitter for htcondor cluster.
    """
    def __init__(self, *args, **kwargs):
        # General start
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__
        self.htcondor_configurations = kwargs.get("slurm_configurations", {})
        self.singularity_image = self.htcondor_configurations.pop("singularity_image", DEFAULT_IMAGE)
        self._initial_dir = os.getcwd()
        self.work_dir = WORK_DIR
        self._setup_wf_id()

        # Job input configurations
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        
        # Resources configurations
        self.request_cpus = self.htcondor_configurations.pop("request_cpus", 1)
        self.request_memory = self.htcondor_configurations.pop("request_memory", "2 GB")
        self.request_disk = self.htcondor_configurations.pop("request_disk", "2 GB")
        
        # Dagman configurations
        self.dagman_maxidle = self.htcondor_configurations.pop("dagman_maxidle", 100000)
        self.dagman_retry = self.htcondor_configurations.pop("dagman_retry", 2)
        self.dagman_maxjobs = self.htcondor_configurations.pop("dagman_maxjobs", 100000)


    def _setup_wf_id(self):
        """
        Set up the workflow ID.
        """
        # If you have named the workflow, use that name. Otherwise, use the current time as name.
        self._wf_id = self.htcondor_configurations.pop("wf_id")
        if self._wf_id:
            self.wf_id = self._wf_id
        else:
            self.wf_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


    def _validate_x509_proxy(self, min_valid_hours=20):
        """
        Ensure $HOME/user_cert exists and has enough time left. 
        """
        logger.debug('Verifying that the ~/user_cert proxy has enough lifetime')
        shell = Shell('grid-proxy-info -timeleft -file ~/user_cert')
        shell.run()
        valid_hours = int(shell.get_outerr()) / 60 / 60
        if valid_hours < min_valid_hours:
            raise RuntimeError('User proxy is only valid for %d hours. Minimum required is %d hours.' \
                               %(valid_hours, min_valid_hours))
    

    def _pegasus_properties(self):
        """
        Writes the file pegasus.properties. 
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
        props["pegasus.catalog.workflow.amqp.url"] = "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"

        # write properties file to ./pegasus.properties
        props.write()
        self._pegasus_properties = props


    def _generate_sc(self):
        sc = SiteCatalog()
        
        # Local site - this is the submit host
        logger.debug("Defining local site")
        local = Site("local")
        # Logs and pegasus output goes here
        scratch_dir = Directory(
            Directory.SHARED_SCRATCH, path='{}/scratch/{}'.format(self.work_dir, self._wf_id)
        )
        scratch_dir.add_file_servers(
            FileServer('file:///{}/scratch/{}'.format(self.work_dir, self.wf_id), Operation.ALL)
        )
        # Jobs outputs goes here
        storage_dir = Directory(
            Directory.LOCAL_STORAGE, path='{}/outputs/{}'.format(self.work_dir, self.wf_id)
        )
        storage_dir.add_file_servers(
            FileServer('file:///{}/outputs/{}'.format(self.work_dir, self.wf_id), Operation.ALL)
        )
        # Add scratch and storage directories to the local site
        local.add_directories(scratch_dir, storage_dir)



    

    def _generate_tc(self):
        raise NotImplementedError
    

    def _generate_rc(self):
        raise NotImplementedError


    def _generate_workflow(self):
        self.wf = Workflow('alea')
        self.sc = self._generate_sc()
        self.tc = self._generate_tc()
        self.rc = self._generate_rc()

    
    def _plan_and_submit(self):
        raise NotImplementedError  


    def submit_workflow(self):
        self._pegasus_properties()

        # Return to initial dir, as we are done.
        logger.info('We are done. Returning to initial directory.')
        os.chdir(self._initial_dir)


class Shell(object):
    """
    Provides a shell callout with buffered stdout/stderr, error handling and timeout
    """ 
    def __init__(self, cmd, timeout_secs = 1*60*60, log_cmd = False, log_outerr = False):
        self._cmd = cmd
        self._timeout_secs = timeout_secs
        self._log_cmd = log_cmd
        self._log_outerr = log_outerr
        self._process = None
        self._out_file = None
        self._outerr = ''
        self._duration = 0.0


    def run(self):
        def target():
                        
            self._process = subprocess.Popen(self._cmd, shell=True, 
                                             stdout=self._out_file, 
                                             stderr=subprocess.STDOUT,
                                             preexec_fn=os.setpgrp)
            self._process.communicate()

        if self._log_cmd:
            print(self._cmd)
                    
        # temp file for the stdout/stderr
        self._out_file = tempfile.TemporaryFile(prefix='outsource-', suffix='.out')
        
        ts_start = time.time()
        
        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self._timeout_secs)
        if thread.is_alive():
            # do our best to kill the whole process group
            try:
                kill_cmd = 'kill -TERM -%d' %(os.getpgid(self._process.pid))
                kp = subprocess.Popen(kill_cmd, shell=True)
                kp.communicate()
                self._process.terminate()
            except:
                pass
            thread.join()
            # log the output
            self._out_file.seek(0)
            stdout = self._out_file.read().decode("utf-8").strip()
            if self._log_outerr and len(stdout) > 0:
                print(stdout)
            self._out_file.close()
            raise RuntimeError('Command timed out after %d seconds: %s' %(self._timeout_secs, self._cmd))
        
        self._duration = time.time() - ts_start
        
        # log the output
        self._out_file.seek(0) 
        self._outerr = self._out_file.read().decode("utf-8").strip()
        if self._log_outerr and len(self._outerr) > 0:
            print(self._outerr)
        self._out_file.close()
        
        if self._process.returncode != 0:
            raise RuntimeError('Command exited with non-zero exit code (%d): %s\n%s' \
                               %(self._process.returncode, self._cmd, self._outerr))

    def get_outerr(self):
        """
        Returns the combined stdout and stderr from the command
        """
        return self._outerr
    
    
    def get_exit_code(self):
        """
        Returns the exit code from the process
        """
        return self._process.returncode
    
    def get_duration(self):
        """
        Returns the timing of the command (seconds)
        """
        return self._duration