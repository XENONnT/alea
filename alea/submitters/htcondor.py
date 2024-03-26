# hyperparameters: configuration
# inputs: templates in the specified local directory
# outputs: all *.h5 files in the output folder

import subprocess
import os
import threading
import tempfile
import time
from alea.submitter import Submitter
import logging
logger = logging.getLogger()


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster.
    """
    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self.htcondor_configurations = kwargs.get("slurm_configurations", {})
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        super().__init__(*args, **kwargs)
    
    def _validate_x509_proxy(self, min_valid_hours=20):
        """Ensure $HOME/user_cert exists and has enough time left
        """
        logger.debug('Verifying that the ~/user_cert proxy has enough lifetime')
        shell = Shell('grid-proxy-info -timeleft -file ~/user_cert')
        shell.run()
        valid_hours = int(shell.get_outerr()) / 60 / 60
        if valid_hours < min_valid_hours:
            raise RuntimeError('User proxy is only valid for %d hours. Minimum required is %d hours.' \
                               %(valid_hours, min_valid_hours))
    
    def _generate_sc(self):
        raise NotImplementedError
    
    def _generate_workflow(self):
        raise NotImplementedError
    
    def _plan_and_submit(self):
        raise NotImplementedError  

    def submit_workflow(self):
        raise NotImplementedError


class Shell(object):
    '''
    Provides a shell callout with buffered stdout/stderr, error handling and timeout
    '''
        
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
        returns the combined stdout and stderr from the command
        """
        return self._outerr
    
    
    def get_exit_code(self):
        """
        returns the exit code from the process
        """
        return self._process.returncode
    
    def get_duration(self):
        """
        returns the timing of the command (seconds)
        """
        return self._duration