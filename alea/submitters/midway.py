import os
import time
import inspect

from utilix import batchq

from alea.submitter import Submitter


# suggested default arguments for utilix.batchq.submit_job
BATCHQ_DEFAULT_ARGUMENTS = {
    'hours': 1,  # in the unit of hours
    'mem_per_cpu': 2000,  # in the unit of Mb
    'container': 'xenonnt-development.simg',
    'bind': ('/dali', '/project', '/project2'),
    'partition': 'xenon1t',
    'qos': 'xenon1t',
    'cpus_per_task': 1,
    'exclude_nodes': 'dali[028-030],midway2-0048',
}


class SubmitterMidway(Submitter):
    """
    Submitter for Midway cluster, using utilix.batchq.submit_job.
    The default batchq arguments are defined in BATCHQ_DEFAULT_ARGUMENTS.
    You can also overwrite them by passing them inside configuration file.

    Keyword Args:
        midway_configurations (dict): The configurations for utilix.batchq.submit_job.
            There can be template_path inside it, indicating the path to the template.
    """

    max_jobs = 100

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterMidway class."""
        self.name = self.__class__.__name__
        self.midway_configurations = kwargs.get('midway_configurations', {})
        self.template_path = self.midway_configurations.pop('template_path', None)
        self.batchq_arguments = {**BATCHQ_DEFAULT_ARGUMENTS, **self.midway_configurations}
        self._check_batchq_arguments()
        super().__init__(*args, **kwargs)
        self.log_dir = self.outputfolder

    def _submit(self, job, **kwargs):
        """
        Submits job to batch queue which actually runs the analysis.

        Args:
            job (str): The job script to be submitted.

        Keyword Args:
            jobname (str): The name of the job.
            log (str): The path to the log file.
        """
        jobname = kwargs.pop('jobname', None)
        if jobname is None:
            jobname = self.name

        log = kwargs.pop('log', None)
        if log is None:
            log = os.path.join(self.log_dir, f'{jobname.lower()}.log')

        kwargs_to_pop = []
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
                kwargs_to_pop.append(key)
        for kw in kwargs_to_pop:
            kwargs.pop(kw)

        self.logging.debug(f"Submitting the following job: '{job}'")
        batchq.submit_job(
            job, jobname=jobname, log=log,
            **{**self.batchq_arguments, **kwargs})

    def _check_batchq_arguments(self):
        """Check if the self.batchq_arguments are valid"""
        args = set(inspect.signature(batchq.submit_job).parameters)
        if not set(self.batchq_arguments).issubset(args):
            raise ValueError(
                f'The following arguments are not supported of utilix.batchq.submit_job: '
                f'{set(self.batchq_arguments) - set(args)}.')

    def submit(self, **kwargs):
        """
        Submits job to batch queue which actually runs the analysis.
        Overwrite the BATCHQ_DEFAULT_ARGUMENTS by configuration file.
        If debug is True, only submit the first job.

        Keyword Args:
            jobname (str): The name of the job.
        """
        _jobname = kwargs.pop('jobname', self.name.lower())
        batchq_kwargs = {}
        for job, (script, output_file) in enumerate(self.computation_tickets_generator()):
            if self.debug:
                print(script)
                if job > 0:
                    break
            while batchq.count_jobs(_jobname) > self.max_jobs:
                self.logging.info('Too many jobs. Sleeping for 30s.')
                time.sleep(30)
            batchq_kwargs['jobname'] = f'{_jobname}_{job:03d}'
            batchq_kwargs['log'] = os.path.join(self.log_dir, f'{output_file}.log')
            self.logging.debug(f"Call '_submit' with job: {job} and kwargs: {batchq_kwargs}.")
            self._submit(script, **batchq_kwargs)
