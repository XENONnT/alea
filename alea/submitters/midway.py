import os
import time
import shlex
from copy import deepcopy
import importlib_resources

from utilix import batchq

from alea.submitter import Submitter


class SubmitterMidway(Submitter):
    max_jobs: int = 100
    hours: int = 1  # in the unit of hours
    mem_per_cpu: int = 2000  # in the unit of Mb
    container: str = 'xenonnt-development.simg'
    bind = ('/dali', '/project2', '/project')
    partition: str = 'xenon1t'
    qos: str = 'xenon1t'
    cpus_per_task: int = 1
    exclude_nodes: str = 'dali[028-030],midway2-0048'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = self.__class__.__name__
        self.log_dir = self.configuration.runner_config['outputfolder']

    def _submit(self, job, **kwargs):
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
            mem_per_cpu=self.mem_per_cpu,
            cpus_per_task=self.cpus_per_task,
            container=self.container,
            partition=self.partition,
            qos=self.qos,
            bind=self.bind,
            hours=self.hours,
            **kwargs)

    def submit(self, **kwargs):
        """Submits job to batch queue which actually runs the analysis"""
        _jobname = kwargs.pop('jobname', self.name.lower())
        batchq_kwargs = deepcopy(kwargs)
        run_toymc_path = importlib_resources.files('alea') / '/scripts/run_toymc.py'
        for job, (script, output_file, _batchq_kwargs) in enumerate(self.configuration.get_submission_script()):
            if self.debug and job > 0:
                break
            while batchq.count_jobs(_jobname) > self.max_jobs:
                self.logging.info('Too many jobs. Sleeping for 30s.')
                time.sleep(30)
            batchq_kwargs.update(_batchq_kwargs)
            batchq_kwargs['log'] = os.path.join(self.log_dir, f'{output_file}.log')
            batchq_kwargs['jobname'] = f'{_jobname}_{job:03d}'
            self.logging.debug(f"Call '_submit' with job: {job} and kwargs: {batchq_kwargs}.")
            script = f'python {run_toymc_path} ' + ' '.join(map(shlex.quote, script.split(' ')))
            self._submit(script, **batchq_kwargs)
