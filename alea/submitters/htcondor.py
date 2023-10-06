from alea.submitter import Submitter
import htcondor
import os


HTC_DEFAULT_ARGUMENTS = {
    "should_transfer_files": "yes",
    "when_to_transfer_output": "on_exit",
}


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster."""
    def __init__(self, *args, **kwargs):
        self.htcondor_configurations = kwargs.get("htcondor_configurations", {})
        self.htcondor_arguments = {**HTC_DEFAULT_ARGUMENTS, **self.htcondor_configurations.pop("htcondor_arguments", {})}
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        self.use_bash_script = self.htcondor_configurations.pop("use_bash_script", None)
        self.log_dir = self.htcondor_configurations.pop("log_dir", None)
        super().__init__(*args, **kwargs)
        if not self.log_dir:
            self.log_dir = self.outputfolder

    def _htcondor_job_from_script(self, script, **kwargs):
        _script_list = script.split(" ")
        _executable = _script_list[1]
        _arguments = " ".join(_script_list[2:]).replace(r'"', r'""')
        jobname = kwargs.pop("jobname", "")

        job = {
            "executable": _executable,
            "arguments": f"\"{_arguments}\"",
            "output": os.path.join(self.log_dir, f"out_{jobname}.txt"),
            "error": os.path.join(self.log_dir, f"err_{jobname}.txt"),
            "log": os.path.join(self.log_dir, f"log_{jobname}.log"),
            **kwargs
        }

        return job

    def _manual_htcondor_job_from_script(self, script, **kwargs):
        _executable = self.use_bash_script
        _arguments = script.replace(r'"', r'""')
        jobname = kwargs.pop("jobname", "")

        job = {
            "executable": _executable,
            "arguments": f"\"{_arguments}\"",
            "output": os.path.join(self.log_dir, f"out_{jobname}.txt"),
            "error": os.path.join(self.log_dir, f"err_{jobname}.txt"),
            "log": os.path.join(self.log_dir, f"log_{jobname}.log"),
            **kwargs
        }

        return job

    def _submit(self, job):
        _job = htcondor.Submit(job)
        self.logging.debug(f"Submitting the following job: '{_job}'")
        submit_result = htcondor.Schedd().submit(_job)
        self.logging.debug(f"Submitted to cluster: {submit_result.cluster()}")

    def submit(self, **kwargs):
        """Submit job to HTCondor
        If debug is True, only submit first job.
        use_bash_script: filepath of executable or None, bypass direct condor
        submission and use user-defined bash script instead. Useful for manual
        setup of singularity container.
        """
        htc_kwargs = {}
        for key, val in self.htcondor_arguments.items():
            if val is not None:
                htc_kwargs[key] = val

        for jobnr, (script, outfilename) in enumerate(self.computation_tickets_generator()):
            htc_kwargs["jobname"] = os.path.splitext(os.path.basename(outfilename))[0]

            if self.debug:
                print(script)
                if jobnr > 0:
                    break

            if self.use_bash_script is None:
                self.logging.debug(f"Generate HTCondor jobticket with script: {script} and kwargs: {htc_kwargs}.")
                jobticket = self._htcondor_job_from_script(script, **htc_kwargs)
            elif os.path.isfile(self.use_bash_script):
                self.logging.debug(f"Generate manual HTCondor jobticket with bashfile: {self.use_bash_script}, script: {script}, and kwargs: {htc_kwargs}.")
                jobticket = self._manual_htcondor_job_from_script(script, **htc_kwargs)
            else:
                raise FileNotFoundError(
                    f"{self.use_bash_script}"
                    "is not a valid filename or does not exist,"
                    "use_bash_script must be valid filepath of an executable, or null"
                )

            self.logging.debug(f"Call _submit with following jobticket: {jobticket}")
            self._submit(jobticket)
        self.logging.info(f"All {jobnr+1} jobs submitted.")
