import subprocess

from alea.submitter import Submitter


class SubmitterLocal(Submitter):
    """Submitter for local machine."""

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterLocal class."""
        super().__init__(*args, **kwargs)

        self.local_configurations = kwargs.get('local_configurations', {})
        self.inputfolder = self.local_configurations.pop('template_path', None)

    def submit(self):
        """Run job in subprocess locally"""
        for job, (script, _) in enumerate(self.computation_tickets_generator()):
            if self.debug:
                print(script)
                if job > 0:
                    break
            subprocess.call(script, shell=True)
