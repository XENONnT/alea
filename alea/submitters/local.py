import shlex
import subprocess

from alea.runner import Runner
from alea.submitter import Submitter


class SubmitterLocal(Submitter):
    """Submitter for local machine."""

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterLocal class."""
        self.local_configurations = kwargs.get('local_configurations', {})
        self.template_path = self.local_configurations.pop('template_path', None)
        super().__init__(*args, **kwargs)

    def submit(self):
        """Run job in subprocess locally.
        If debug is True, only return the first instance of Runner.
        """
        for _, (script, _) in enumerate(self.computation_tickets_generator()):
            if self.debug:
                print(script)
                kwargs = Submitter.init_runner_from_args_string(shlex.split(script)[1:])
                runner = Runner(**kwargs)
                return runner
            subprocess.call(script, shell=True)
