import shutil

import pytest
from unittest import TestCase

from alea.submitter import Submitter
from alea.submitters.local import SubmitterLocal


@pytest.mark.usefixtures("rm_cache")
class TestSubmitter(TestCase):
    """Test of the Submitter class."""

    @classmethod
    def setUp(cls):
        """Initialise the Submitter instance."""
        cls.config_file_path = "unbinned_wimp_running.yaml"
        cls.outputfolder = "test_output"
        cls.set_submitter(cls)

    def set_submitter(self):
        """Set a new submitter instance with SubmitterLocal."""
        self.submitter = SubmitterLocal.from_config(
            self.config_file_path,
            computation="discovery_power",
            debug=True,
            outputfolder=self.outputfolder,
        )

    def test_init(self):
        """Test of __init__ method."""
        with self.assertRaises(
            RuntimeError, msg="Should raise error when directly instantiating Submitter"
        ):
            Submitter.from_config(self.config_file_path)

    def test_submit(self):
        """Test of submit method."""
        self.submitter.submit()
        shutil.rmtree(self.outputfolder, ignore_errors=False)

    def test_all_runner_kwargs(self):
        """Test of all_runner_kwargs method."""
        self.submitter.all_runner_kwargs()
