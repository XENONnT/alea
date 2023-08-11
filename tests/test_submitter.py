import shutil

import pytest
from unittest import TestCase

from alea.submitters import Submitter, SubmitterLocal


@pytest.mark.usefixtures('rm_cache')
class TestSubmitter(TestCase):
    """Test of the Submitter class"""

    @classmethod
    def setUp(cls):
        """Initialise the Submitter instance"""
        cls.config_file_path = 'unbinned_wimp_running.yaml'
        cls.outputfolder = 'test_output.h5'
        cls.set_submitter(cls)

    def set_submitter(self):
        """Set a new submitter instance with SubmitterLocal"""
        self.submitter = SubmitterLocal.from_config(
            self.config_file_path,
            computation='discovery_power',
            debug=True,
            outputfolder=self.outputfolder)

    def test_init(self):
        """Test of __init__ method"""
        try:
            error_raised = True
            Submitter.from_config(self.config_file_path)
            error_raised = False
        except Exception:
            print('Error correctly raised when directly instantiating Submitter')
        else:
            if not error_raised:
                raise RuntimeError(
                    'Should raise error when directly instantiating Submitter')

    def test_submit(self):
        """Test of submit method"""
        self.submitter.submit()
        shutil.rmtree(self.outputfolder, ignore_errors=True)
