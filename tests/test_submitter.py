import shutil

import pytest
from unittest import TestCase

from alea.submitters import SubmitterLocal


@pytest.mark.usefixtures('rm_cache')
class TestSubmitter(TestCase):
    """Test of the Submitter class"""

    @classmethod
    def setUp(cls):
        """Initialise the Submitter instance"""
        cls.outputfolder = 'test_output'
        cls.set_submitter(cls)

    def set_submitter(self):
        """Set a new submitter instance with SubmitterLocal"""
        self.submitter = SubmitterLocal.from_config(
            'unbinned_wimp_running.yaml',
            computation='discovery_power',
            debug=True,
            outputfolder=self.outputfolder)

    def test_submitter(self):
        """Test of submit method"""
        self.submitter.submit()
        shutil.rmtree(self.outputfolder, ignore_errors=True)
