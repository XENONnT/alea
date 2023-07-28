import pytest
import shutil


@pytest.fixture(scope='class')
def rm_cache():
    """Remove pdf_cache directory before and after initializing the TestCase."""
    shutil.rmtree('pdf_cache', ignore_errors=True)
    yield
    shutil.rmtree('pdf_cache', ignore_errors=True)
