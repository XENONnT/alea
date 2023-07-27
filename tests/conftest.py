import pytest
import shutil


@pytest.fixture(scope='class')
def rm_cache():
    shutil.rmtree('pdf_cache', ignore_errors=True)
