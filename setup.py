from setuptools import setup, find_packages
setup(
    name="alea",
    version="0.2",
    packages=['alea',"alea.likelihoods"],
    package_dir={'alea': 'alea'},
    package_data={'alea': ['runpy.sbatch', 'data/*.hdf','scripts/*',"data/tutorial_cache/*"]},
    include_package_data=True,
    author='XENON collaboration',
    author_email="knut.dundas.mora@columbia.edu",
    description="blueice-based inference for nT",

)
