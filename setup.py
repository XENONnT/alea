from setuptools import setup, find_packages
setup(
    name="binference",
    version="0.2",
    packages=['binference',"binference.likelihoods"],
    package_dir={'binference': 'binference'},
    package_data={'binference': ['runpy.sbatch', 'data/*.hdf','scripts/*',"data/tutorial_cache/*"]},
    include_package_data=True,
    author='XENON collaboration',
    author_email="knut.dundas.mora@columbia.edu",
    description="blueice-based inference for nT",

)
