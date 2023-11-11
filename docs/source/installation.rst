:orphan:

Installation
============
You can install alea from PyPI using pip but **beware that it is
listed there as alea-inference!** Thus, you need to run

.. code-block:: console

    $ pip install alea-inference


For the latest version, you can install directly from the GitHub
repository by cloning the repository and running

.. code-block:: console

    $ cd alea
    $ pip install .

If you want to use the submission and toymc running scripts, you
should append `.local/bin`(pip install direction) to your `PATH`
environment variable. For example, `export PATH=$HOME/.local/bin`.


For developers, it is recommended to install alea in editable mode.

.. code-block:: console

    $ cd alea
    $ pip install -e .

And append `alea/bin`(pip install direction) to your `PATH`.


You are now ready to use alea!
