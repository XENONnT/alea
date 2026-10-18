:orphan:

Submitters
==========

Submitters manage the submission and execution of toy Monte Carlo studies to computing clusters or local machines. They handle job orchestration, parameter variation, and result aggregation for large-scale statistical inference campaigns.

The base class is :class:`alea.submitter.Submitter`, which provides the core functionality for generating submission scripts and managing job workflows. Different submitter implementations target various batch systems:

* :class:`alea.submitters.local.SubmitterLocal` — for running jobs locally or in sequence
* :class:`alea.submitters.slurm.SubmitterSLURM` — for SLURM cluster systems
* :class:`alea.submitters.htcondor.SubmitterHTCondor` — for HTCondor batch systems
* :class:`alea.submitters.rcc_slurm.SubmitterRccSlurm` — for RCC SLURM systems

**Key responsibilities:**

* Parse configuration files to define computational workflows
* Generate parameter variations (sweeping over multiple parameter values)
* Create runner instances with appropriate configurations
* Submit or execute jobs on the target system
* Support debug mode for testing submissions
* Handle resubmission of incomplete jobs

Submitters are initialized from a YAML configuration file that specifies the statistical model, parameter-of-interest (POI), and computation options. For details on configuring submitters, see the :doc:`/configuration/runner` page.
