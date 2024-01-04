0.2.1 / 2023-12-08
------------------
* Add optional argument `degree_of_freedom` for `asymptotic_critical_value` by @dachengx in https://github.com/XENONnT/alea/pull/86
* Update readthedocs configurations by @dachengx in https://github.com/XENONnT/alea/pull/88
* Update tutorials by @hammannr in https://github.com/XENONnT/alea/pull/89
* Add column to toyMC results with minuit convergence flag by @kdund in https://github.com/XENONnT/alea/pull/91
* Debug a typo at docstring of fittable parameter by @dachengx in https://github.com/XENONnT/alea/pull/95
* Improve documentation by @hammannr in https://github.com/XENONnT/alea/pull/101
* Update Neyman threshold when changing runner_args by @hammannr in https://github.com/XENONnT/alea/pull/100
* Allow submitter to skip the already succeeded files by @dachengx in https://github.com/XENONnT/alea/pull/94
* Print time usage of `Runner.run` by @dachengx in https://github.com/XENONnT/alea/pull/104
* Get expectation values per likelihood term by @hammannr in https://github.com/XENONnT/alea/pull/106
* Prevent arguments to submission variations being changed by deepcopy-ing them.  by @dachengx in https://github.com/XENONnT/alea/pull/107
* Make error message more explicit that an excecutable is not found and… by @kdund in https://github.com/XENONnT/alea/pull/109
* Read poi and expectation directly from `output_filename` to accelerate `NeymanConstructor` by @dachengx in https://github.com/XENONnT/alea/pull/108
* Direct call of used parameters of model by @dachengx in https://github.com/XENONnT/alea/pull/112
* Add function to get all sources names from all likelihoods by @dachengx in https://github.com/XENONnT/alea/pull/111
* Make sure values of parameters that need re-initialization are not changed by @hammannr in https://github.com/XENONnT/alea/pull/110
* Allow all computation names by @kdund in https://github.com/XENONnT/alea/pull/116
* Debug for the missing argument in `_read_poi` by @dachengx in https://github.com/XENONnT/alea/pull/118
* Remove unnecessary warning given new ptype constraints by @dachengx in https://github.com/XENONnT/alea/pull/119


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.0...v0.2.1


0.2.0 / 2023-09-01
------------------
* Proposal to use pre-commit for continuous integration by @dachengx in https://github.com/XENONnT/alea/pull/78
* Example notebooks by @hammannr in https://github.com/XENONnT/alea/pull/75
* Simplify TemplateSource, CombinedSource and SpectrumTemplateSource by @dachengx in https://github.com/XENONnT/alea/pull/69
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/XENONnT/alea/pull/80
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in https://github.com/XENONnT/alea/pull/82
* Add Submitter and NeymanConstructor by @dachengx in https://github.com/XENONnT/alea/pull/79

New Contributors
* @pre-commit-ci made their first contribution in https://github.com/XENONnT/alea/pull/80

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.1.0...v0.2.0


0.1.0 / 2023-08-11
------------------
* Unify and clean code style and docstring by @dachengx in https://github.com/XENONnT/alea/pull/68
* First runner manipulating statistical model by @dachengx in https://github.com/XENONnT/alea/pull/50
* Set best_fit_args to confidence_interval_args if None by @kdund in https://github.com/XENONnT/alea/pull/76
* Livetime scaling by @kdund in https://github.com/XENONnT/alea/pull/73


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.0.0...v0.1.0


0.0.0 / 2023-07-28
------------------
* readme update with pointer to previous work in lieu of commit history by @kdund in https://github.com/XENONnT/alea/pull/8
* Adds a statistical model base class (under construction by @kdund in https://github.com/XENONnT/alea/pull/7
* change folder/module name by @kdund in https://github.com/XENONnT/alea/pull/9
* Move submission_script.py also from binference to here by @dachengx in https://github.com/XENONnT/alea/pull/10
* Add simple gaussian model by @hammannr in https://github.com/XENONnT/alea/pull/12
* Parameter class by @hammannr in https://github.com/XENONnT/alea/pull/19
* Confidence intervals by @kdund in https://github.com/XENONnT/alea/pull/27
* Update README.md by @kdund in https://github.com/XENONnT/alea/pull/29
* Init code style checking, pytest, and coverage by @dachengx in https://github.com/XENONnT/alea/pull/31
* Add templates for wimp example by @hoetzsch in https://github.com/XENONnT/alea/pull/30
* Removes all hash for parameters not used for each source, and for all… by @kdund in https://github.com/XENONnT/alea/pull/37
* First implementation of an nT-like likelihood by @hammannr in https://github.com/XENONnT/alea/pull/32
* Check if some parameter is not set as guess when fitting by @kdund in https://github.com/XENONnT/alea/pull/44
* Fix likelihood_names check in statistical_model.store_data to handle unnamed likelihoods by @kdund in https://github.com/XENONnT/alea/pull/45
* Create pull_request_template.md by @dachengx in https://github.com/XENONnT/alea/pull/46
* Codes style cleaning by @dachengx in https://github.com/XENONnT/alea/pull/49
* First runner manipulating statistical model by @dachengx in https://github.com/XENONnT/alea/pull/47
* Run test on main not master by @dachengx in https://github.com/XENONnT/alea/pull/55
* Simplify file structure by @dachengx in https://github.com/XENONnT/alea/pull/51
* Move `blueice_extended_model` to `models` by @dachengx in https://github.com/XENONnT/alea/pull/56
* Change data format to only use structured arrays  by @kdund in https://github.com/XENONnT/alea/pull/42
* Another fitting test by @kdund in https://github.com/XENONnT/alea/pull/59
* Add first tests module and file indexing system by @dachengx in https://github.com/XENONnT/alea/pull/54
* Shape parameters by @hammannr in https://github.com/XENONnT/alea/pull/58
* Recover examples folder, update file indexing, add notebooks folder, remove legacies by @dachengx in https://github.com/XENONnT/alea/pull/61
* Remove pdf_cache folder before pytest by @dachengx in https://github.com/XENONnT/alea/pull/65
* Make 0.0.0, initialize documentation structure based on readthedocs, add badges to README by @dachengx in https://github.com/XENONnT/alea/pull/66

New Contributors
* @kdund made their first contribution in https://github.com/XENONnT/alea/pull/8
* @dachengx made their first contribution in https://github.com/XENONnT/alea/pull/10
* @hammannr made their first contribution in https://github.com/XENONnT/alea/pull/12
* @hoetzsch made their first contribution in https://github.com/XENONnT/alea/pull/30
