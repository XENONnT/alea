0.3.2 / 2025-02-03
------------------
* Use `importlib.resources.files` and support python 3.11 by @dachengx in https://github.com/XENONnT/alea/pull/240

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.3.1...v0.3.2


0.3.1 / 2024-12-11
------------------
* Set `request_cpus` as integer by @dachengx in https://github.com/XENONnT/alea/pull/220
* Keep only one `Directory.SHARED_SCRATCH` by @dachengx in https://github.com/XENONnT/alea/pull/222
* Revert "Keep only one `Directory.SHARED_SCRATCH`" by @dachengx in https://github.com/XENONnT/alea/pull/223
* Add the workflow id to the folder in staging-davs by @dachengx in https://github.com/XENONnT/alea/pull/224
* Fix livetime bug in get_expectation_values() by @hammannr in https://github.com/XENONnT/alea/pull/227
* Add file hash to blueice hash by @hammannr in https://github.com/XENONnT/alea/pull/225
* Make metadata error message more informative by @hammannr in https://github.com/XENONnT/alea/pull/229
* Add unused needs_reinit parameters to ignored parameters by @hammannr in https://github.com/XENONnT/alea/pull/230
* Tar user-installed blueice to osg nodes by @hammannr in https://github.com/XENONnT/alea/pull/231
* Source-wise interpolation by @hammannr in https://github.com/XENONnT/alea/pull/228
* Fix bug, only add tarball when git installed by @dachengx in https://github.com/XENONnT/alea/pull/233
* Switch to master for docformatter by @dachengx in https://github.com/XENONnT/alea/pull/234
* Update blueice dependence by @hammannr in https://github.com/XENONnT/alea/pull/236


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.3.0...v0.3.1


0.3.0 / 2024-09-21
------------------
* Use `pyproject.toml` to install alea-inference by @dachengx in https://github.com/XENONnT/alea/pull/192
* Removed deprecated `_pegasus_properties` by @FaroutYLq in https://github.com/XENONnT/alea/pull/196
* Simplify `SubmitterHTCondor` by @dachengx in https://github.com/XENONnT/alea/pull/193
* Debug for pypi build by @dachengx in https://github.com/XENONnT/alea/pull/197
* There was a typo in docstr by @FaroutYLq in https://github.com/XENONnT/alea/pull/198
* Prefer f-string than format by @dachengx in https://github.com/XENONnT/alea/pull/200
* Use tree structure work directory by @dachengx in https://github.com/XENONnT/alea/pull/202
* Use MB all the time in `SubmitterHTCondor`, no more kB by @dachengx in https://github.com/XENONnT/alea/pull/201
* Decompress outputs into `outputfolder` by @dachengx in https://github.com/XENONnT/alea/pull/199
* Support more `toydata_mode` by @dachengx in https://github.com/XENONnT/alea/pull/206
* No need to plan or submit workflow if no job added by @dachengx in https://github.com/XENONnT/alea/pull/207
* Save log for OSG jobs by @dachengx in https://github.com/XENONnT/alea/pull/208
* Use `shlex.quote` to convert the arguments into unix format by @dachengx in https://github.com/XENONnT/alea/pull/209
* Fix the usage of scripts by @dachengx in https://github.com/XENONnT/alea/pull/210
* Add `spectrum_axis` configuration for `SpectrumTemplateSource` by @dachengx in https://github.com/XENONnT/alea/pull/212
* Use utilix to validate X509 proxy by @dachengx in https://github.com/XENONnT/alea/pull/213
* Tarball all needed templates from different folder by @dachengx in https://github.com/XENONnT/alea/pull/214
* Fix the path of `alea_run_toymc` script by @dachengx in https://github.com/XENONnT/alea/pull/216
* Tarball alea for later user installation by @dachengx in https://github.com/XENONnT/alea/pull/215
* Use `install.sh` from utilix by @dachengx in https://github.com/XENONnT/alea/pull/217
* Bump version of utilix by @dachengx in https://github.com/XENONnT/alea/pull/218


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.8...v0.3.0


0.2.8 / 2024-08-26
------------------
* Fix OSG submission by @hammannr in https://github.com/XENONnT/alea/pull/189


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.7...v0.2.8

0.2.7 / 2024-08-07
------------------
* Trigger PyPI workflow on "published" by @hammannr in https://github.com/XENONnT/alea/pull/185
* Fix apply efficiency by @hammannr in https://github.com/XENONnT/alea/pull/187
* Conditional Parameters by @hammannr in https://github.com/XENONnT/alea/pull/186
* Enable choosing the fit strategy by @hammannr in https://github.com/XENONnT/alea/pull/182


**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.6...v0.2.7

0.2.6 / 2024-07-31
------------------
* Defunctionalize `apply_efficiency`, apply efficiency when `efficiency_name` is specified by @dachengx in https://github.com/XENONnT/alea/pull/183

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.5...v0.3.0


0.2.5 / 2024-07-30
------------------
* Consistent sorting for BlueiceExtendedModel by @hammannr in https://github.com/XENONnT/alea/pull/149
* Fixed data storing by @hammannr in https://github.com/XENONnT/alea/pull/152
* Add lxml_html_clean to fix readthedocs building error by @zihaoxu98 in https://github.com/XENONnT/alea/pull/157
* Fitting index variables by @zihaoxu98 in https://github.com/XENONnT/alea/pull/156
* Print Argument combinations to be submitted by @hammannr in https://github.com/XENONnT/alea/pull/151
* Minor changes to fitting index variables (PR #156) by @hammannr in https://github.com/XENONnT/alea/pull/159
* Set `i_batch` for `SubmitterLocal` when submitting by @dachengx in https://github.com/XENONnT/alea/pull/164
* Debug for interpolator deduction of `NeymanConstructor` by @dachengx in https://github.com/XENONnT/alea/pull/165
* The first i batch should be 0 by @dachengx in https://github.com/XENONnT/alea/pull/166
* Try prefix every file path in likelihood configuration with template folder by @dachengx in https://github.com/XENONnT/alea/pull/169
* Forbid prexing every key when adapt_likelihood_config_for_blueice by @FaroutYLq in https://github.com/XENONnT/alea/pull/170
* Refactored Pegasus-based OSG submitter by @FaroutYLq in https://github.com/XENONnT/alea/pull/163
* Try fixing https://github.com/XENONnT/alea/issues/173 by @dachengx in https://github.com/XENONnT/alea/pull/176
* Allow assigning kwargs in debug mode by @dachengx in https://github.com/XENONnT/alea/pull/174
* Allow `confidence_level` in filename by @dachengx in https://github.com/XENONnT/alea/pull/179
* Add 68% coverage as one of the defaults of `confidence_levels` by @dachengx in https://github.com/XENONnT/alea/pull/180
* Document to increase CPUs by @FaroutYLq in https://github.com/XENONnT/alea/pull/178

New Contributors
* @FaroutYLq made their first contribution in https://github.com/XENONnT/alea/pull/170

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.4...v0.2.5


0.2.4 / 2024-03-18
------------------
* Point away from alea for physics models by @kdund in https://github.com/XENONnT/alea/pull/143
* Make "piecewise" the default pdf interpolation by @hammannr in https://github.com/XENONnT/alea/pull/142
* Enforce bins in config and template to match by @hammannr in https://github.com/XENONnT/alea/pull/144
* Make model histograms accessible by @hammannr in https://github.com/XENONnT/alea/pull/140
* Make local submitter verbose by @hammannr in https://github.com/XENONnT/alea/pull/146
* Estimator of signal multiplier based on perturbation theory by @zihaoxu98 in https://github.com/XENONnT/alea/pull/147

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.3...v0.2.4


0.2.3 / 2024-02-22
------------------
* Improve check of already made toydata and output by @dachengx in https://github.com/XENONnT/alea/pull/128
* Combine several jobs into one to save computation resources by @dachengx in https://github.com/XENONnT/alea/pull/131
* Check `locate` loaded package by @dachengx in https://github.com/XENONnT/alea/pull/134
* Update `hypotheses` and `common_hypothesis` by `pre_process_poi` by @dachengx in https://github.com/XENONnT/alea/pull/135
* Print total number of submitted jobs by @dachengx in https://github.com/XENONnT/alea/pull/137

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.2...v0.2.3


0.2.2 / 2024-01-13
------------------
* Save dtype of `valid_fit` as bool by @dachengx in https://github.com/XENONnT/alea/pull/123
* Optional setting of random seed for debugging by @dachengx in https://github.com/XENONnT/alea/pull/122
* Tiny minor change on docstring by @dachengx in https://github.com/XENONnT/alea/pull/126
* Change example filename extension by @kdund in https://github.com/XENONnT/alea/pull/93
* Add axis_names to example templates by @hammannr in https://github.com/XENONnT/alea/pull/127
* Evaluate `blueice_anchors` expression by @dachengx in https://github.com/XENONnT/alea/pull/124
* Update pypi to use trusted publisher by @dachengx in https://github.com/XENONnT/alea/pull/130
* Update versions of `blueice` and `inference-interface` by @dachengx in https://github.com/XENONnT/alea/pull/132

**Full Changelog**: https://github.com/XENONnT/alea/compare/v0.2.1...v0.2.2


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
