:orphan:

Runners
=======

Data Format
===========
alea uses the
`inference_interface <https://github.com/XENONnT/inference_interface>`_
to handle storing fit results, toy data and to load _templates_
-- histograms that define a PDF for e.g. the blueice_extended_model.
inference_interface uses the pyhdf format, with specific fields for:
* fit results: uses numpy_to_toyfile and toyfile_to_numpy to write
and load best-fit values, upper/lower limits and other values from toy fits.
* toy data: uses toydata_to_file and toydata_from_file to
write and load toy data-- alea uses this format to read/write data.
* templates: uses template_to_multihist and
multihist_to_template, alternatively numpy_to_template,
template_to_numpy (root version also exists but is not uses)

To emphasize that we encourage this standard format for all
alea
read/writing,
we use the file
ending .ii.h5 for the files in these formats.
