"""Alea's IndexMorpher must end up in the registry blueice actually reads."""

import blueice.likelihood

from alea.utils import IndexMorpher


def test_index_morpher_is_registered_where_blueice_looks():
    """This is the dict LogLikelihoodBase.prepare indexes with likelihood_config's 'morpher' option,
    so registering anywhere else is a silent no-op."""
    assert blueice.likelihood.MORPHERS["IndexMorpher"] is IndexMorpher


def test_index_morpher_is_a_morpher():
    from blueice.pdf_morphers import Morpher

    assert issubclass(IndexMorpher, Morpher)
