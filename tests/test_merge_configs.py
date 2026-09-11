from alea.utils import load_yaml, merge_statistical_model_configs
from alea.models import BlueiceExtendedModel


def test_merge_two_example_configs():
    a = load_yaml("unbinned_wimp_statistical_model_simple.yaml")
    b = load_yaml("unbinned_wimp_statistical_model.yaml")
    merged = merge_statistical_model_configs(a, b)
    na = len(a["likelihood_config"]["likelihood_terms"])
    nb = len(b["likelihood_config"]["likelihood_terms"])
    assert len(merged["likelihood_config"]["likelihood_terms"]) == na + nb
    # ensure we can initialize a model from the merged config
    model = BlueiceExtendedModel(
        parameter_definition=merged["parameter_definition"],
        likelihood_config=merged["likelihood_config"],
    )
    assert len(model.likelihood_list()) == na + nb
