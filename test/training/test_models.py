import pytest
from src.training.create_models import get_models_for_experiment


def test_get_models_for_experiment():
    for model, _ in get_models_for_experiment():
        assert model is not None
        model_config = (
            model.get_config()
        )  # Returns pretty much every information about your model
        batch_input_shape = model_config["layers"][0]["config"]["batch_input_shape"]
        assert batch_input_shape == (None, 32, 32, 3)

        output_shape = model_config["layers"][-1]["config"]["units"]
        assert output_shape == 10
