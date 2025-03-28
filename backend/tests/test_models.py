import pytest
from pydantic import ValidationError

from py_challenge_backend.models import (
    Animal,
    AnimalCharacteristics,
    AnimalSpecies,
    ModelMetrics,
    ModelPrediction,
    TrainedModelResponse,
    TrainInput,
)


def test_train_input_defaults():
    model = TrainInput()
    assert model.seed == 42
    assert model.number_of_datapoints == 500


def test_train_input_validation():
    with pytest.raises(ValidationError):
        TrainInput(number_of_datapoints=-10)  # Invalid value for number_of_datapoints


def test_animal_characteristics_validation():
    valid_data = {
        "walks_on_n_legs": 4,
        "height": 1.2,
        "weight": 35.0,
        "has_wings": False,
        "has_tail": True,
    }
    model = AnimalCharacteristics(**valid_data)
    assert model.walks_on_n_legs == 4
    assert model.height == 1.2
    assert model.weight == 35.0
    assert not model.has_wings
    assert model.has_tail

    with pytest.raises(ValidationError):
        AnimalCharacteristics(walks_on_n_legs=-1)  # Invalid value for legs


def test_animal_model():
    characteristics = AnimalCharacteristics(
        walks_on_n_legs=4, height=1.2, weight=35.0, has_wings=False, has_tail=True
    )
    animal = Animal(species=AnimalSpecies.DOG, characteristics=characteristics)
    assert animal.species == AnimalSpecies.DOG
    assert animal.characteristics == characteristics


def test_model_metrics_validation():
    valid_data = {
        "accuracy": 0.95,
        "precision": 0.9,
        "recall": 0.85,
        "f1": 0.88,
        "confusion": [[50, 2], [1, 47]],
    }
    metrics = ModelMetrics(**valid_data)
    assert metrics.accuracy == 0.95
    assert metrics.confusion == [[50, 2], [1, 47]]

    with pytest.raises(ValidationError):
        ModelMetrics(accuracy=1.5)  # Invalid accuracy value


def test_trained_model_response():
    metrics = ModelMetrics(
        accuracy=0.95, precision=0.9, recall=0.85, f1=0.88, confusion=[[50, 2], [1, 47]]
    )
    response = TrainedModelResponse(model_id="model_123", metrics=metrics)
    assert response.model_id == "model_123"
    assert response.metrics == metrics


def test_model_prediction():
    probabilities = {
        AnimalSpecies.KANGAROO: 0.1,
        AnimalSpecies.ELEPHANT: 0.2,
        AnimalSpecies.CHICKEN: 0.3,
        AnimalSpecies.DOG: 0.4,
        AnimalSpecies.UNKNOWN: 0.0,
    }
    prediction = ModelPrediction(species=AnimalSpecies.DOG, probabilities=probabilities)
    assert prediction.species == AnimalSpecies.DOG
    assert prediction.probabilities[AnimalSpecies.DOG] == 0.4

    with pytest.raises(ValidationError):
        ModelPrediction(probabilities={"invalid_species": 0.5})  # Invalid key
