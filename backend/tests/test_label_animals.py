import pytest

from py_challenge_backend.ml.train import label_animals
from py_challenge_backend.models import Animal, AnimalCharacteristics


@pytest.fixture
def animal_characteristics():
    """Mock animal characteristics data."""
    return [
        AnimalCharacteristics(
            walks_on_n_legs=5, height=80, has_wings=False, weight=20, has_tail=False
        )
    ] + [
        AnimalCharacteristics(
            walks_on_n_legs=2, height=0.3, has_wings=True, weight=2.5, has_tail=True
        ),
        AnimalCharacteristics(
            walks_on_n_legs=2, height=1.5, has_wings=False, weight=70, has_tail=True
        ),
        AnimalCharacteristics(
            walks_on_n_legs=4, height=3.0, has_wings=False, weight=500, has_tail=True
        ),
        AnimalCharacteristics(
            walks_on_n_legs=4, height=0.5, has_wings=False, weight=20, has_tail=True
        ),
    ] * 100


def test_label_animals(animal_characteristics):
    """Test the label_animals function."""
    labelled_animals = label_animals(animal_characteristics)

    assert isinstance(labelled_animals, list)
    assert all(isinstance(animal, Animal) for animal in labelled_animals)
    assert len(labelled_animals) == len(animal_characteristics)
