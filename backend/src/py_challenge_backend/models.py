from enum import Enum

from pydantic import BaseModel, Field


class TrainInput(BaseModel):
    seed: int = Field(
        default=42,
        title="Seed",
        description="The seed to use for the random number generator",
    )
    number_of_datapoints: int = Field(
        default=500,
        title="Number of datapoints",
        description="The number of datapoints to generate",
        gt=0,
    )


class AnimalSpecies(str, Enum):
    KANGAROO = "kangaroo"
    ELEPHANT = "elephant"
    CHICKEN = "chicken"
    DOG = "dog"
    UNKNOWN = "unknown"


class AnimalCharacteristics(BaseModel):
    walks_on_n_legs: int = Field(
        title="Walks on 'n' legs",
        description="The number of legs the animal walks on",
    )
    height: float = Field(
        title="Height",
        description="The height of the animal in meters",
    )
    weight: float = Field(
        title="Weight", description="The weight of the animal in kilograms"
    )
    has_wings: bool = Field(
        title="Has wings?", description="Whether the animal has wings"
    )
    has_tail: bool = Field(
        title="Has tail?", description="Whether the animal has a tail"
    )


class Animal(BaseModel):
    species: AnimalSpecies = Field(
        title="Species",
        description="The species of the animal",
    )
    characteristics: AnimalCharacteristics = Field(
        title="Characteristics",
        description="The characteristics of the animal",
    )


class ModelMetrics(BaseModel):
    accuracy: float = Field(
        title="Accuracy",
        description="The accuracy of the model",
        gt=0,
        le=1,
    )
    precision: float = Field(
        title="Precision",
        description="The precision of the model",
        gt=0,
        le=1,
    )
    recall: float = Field(
        title="Recall",
        description="The recall of the model",
        gt=0,
        le=1,
    )
    f1: float = Field(
        title="F1",
        description="The F1 score of the model",
        gt=0,
        le=1,
    )
    confusion: list[list[int]] = Field(
        title="Confusion matrix",
        description="The confusion matrix of the model",
    )


class TrainedModelResponse(BaseModel):
    model_id: str = Field(
        title="Model ID",
        description="The ID of the trained model",
    )
    metrics: ModelMetrics = Field(
        title="Model metrics",
        description="The metrics of the trained model",
    )


class ModelPrediction(BaseModel):
    species: AnimalSpecies = Field(
        title="Species",
        description="The predicted species",
    )
    probabilities: dict[AnimalSpecies, float] = Field(
        title="Probabilities",
        description="The probabilities of each species",
    )
