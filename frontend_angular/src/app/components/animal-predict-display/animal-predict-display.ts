import { Component, input } from '@angular/core';
import { JsonPipe } from '@angular/common';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';

@Component({
  selector: 'app-animal-predict-display',
  imports: [JsonPipe],
  templateUrl: './animal-predict-display.html',
})
export class AnimalPredictDisplay {
  predictedAnimal = input<PredictAnimalResponse>();

}
