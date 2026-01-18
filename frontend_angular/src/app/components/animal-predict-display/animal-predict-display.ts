import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';
import { emojifyAnimal } from '../../shared/utils/emoji';

@Component({
  selector: 'app-animal-predict-display',
  imports: [CommonModule],
  templateUrl: './animal-predict-display.html',
})
export class AnimalPredictDisplay {
  predictedAnimal = input<PredictAnimalResponse>();

  addEmojiToAnimal = emojifyAnimal

}
