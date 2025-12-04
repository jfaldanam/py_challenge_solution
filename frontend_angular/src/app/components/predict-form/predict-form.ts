import { Component, inject, input, output } from '@angular/core';
import { ChallengeDBService } from '../../services/challenge-db.service';
import { AnimalDescription } from '../../shared/interfaces/predict-animal-request.model';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';

@Component({
  selector: 'app-predict-form',
  imports: [],
  templateUrl: './predict-form.html',
})
export class PredictForm {
  availableModels = input.required<string[]>();
  predictedAnimal = output<PredictAnimalResponse>();
  challengeDB = inject(ChallengeDBService);


  triggerPredictAnimal(modelId: string, animalDescription: AnimalDescription) {
    this.challengeDB.predictAnimal(modelId, [
        animalDescription
      ]).subscribe(response => {
        const predictedResponse = response[0]; // As we only sent one animal, we expect only one response
        this.predictedAnimal.emit(predictedResponse);
    });
  }

  // Allow template to access global Number function
  get Number() {  return Number; }
}
