import { Component, inject, input, output } from '@angular/core';
import { ChallengeDBService } from '../../services/challenge-db.service';
import { AnimalDescription } from '../../shared/interfaces/predict-animal-request.model';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';
import { ToastState } from '../../shared/interfaces/notification-toast-state.model';

@Component({
  selector: 'app-predict-form',
  imports: [],
  templateUrl: './predict-form.html',
})
export class PredictForm {
  challengeDB = inject(ChallengeDBService);
  availableModels = input.required<string[]>();
  predictedAnimal = output<PredictAnimalResponse>();
  showToast = output<ToastState>();

  triggerPredictAnimal(modelId: string, animalDescription: AnimalDescription) {
    this.challengeDB.predictAnimal(modelId, [
        animalDescription
      ]).subscribe(response => {
        const predictedResponse = response[0]; // As we only sent one animal, we expect only one response
        this.predictedAnimal.emit(predictedResponse);
        let toastSpeciesStr = predictedResponse.species;
        try {
          toastSpeciesStr = this.emojifyAnimal(predictedResponse.species);
        } catch {
          // Keep str only value
        }
        this.showToast.emit({ visible: true, message: `Prediction succesful, animal is classified as a ${toastSpeciesStr}` })
    });
  }

  emojifyAnimal(species: string) {
    console.log(species)
    switch (species) {
      case "dog":
        return "🐶 dog";
      case "chicken":
        return "🐔 chicken";
      case "kangaroo":
        return "🦘kangaroo";
      case "elephant":
        return "🐘 elephant";
      case "unknown":
        return "❓ unknown";
      default:
        throw Error("Uknown animal species");
    }
  }

  // Allow template to access global Number function
  get Number() {  return Number; }
}
