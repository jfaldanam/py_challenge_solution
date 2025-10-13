import { Component, signal, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ChallengeDBService } from './services/challenge-db.service';
import { AnimalDescription } from './interfaces/predictanimalrequest';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  template: `
  <main class="text-black-600 space-y-6 p-8">
    <div class="flex flex-col md:flex-row">
      <div class="center px-4">
        <form (submit)="triggerTrainModel(seedInput.value, datapointsInput.value); $event.preventDefault();">
          <h2 class="text-lg font-semibold mb-4">Train a New Model</h2>
          <label for="seed" class="block mb-2 font-medium">Seed:</label>
          <input #seedInput type="number" id="seed" name="seed" value="42" class="border border-gray-300 rounded p-2 mb-4" />
          <label for="datapoints" class="block mb-2 font-medium">Number of datapoints:</label>
          <input #datapointsInput type="number" id="datapoints" name="datapoints" value="600" class="border border-gray-300 rounded p-2 mb-4" />
          <label for="trainModel" class="block mb-2 font-medium">Train a new model:</label>
          <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Train Model
          </button>
        </form>
      </div>
      <div class="center px-4">
        <form (submit)="triggerPredictAnimal(selectedModelInput.value, { walks_on_n_legs: Number(animalLegs.value), height: Number(animalHeight.value), weight: Number(animalWeight.value), has_wings: animalWings.checked, has_tail: animalTail.checked }); $event.preventDefault();">
          <h2 class="text-lg font-semibold mb-4">Send animal to predict</h2>
          <label for="models">Available models:</label><br>
          <select #selectedModelInput class="border border-gray-300 rounded p-2 " name="models" id="models">
            @for(model of availableModels; track model) {
              <option value="{{model}}">{{model}}</option>
            }
          </select><br>
          <label for="animalLegs" class="block mb-2 font-medium">Number of legs:</label>
          <input #animalLegs type="number" id="animalLegs" name="animalLegs" value="4" class="border border-gray-300 rounded p-2 mb-4" /><br>
          <label for="animalHeight" class="block mb-2 font-medium">Height (m):</label>
          <input #animalHeight type="number" id="animalHeight" name="animalHeight" value="0.3" class="border border-gray-300 rounded p-2 mb-4" /><br>
          <label for="animalWeight" class="block mb-2 font-medium">Weight (kg):</label>
          <input #animalWeight type="number" id="animalWeight" name="animalWeight" value="10" class="border border-gray-300 rounded p-2 mb-4" /><br>
          <input #animalWings type="checkbox" id="animalWings" name="animalWings" class="mb-4 px-2" /> <label for="animalWings" class="mr-4">Has wings</label>
          <input #animalTail type="checkbox" id="animalTail" name="animalTail" checked class="mb-4 px-2" /> <label for="animalTail">Has tail</label>
          <label for="predictAnimal" class="block mb-2 font-medium">Predict animal with selected model:</label>
          <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Predict Animal
          </button>
        </form>
      </div>
    </div>
  </main>
  <router-outlet />
`,
})
export class App {
  protected readonly title = signal('py_challenge_frontend_angular');
  availableModels: string[] = [];
  challengeDB = inject(ChallengeDBService);

  constructor() {
    this.refreshModels();
  }

  refreshModels() {
    this.challengeDB.getAvailableModels().subscribe(data => {
      this.availableModels = data;
    });
  }
  
  triggerTrainModel(seed: string, number_of_datapoints: string) {
    const seedNum = Number(seed);
    const datapointsNum = Number(number_of_datapoints);
    this.challengeDB.trainModel({seed: seedNum, number_of_datapoints: datapointsNum}).subscribe(() => {
      this.refreshModels();
      alert('Model trained successfully!');
    });
  }

  triggerPredictAnimal(modelId: string, animalDescription: AnimalDescription) {
    this.challengeDB.predictAnimal(modelId, [
        animalDescription
      ]).subscribe(response => {
        const predictedResponse = response[0]; // As we only sent one animal, we expect only one response
      alert(`Predicted species: ${predictedResponse.species}\nProbabilities: ${JSON.stringify(predictedResponse.probabilities)}`);
    });
  }

  // Allow template to access global Number function
  get Number() {  return Number; }
}
