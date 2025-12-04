import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { TrainForm } from './components/train-form/train-form';
import { PredictForm } from './components/predict-form/predict-form';
import { PredictAnimalResponse } from './shared/interfaces/predict-animal-response.model';
import { AnimalPredictDisplay } from './components/animal-predict-display/animal-predict-display';


@Component({
  selector: 'app-root',
  imports: [RouterOutlet, TrainForm, PredictForm, AnimalPredictDisplay],
  template: `
  <main class="text-black-600 space-y-6 p-8">
    <div class="flex flex-col md:flex-row">
      <app-train-form (availableModels)="updateAvailableModels($event)" />
      <app-predict-form [availableModels]="availableModels" (predictedAnimal)="handlePrediction($event)" />
      <app-animal-predict-display [predictedAnimal]="predictedAnimal" />
    </div>
  </main>
  <router-outlet />
`,
})
export class App {
  protected readonly title = signal('py_challenge_frontend_angular');
  availableModels: string[] = [];
  predictedAnimal: PredictAnimalResponse | undefined;

  updateAvailableModels(models: string[]) {
    this.availableModels = models;
  }
  handlePrediction(prediction: PredictAnimalResponse) {
    this.predictedAnimal = prediction;
  }
}
