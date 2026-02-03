import { Component, inject, output } from '@angular/core';
import { ChallengeDBService } from '../../services/challenge-db.service';
import { ToastState } from '../../shared/interfaces/notification-toast-state.model';


@Component({
  selector: 'app-train-form',
  imports: [],
  templateUrl: './train-form.html',
})
export class TrainForm {
  challengeDB = inject(ChallengeDBService);
  cachedAvailableModels: string[] = [];
  availableModels = output<string[]>();
  showToast = output<ToastState>();

  constructor() {
    this.refreshModels();
  }

  refreshModels() {
    this.challengeDB.getAvailableModels().subscribe({
      next: (data) => {
        this.cachedAvailableModels = data;
        this.availableModels.emit(data);
      },
      error: (err) => {
        console.error('Failed to fetch available models:', err);
        this.showToast.emit({ visible: true, message: 'Network error: Unable to fetch available models.', state: 'error' });
      }
    });
  }

  triggerTrainModel(seed: string, number_of_datapoints: string) {
    const seedNum = Number(seed);
    const datapointsNum = Number(number_of_datapoints);
    const modelId = `seed-${seedNum}-datapoints-${datapointsNum}`;
    if (this.cachedAvailableModels.includes(modelId)) {
      this.showToast.emit({ visible: true, message: 'Model already exists! Skipping training.', state: 'info' });
      return;
    }

    this.challengeDB.trainModel({seed: seedNum, number_of_datapoints: datapointsNum}).subscribe({
      next: () => {
        this.refreshModels();
        this.showToast.emit({ visible: true, message: 'Model trained successfully!', state: 'success' });
      },
      error: (err) => {
        console.error('Failed to train a model:', err);
        this.showToast.emit({ visible: true, message: 'Network error: Unable to train model.', state: 'error' });
      }
    });
  }
}
