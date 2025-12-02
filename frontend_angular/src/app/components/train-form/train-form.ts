import { Component, inject, output } from '@angular/core';
import { ChallengeDBService } from '../../services/challenge-db.service';


@Component({
  selector: 'app-train-form',
  imports: [],
  templateUrl: './train-form.html',
})
export class TrainForm {
  challengeDB = inject(ChallengeDBService);
  availableModels = output<string[]>();
  constructor() {
    this.refreshModels();
  }

  refreshModels() {
    this.challengeDB.getAvailableModels().subscribe(data => {
      this.availableModels.emit(data);
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
}
