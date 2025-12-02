import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { TrainModelRequest } from '../shared/interfaces/train-model-request.model';
import { TrainModelResponse } from '../shared/interfaces/train-model-response.model';
import { AnimalDescription } from '../shared/interfaces/predict-animal-request.model';
import { PredictAnimalResponse } from '../shared/interfaces/predict-animal-response.model';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})

export class ChallengeDBService {
  backendUrl: string = environment.backendUrl;
  http: HttpClient = inject(HttpClient);

  getAvailableModels(): Observable<string[]> {
    return this.http.get<string[]>(`${this.backendUrl}/models`);
  }

  trainModel(request: TrainModelRequest): Observable<TrainModelResponse> {
    return this.http.post<TrainModelResponse>(`${this.backendUrl}/models/train`, request);
  }

  predictAnimal(modelId: string, request: AnimalDescription[]): Observable<PredictAnimalResponse[]> {
    return this.http.post<PredictAnimalResponse[]>(`${this.backendUrl}/models/predict?model_id=${modelId}`, request);
  }
}
