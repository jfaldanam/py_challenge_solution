export interface TrainModelResponse {
  model_id: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    confusion: number[][];
  };
}