export interface PredictAnimalResponse {
  species: string;
  probabilities: Record<string, number>;
}