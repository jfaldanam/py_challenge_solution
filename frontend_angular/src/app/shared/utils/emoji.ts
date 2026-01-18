import { SupportedAnimals } from "../interfaces/animals.enum";

function emojifyAnimal(species: SupportedAnimals | string) {
  if (typeof species === "string")
    species = species.toUpperCase() as SupportedAnimals;

  switch (species) {
    case SupportedAnimals.Dog:
        return "🐶 dog";
    case SupportedAnimals.Chicken:
        return "🐔 chicken";
    case SupportedAnimals.Kangaroo:
      return "🦘kangaroo";
    case SupportedAnimals.Elephant:
        return "🐘 elephant";
    case SupportedAnimals.Unknown:
      return "❓ unknown";
    default:
      throw Error("Uknown animal species");
  }
}

export { emojifyAnimal }
