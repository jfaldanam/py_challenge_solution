import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component } from '@angular/core';
import { By } from '@angular/platform-browser';

import { AnimalPredictDisplay } from './animal-predict-display';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';

// Test host component to pass inputs to AnimalPredictDisplay
@Component({
  template: `<app-animal-predict-display [predictedAnimal]="animal" />`,
  imports: [AnimalPredictDisplay],
})
class TestHostComponent {
  animal: PredictAnimalResponse | undefined;
}

describe('AnimalPredictDisplay', () => {
  let hostComponent: TestHostComponent;
  let fixture: ComponentFixture<TestHostComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent, AnimalPredictDisplay],
    }).compileComponents();

    fixture = TestBed.createComponent(TestHostComponent);
    hostComponent = fixture.componentInstance;
  });

  it('should create', () => {
    fixture.detectChanges();
    const displayElement = fixture.debugElement.query(By.directive(AnimalPredictDisplay));
    expect(displayElement).toBeTruthy();
  });

  it('should not render anything when predictedAnimal is undefined', () => {
    hostComponent.animal = undefined;
    fixture.detectChanges();

    const heading = fixture.nativeElement.querySelector('h1');
    expect(heading).toBeNull();
  });

  it('should render species with emoji when predictedAnimal is provided', () => {
    hostComponent.animal = {
      species: 'DOG',
      probabilities: { DOG: 0.9, CHICKEN: 0.05, ELEPHANT: 0.03, KANGAROO: 0.02 },
    };
    fixture.detectChanges();

    const heading = fixture.nativeElement.querySelector('h1');
    expect(heading).toBeTruthy();
    expect(heading.textContent).toContain('🐶 dog');
  });

  it('should display all probabilities in a table', () => {
    hostComponent.animal = {
      species: 'ELEPHANT',
      probabilities: { DOG: 0.1, CHICKEN: 0.1, ELEPHANT: 0.7, KANGAROO: 0.1 },
    };
    fixture.detectChanges();

    const tableRows = fixture.nativeElement.querySelectorAll('tbody tr');
    expect(tableRows.length).toBe(4);
  });

  it('should display probabilities as percentages', () => {
    hostComponent.animal = {
      species: 'CHICKEN',
      probabilities: { CHICKEN: 0.85 },
    };
    fixture.detectChanges();

    const probabilityCell = fixture.nativeElement.querySelector('tbody td:last-child');
    expect(probabilityCell.textContent).toContain('85%');
  });

  it('should render emojified animal names in the probability table', () => {
    hostComponent.animal = {
      species: 'KANGAROO',
      probabilities: { KANGAROO: 0.95 },
    };
    fixture.detectChanges();

    const speciesCell = fixture.nativeElement.querySelector('tbody td:first-child');
    expect(speciesCell.textContent).toContain('🦘');
  });

  it('should have correct table headers', () => {
    hostComponent.animal = {
      species: 'DOG',
      probabilities: { DOG: 1 },
    };
    fixture.detectChanges();

    const headers = fixture.nativeElement.querySelectorAll('th');
    expect(headers.length).toBe(2);
    expect(headers[0].textContent).toContain('Species');
    expect(headers[1].textContent).toContain('Probability');
  });

  it('should have a caption explaining the table', () => {
    hostComponent.animal = {
      species: 'DOG',
      probabilities: { DOG: 1 },
    };
    fixture.detectChanges();

    const caption = fixture.nativeElement.querySelector('caption');
    expect(caption).toBeTruthy();
    expect(caption.textContent).toContain('Probabilities');
  });

  it('should handle UNKNOWN species', () => {
    hostComponent.animal = {
      species: 'UNKNOWN',
      probabilities: { UNKNOWN: 0.5, DOG: 0.25, CHICKEN: 0.25 },
    };
    fixture.detectChanges();

    const heading = fixture.nativeElement.querySelector('h1');
    expect(heading.textContent).toContain('❓ unknown');
  });

  it('should update display when predictedAnimal changes', () => {
    hostComponent.animal = {
      species: 'DOG',
      probabilities: { DOG: 0.9 },
    };
    fixture.detectChanges();

    let heading = fixture.nativeElement.querySelector('h1');
    expect(heading.textContent).toContain('🐶 dog');

    // Change the animal
    hostComponent.animal = {
      species: 'ELEPHANT',
      probabilities: { ELEPHANT: 0.8 },
    };
    fixture.detectChanges();

    heading = fixture.nativeElement.querySelector('h1');
    expect(heading.textContent).toContain('🐘 elephant');
  });

  it('should handle lowercase species names', () => {
    hostComponent.animal = {
      species: 'chicken',
      probabilities: { chicken: 0.9 },
    };
    fixture.detectChanges();

    const heading = fixture.nativeElement.querySelector('h1');
    expect(heading.textContent).toContain('🐔 chicken');
  });
});