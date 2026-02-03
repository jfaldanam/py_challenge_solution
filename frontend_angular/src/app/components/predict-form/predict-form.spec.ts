import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { of, throwError } from 'rxjs';

import { PredictForm } from './predict-form';

import { ChallengeDBService } from '../../services/challenge-db.service';
import { PredictAnimalResponse } from '../../shared/interfaces/predict-animal-response.model';
import { ToastState } from '../../shared/interfaces/notification-toast-state.model';
import { AnimalDescription } from '../../shared/interfaces/predict-animal-request.model';

// Suppress console.error during tests to avoid noise from expected error handling
let consoleErrorSpy: jasmine.Spy;


describe('PredictForm', () => {
  let component: PredictForm;
  let mockChallengeDBService: jasmine.SpyObj<ChallengeDBService>;

  // Test wrapper component to provide required inputs
  @Component({
    selector: 'app-test-host',
    standalone: true,
    imports: [PredictForm],
    template: `
      <app-predict-form
        [availableModels]="availableModels()"
        (predictedAnimal)="onPredictedAnimal($event)"
        (showToast)="onShowToast($event)"
      />
    `
  })
  class TestHostComponent {
    availableModels = signal<string[]>([]);
    predictedAnimalResult: PredictAnimalResponse | null = null;
    toastState: ToastState | null = null;

    onPredictedAnimal(prediction: PredictAnimalResponse) {
      this.predictedAnimalResult = prediction;
    }

    onShowToast(toast: ToastState) {
      this.toastState = toast;
    }
  }

  let hostComponent: TestHostComponent;
  let hostFixture: ComponentFixture<TestHostComponent>;


  beforeEach(async () => {
    // Create a spy object for the ChallengeDBService
    mockChallengeDBService = jasmine.createSpyObj('ChallengeDBService', ['predictAnimal']);

    await TestBed.configureTestingModule({
      imports: [PredictForm, TestHostComponent],
      providers: [
        { provide: ChallengeDBService, useValue: mockChallengeDBService }
      ]
    }).compileComponents();

    hostFixture = TestBed.createComponent(TestHostComponent);
    hostComponent = hostFixture.componentInstance;

    // Get the PredictForm component instance
    hostFixture.detectChanges();
    const predictFormDebugElement = hostFixture.debugElement.children[0];
    component = predictFormDebugElement.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should have access to availableModels input', () => {
    hostComponent.availableModels.set(['model-1', 'model-2']);
    hostFixture.detectChanges();

    expect(component.availableModels()).toEqual(['model-1', 'model-2']);
  });

  it('should render available models in the select dropdown', () => {
    hostComponent.availableModels.set(['seed-42-datapoints-600', 'seed-123-datapoints-1000']);
    hostFixture.detectChanges();

    const compiled = hostFixture.nativeElement as HTMLElement;
    const options = compiled.querySelectorAll('select option');

    expect(options.length).toBe(2);
    expect(options[0].textContent).toContain('seed-42-datapoints-600');
    expect(options[1].textContent).toContain('seed-123-datapoints-1000');
  });

  it('should show "No models available" when availableModels is empty', () => {
    hostComponent.availableModels.set([]);
    hostFixture.detectChanges();

    const compiled = hostFixture.nativeElement as HTMLElement;
    const options = compiled.querySelectorAll('select option');

    expect(options.length).toBe(1);
    expect(options[0].textContent).toContain('No models available');
    expect(options[0].hasAttribute('disabled')).toBeTrue();
  });

  it('should emit info toast when model is empty string', fakeAsync(() => {
    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('', animalDescription);
    tick();

    expect(hostComponent.toastState).toEqual({
      visible: true,
      message: 'Please select a valid model before running prediction.',
      state: 'info'
    });
    expect(mockChallengeDBService.predictAnimal).not.toHaveBeenCalled();
  }));

  it('should emit info toast when model is whitespace only', fakeAsync(() => {
    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('   ', animalDescription);
    tick();

    expect(hostComponent.toastState).toEqual({
      visible: true,
      message: 'Please select a valid model before running prediction.',
      state: 'info'
    });
    expect(mockChallengeDBService.predictAnimal).not.toHaveBeenCalled();
  }));

  it('should call predictAnimal service with correct parameters', fakeAsync(() => {
    const mockResponse: PredictAnimalResponse[] = [{
      species: 'DOG',
      probabilities: { 'DOG': 0.9, 'CHICKEN': 0.05, 'ELEPHANT': 0.03, 'KANGAROO': 0.02 }
    }];
    mockChallengeDBService.predictAnimal.and.returnValue(of(mockResponse));

    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('model-1', animalDescription);
    tick();

    expect(mockChallengeDBService.predictAnimal).toHaveBeenCalledWith('model-1', [animalDescription]);
  }));

  it('should emit predictedAnimal on successful prediction', fakeAsync(() => {
    const mockResponse: PredictAnimalResponse[] = [{
      species: 'DOG',
      probabilities: { 'DOG': 0.9, 'CHICKEN': 0.05, 'ELEPHANT': 0.03, 'KANGAROO': 0.02 }
    }];
    mockChallengeDBService.predictAnimal.and.returnValue(of(mockResponse));

    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('model-1', animalDescription);
    tick();

    expect(hostComponent.predictedAnimalResult).toEqual(mockResponse[0]);
  }));

  it('should emit success toast on successful prediction', fakeAsync(() => {
    const mockResponse: PredictAnimalResponse[] = [{
      species: 'DOG',
      probabilities: { 'DOG': 0.9, 'CHICKEN': 0.05, 'ELEPHANT': 0.03, 'KANGAROO': 0.02 }
    }];
    mockChallengeDBService.predictAnimal.and.returnValue(of(mockResponse));

    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('model-1', animalDescription);
    tick();

    expect(hostComponent.toastState?.state).toBe('success');
    expect(hostComponent.toastState?.visible).toBeTrue();
    expect(hostComponent.toastState?.message).toContain('Prediction succesful');
    expect(hostComponent.toastState?.message).toContain('🐶 dog');
  }));

  it('should emit error toast on prediction failure', fakeAsync(() => {
    mockChallengeDBService.predictAnimal.and.returnValue(throwError(() => new Error('Network error')));
    // Mock console.error to suppress output during test
    consoleErrorSpy = spyOn(console, 'error');

    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 4,
      height: 0.5,
      weight: 10,
      has_wings: false,
      has_tail: true
    };

    component.triggerPredictAnimal('model-1', animalDescription);
    tick();

    expect(hostComponent.toastState).toEqual({
      visible: true,
      message: 'Network error: Unable to run inference on provided data.',
      state: 'error'
    });

    // Restore console.error
    consoleErrorSpy.and.callThrough();
  }));

  it('should handle unknown species gracefully', fakeAsync(() => {
    const mockResponse: PredictAnimalResponse[] = [{
      species: 'UNKNOWN',
      probabilities: { 'DOG': 0.25, 'CHICKEN': 0.25, 'ELEPHANT': 0.25, 'KANGAROO': 0.25 }
    }];
    mockChallengeDBService.predictAnimal.and.returnValue(of(mockResponse));

    const animalDescription: AnimalDescription = {
      walks_on_n_legs: 8,
      height: 0.1,
      weight: 0.5,
      has_wings: false,
      has_tail: false
    };

    component.triggerPredictAnimal('model-1', animalDescription);
    tick();

    expect(hostComponent.predictedAnimalResult?.species).toBe('UNKNOWN');
    expect(hostComponent.toastState?.message).toContain('❓ unknown');
  }));

  it('should expose Number function for template', () => {
    expect(component.Number).toBe(Number);
    expect(component.Number('42')).toBe(42);
    expect(component.Number('3.14')).toBeCloseTo(3.14);
  });

  it('should render all form inputs', () => {
    const compiled = hostFixture.nativeElement as HTMLElement;

    expect(compiled.querySelector('#models')).toBeTruthy();
    expect(compiled.querySelector('#animalLegs')).toBeTruthy();
    expect(compiled.querySelector('#animalHeight')).toBeTruthy();
    expect(compiled.querySelector('#animalWeight')).toBeTruthy();
    expect(compiled.querySelector('#animalWings')).toBeTruthy();
    expect(compiled.querySelector('#animalTail')).toBeTruthy();
  });

  it('should render submit button', () => {
    const compiled = hostFixture.nativeElement as HTMLElement;
    const button = compiled.querySelector('button[type="submit"]');

    expect(button).toBeTruthy();
    expect(button?.textContent).toContain('Predict Animal');
  });

  it('should render form title', () => {
    const compiled = hostFixture.nativeElement as HTMLElement;
    const title = compiled.querySelector('h2');

    expect(title).toBeTruthy();
    expect(title?.textContent).toContain('Send animal to predict');
  });
});
