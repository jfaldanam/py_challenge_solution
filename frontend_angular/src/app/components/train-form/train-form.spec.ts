import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { TrainForm } from './train-form';
import { ChallengeDBService } from '../../services/challenge-db.service';
import { TrainModelResponse } from '../../shared/interfaces/train-model-response.model';

// Suppress console.error during tests to avoid noise from expected error handling
let consoleErrorSpy: jasmine.Spy;

// Helper function to create a mock TrainModelResponse
function createMockTrainResponse(modelId: string): TrainModelResponse {
  return {
    model_id: modelId,
    metrics: {
      accuracy: 0.95,
      precision: 0.94,
      recall: 0.93,
      f1: 0.935,
      confusion: [[45, 5], [3, 47]]
    }
  };
}

describe('TrainForm', () => {
  let component: TrainForm;
  let fixture: ComponentFixture<TrainForm>;
  let mockChallengeDBService: jasmine.SpyObj<ChallengeDBService>;

  beforeEach(async () => {
    // Create a mock of the ChallengeDBService with the methods we need
    mockChallengeDBService = jasmine.createSpyObj('ChallengeDBService', [
      'getAvailableModels',
      'trainModel'
    ]);

    // Default mock implementation - return empty array for models
    mockChallengeDBService.getAvailableModels.and.returnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [TrainForm],
      providers: [
        { provide: ChallengeDBService, useValue: mockChallengeDBService }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(TrainForm);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('initialization', () => {
    it('should fetch available models on construction', () => {
      expect(mockChallengeDBService.getAvailableModels).toHaveBeenCalled();
    });

    it('should emit available models when fetched successfully', () => {
      const testModels = ['model-1', 'model-2'];
      mockChallengeDBService.getAvailableModels.and.returnValue(of(testModels));

      spyOn(component.availableModels, 'emit');

      component.refreshModels();

      expect(component.availableModels.emit).toHaveBeenCalledWith(testModels);
    });

    it('should cache available models locally', () => {
      const testModels = ['seed-42-datapoints-600', 'seed-123-datapoints-1000'];
      mockChallengeDBService.getAvailableModels.and.returnValue(of(testModels));

      component.refreshModels();

      expect(component.cachedAvailableModels).toEqual(testModels);
    });

    it('should emit error toast when fetching models fails', () => {
      mockChallengeDBService.getAvailableModels.and.returnValue(
        throwError(() => new Error('Network error'))
      );
      // Mock console.error to suppress output during test
      consoleErrorSpy = spyOn(console, 'error');
      spyOn(component.showToast, 'emit');

      component.refreshModels();

      expect(component.showToast.emit).toHaveBeenCalledWith({
        visible: true,
        message: 'Network error: Unable to fetch available models.',
        state: 'error'
      });
      // Restore console.error
      consoleErrorSpy.and.callThrough();
    });
  });

  describe('triggerTrainModel', () => {
    beforeEach(() => {
      // Reset to empty cached models for training tests
      component.cachedAvailableModels = [];
    });

    it('should call trainModel service with correct parameters', () => {
      mockChallengeDBService.trainModel.and.returnValue(of(createMockTrainResponse('seed-42-datapoints-600')));
      mockChallengeDBService.getAvailableModels.and.returnValue(of([]));

      component.triggerTrainModel('42', '600');

      expect(mockChallengeDBService.trainModel).toHaveBeenCalledWith({
        seed: 42,
        number_of_datapoints: 600
      });
    });

    it('should emit success toast when training completes', fakeAsync(() => {
      mockChallengeDBService.trainModel.and.returnValue(of(createMockTrainResponse('seed-42-datapoints-600')));
      mockChallengeDBService.getAvailableModels.and.returnValue(of(['seed-42-datapoints-600']));

      spyOn(component.showToast, 'emit');

      component.triggerTrainModel('42', '600');
      tick();

      expect(component.showToast.emit).toHaveBeenCalledWith({
        visible: true,
        message: 'Model trained successfully!',
        state: 'success'
      });
    }));

    it('should refresh models after successful training', fakeAsync(() => {
      mockChallengeDBService.trainModel.and.returnValue(of(createMockTrainResponse('seed-42-datapoints-600')));
      mockChallengeDBService.getAvailableModels.and.returnValue(of(['seed-42-datapoints-600']));

      // Reset call count after initial constructor call
      mockChallengeDBService.getAvailableModels.calls.reset();

      component.triggerTrainModel('42', '600');
      tick();

      expect(mockChallengeDBService.getAvailableModels).toHaveBeenCalled();
    }));

    it('should emit error toast when training fails', () => {
      mockChallengeDBService.trainModel.and.returnValue(
        throwError(() => new Error('Training failed'))
      );
      // Mock console.error to suppress output during test
      consoleErrorSpy = spyOn(console, 'error');
      spyOn(component.showToast, 'emit');

      component.triggerTrainModel('42', '600');

      expect(component.showToast.emit).toHaveBeenCalledWith({
        visible: true,
        message: 'Network error: Unable to train model.',
        state: 'error'
      });

      // Restore console.error
      consoleErrorSpy.and.callThrough();
    });

    it('should skip training if model already exists', () => {
      component.cachedAvailableModels = ['seed-42-datapoints-600'];

      spyOn(component.showToast, 'emit');

      component.triggerTrainModel('42', '600');

      expect(mockChallengeDBService.trainModel).not.toHaveBeenCalled();
      expect(component.showToast.emit).toHaveBeenCalledWith({
        visible: true,
        message: 'Model already exists! Skipping training.',
        state: 'info'
      });
    });

    it('should convert string parameters to numbers correctly', () => {
      mockChallengeDBService.trainModel.and.returnValue(of(createMockTrainResponse('seed-123-datapoints-1000')));
      mockChallengeDBService.getAvailableModels.and.returnValue(of([]));

      component.triggerTrainModel('123', '1000');

      expect(mockChallengeDBService.trainModel).toHaveBeenCalledWith({
        seed: 123,
        number_of_datapoints: 1000
      });
    });
  });

  describe('template rendering', () => {
    it('should have a form with seed input', () => {
      const compiled = fixture.nativeElement as HTMLElement;
      const seedInput = compiled.querySelector('input#seed');
      expect(seedInput).toBeTruthy();
      expect(seedInput?.getAttribute('type')).toBe('number');
    });

    it('should have a form with datapoints input', () => {
      const compiled = fixture.nativeElement as HTMLElement;
      const datapointsInput = compiled.querySelector('input#datapoints');
      expect(datapointsInput).toBeTruthy();
      expect(datapointsInput?.getAttribute('type')).toBe('number');
    });

    it('should have a submit button', () => {
      const compiled = fixture.nativeElement as HTMLElement;
      const button = compiled.querySelector('button[type="submit"]');
      expect(button).toBeTruthy();
      expect(button?.textContent?.trim()).toContain('Train Model');
    });

    it('should display the form heading', () => {
      const compiled = fixture.nativeElement as HTMLElement;
      const heading = compiled.querySelector('h2');
      expect(heading?.textContent).toContain('Train a New Model');
    });
  });
});
