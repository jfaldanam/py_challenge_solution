import { ComponentFixture, TestBed } from '@angular/core/testing';
import { App } from './app';
import { PredictAnimalResponse } from './shared/interfaces/predict-animal-response.model';
import { ToastState } from './shared/interfaces/notification-toast-state.model';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { By } from '@angular/platform-browser';
import { TrainForm } from './components/train-form/train-form';
import { PredictForm } from './components/predict-form/predict-form';
import { NotificationToast } from './components/notification-toast/notification-toast';

describe('App', () => {
  let component: App;
  let fixture: ComponentFixture<App>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(App);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create the app', () => {
    expect(component).toBeTruthy();
  });

  describe('Model Management Flow', () => {
    it('should pass available models to predict-form when train-form emits them', () => {
      const mockModels = ['model-1', 'model-2', 'model-3'];
      const trainForm = fixture.debugElement.query(By.directive(TrainForm));

      trainForm.triggerEventHandler('availableModels', mockModels);
      fixture.detectChanges();

      const predictForm = fixture.debugElement.query(By.directive(PredictForm));
      expect(predictForm.componentInstance.availableModels()).toEqual(mockModels);
    });

    it('should update predict-form when models list changes', () => {
      const initialModels = ['old-model'];
      const newModels = ['new-model-1', 'new-model-2'];
      const trainForm = fixture.debugElement.query(By.directive(TrainForm));

      trainForm.triggerEventHandler('availableModels', initialModels);
      fixture.detectChanges();

      trainForm.triggerEventHandler('availableModels', newModels);
      fixture.detectChanges();

      const predictForm = fixture.debugElement.query(By.directive(PredictForm));
      expect(predictForm.componentInstance.availableModels()).toEqual(newModels);
    });
  });

  describe('Prediction Flow', () => {
    it('should display prediction results when predict-form emits a prediction', () => {
      const mockPrediction: PredictAnimalResponse = {
        species: 'DOG',
        probabilities: { DOG: 0.9, CHICKEN: 0.05, ELEPHANT: 0.03, KANGAROO: 0.02 }
      };
      const predictForm = fixture.debugElement.query(By.directive(PredictForm));

      predictForm.triggerEventHandler('predictedAnimal', mockPrediction);
      fixture.detectChanges();

      expect(component.predictedAnimal).toEqual(mockPrediction);
    });

    it('should update display when a new prediction replaces an existing one', () => {
      const firstPrediction: PredictAnimalResponse = {
        species: 'CHICKEN',
        probabilities: { CHICKEN: 0.8 }
      };
      const secondPrediction: PredictAnimalResponse = {
        species: 'ELEPHANT',
        probabilities: { ELEPHANT: 0.95 }
      };
      const predictForm = fixture.debugElement.query(By.directive(PredictForm));

      predictForm.triggerEventHandler('predictedAnimal', firstPrediction);
      fixture.detectChanges();

      predictForm.triggerEventHandler('predictedAnimal', secondPrediction);
      fixture.detectChanges();

      expect(component.predictedAnimal?.species).toBe('ELEPHANT');
    });
  });

  describe('Toast Notification Flow', () => {
    it('should show toast when train-form emits a toast event', () => {
      const toastState: ToastState = {
        visible: true,
        message: 'Model trained successfully!',
        state: 'success'
      };
      const trainForm = fixture.debugElement.query(By.directive(TrainForm));

      trainForm.triggerEventHandler('showToast', toastState);
      fixture.detectChanges();

      const toast = fixture.debugElement.query(By.directive(NotificationToast));
      expect(toast.componentInstance.state().visible).toBeTrue();
      expect(toast.componentInstance.state().message).toBe('Model trained successfully!');
    });

    it('should show toast when predict-form emits a toast event', () => {
      const toastState: ToastState = {
        visible: true,
        message: 'Prediction successful!',
        state: 'success'
      };
      const predictForm = fixture.debugElement.query(By.directive(PredictForm));

      predictForm.triggerEventHandler('showToast', toastState);
      fixture.detectChanges();

      const toast = fixture.debugElement.query(By.directive(NotificationToast));
      expect(toast.componentInstance.state().visible).toBeTrue();
      expect(toast.componentInstance.state().message).toBe('Prediction successful!');
    });

    it('should clear toast when notification-toast emits clearToast event', () => {
      // First show a toast
      component.toastState = { visible: true, message: 'Test message', state: 'info' };
      fixture.detectChanges();

      const toast = fixture.debugElement.query(By.directive(NotificationToast));
      const clearState: ToastState = { visible: false, message: '', state: 'info' };

      toast.triggerEventHandler('clearToast', clearState);
      fixture.detectChanges();

      expect(component.toastState.visible).toBeFalse();
    });

    it('should not update toast when undefined is passed', () => {
      const originalState: ToastState = {
        visible: true,
        message: 'Original message',
        state: 'info'
      };
      component.toastState = { ...originalState };

      component.updateToast(undefined);

      expect(component.toastState).toEqual(originalState);
    });

    it('should display error toasts from failed operations', () => {
      const errorToast: ToastState = {
        visible: true,
        message: 'Network error: Unable to train model.',
        state: 'error'
      };
      const trainForm = fixture.debugElement.query(By.directive(TrainForm));

      trainForm.triggerEventHandler('showToast', errorToast);
      fixture.detectChanges();

      expect(component.toastState.state).toBe('error');
      expect(component.toastState.message).toContain('Network error');
    });
  });

  describe('Component Integration', () => {
    it('should render all required child components', () => {
      expect(fixture.debugElement.query(By.css('app-navbar'))).toBeTruthy();
      expect(fixture.debugElement.query(By.css('app-train-form'))).toBeTruthy();
      expect(fixture.debugElement.query(By.css('app-predict-form'))).toBeTruthy();
      expect(fixture.debugElement.query(By.css('app-animal-predict-display'))).toBeTruthy();
      expect(fixture.debugElement.query(By.css('app-notification-toast'))).toBeTruthy();
    });

    it('should have a main content area', () => {
      const main = fixture.debugElement.query(By.css('main'));
      expect(main).toBeTruthy();
    });
  });
});