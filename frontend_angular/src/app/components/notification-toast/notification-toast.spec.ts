import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component } from '@angular/core';
import { NotificationToast } from './notification-toast';
import { ToastState } from '../../shared/interfaces/notification-toast-state.model';

@Component({
  template: `
    <app-notification-toast
      [state]="toastState"
      (clearToast)="onClearToast($event)"
    />
  `,
  imports: [NotificationToast]
})
class TestHostComponent {
  toastState: ToastState = { visible: false, message: '', state: 'info' };
  clearToastEvents: ToastState[] = [];

  onClearToast(state: ToastState) {
    this.clearToastEvents.push(state);
  }
}

describe('NotificationToast', () => {
  let hostComponent: TestHostComponent;
  let hostFixture: ComponentFixture<TestHostComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent, NotificationToast]
    }).compileComponents();

    hostFixture = TestBed.createComponent(TestHostComponent);
    hostComponent = hostFixture.componentInstance;
    hostFixture.detectChanges();
  });

  it('should create', () => {
    const toastElement = hostFixture.nativeElement.querySelector('app-notification-toast');
    expect(toastElement).toBeTruthy();
  });

  describe('visibility behavior', () => {
    it('should hide toast content when visible is false', () => {
      hostComponent.toastState = { visible: false, message: 'Hidden message', state: 'info' };
      hostFixture.detectChanges();

      const toastContent = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(toastContent).toBeNull();
    });

    it('should show toast content when visible is true', () => {
      hostComponent.toastState = { visible: true, message: 'Visible message', state: 'info' };
      hostFixture.detectChanges();

      const toastContent = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(toastContent).toBeTruthy();
    });

    it('should toggle visibility when state changes', () => {
      // Initially hidden
      hostComponent.toastState = { visible: false, message: 'Test', state: 'info' };
      hostFixture.detectChanges();
      expect(hostFixture.nativeElement.querySelector('app-notification-toast span')).toBeNull();

      // Show toast
      hostComponent.toastState = { visible: true, message: 'Test', state: 'info' };
      hostFixture.detectChanges();
      expect(hostFixture.nativeElement.querySelector('app-notification-toast span')).toBeTruthy();

      // Hide again
      hostComponent.toastState = { visible: false, message: 'Test', state: 'info' };
      hostFixture.detectChanges();
      expect(hostFixture.nativeElement.querySelector('app-notification-toast span')).toBeNull();
    });
  });

  describe('message display', () => {
    it('should display the provided message', () => {
      const testMessage = 'Test notification message';
      hostComponent.toastState = { visible: true, message: testMessage, state: 'info' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe(testMessage);
    });

    it('should update message when state changes', () => {
      hostComponent.toastState = { visible: true, message: 'First message', state: 'info' };
      hostFixture.detectChanges();

      let messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('First message');

      hostComponent.toastState = { visible: true, message: 'Second message', state: 'info' };
      hostFixture.detectChanges();

      messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('Second message');
    });

    it('should handle empty message', () => {
      hostComponent.toastState = { visible: true, message: '', state: 'info' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('');
    });

    it('should handle long messages', () => {
      const longMessage = 'A'.repeat(500);
      hostComponent.toastState = { visible: true, message: longMessage, state: 'info' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe(longMessage);
    });
  });

  describe('close button behavior', () => {
    it('should emit clearToast event with reset state when close button is clicked', () => {
      hostComponent.toastState = { visible: true, message: 'Click to close', state: 'success' };
      hostFixture.detectChanges();

      const closeButton = hostFixture.nativeElement.querySelector('app-notification-toast button');
      closeButton.click();
      hostFixture.detectChanges();

      expect(hostComponent.clearToastEvents.length).toBe(1);
      expect(hostComponent.clearToastEvents[0]).toEqual({
        visible: false,
        message: '',
        state: 'info'
      });
    });

    it('should have a clickable close button when toast is visible', () => {
      hostComponent.toastState = { visible: true, message: 'Has close button', state: 'error' };
      hostFixture.detectChanges();

      const closeButton = hostFixture.nativeElement.querySelector('app-notification-toast button');
      expect(closeButton).toBeTruthy();
      expect(closeButton.disabled).toBeFalsy();
    });

    it('should emit clearToast for each click', () => {
      hostComponent.toastState = { visible: true, message: 'Multiple clicks', state: 'info' };
      hostFixture.detectChanges();

      const closeButton = hostFixture.nativeElement.querySelector('app-notification-toast button');

      closeButton.click();
      closeButton.click();
      closeButton.click();

      expect(hostComponent.clearToastEvents.length).toBe(3);
    });

    it('should always emit the same reset state regardless of current toast state', () => {
      // Test with success state
      hostComponent.toastState = { visible: true, message: 'Success!', state: 'success' };
      hostFixture.detectChanges();
      hostFixture.nativeElement.querySelector('app-notification-toast button').click();

      // Test with error state
      hostComponent.toastState = { visible: true, message: 'Error!', state: 'error' };
      hostFixture.detectChanges();
      hostFixture.nativeElement.querySelector('app-notification-toast button').click();

      // Test with info state
      hostComponent.toastState = { visible: true, message: 'Info!', state: 'info' };
      hostFixture.detectChanges();
      hostFixture.nativeElement.querySelector('app-notification-toast button').click();

      // All emitted events should be the same reset state
      expect(hostComponent.clearToastEvents.length).toBe(3);
      hostComponent.clearToastEvents.forEach(event => {
        expect(event).toEqual({ visible: false, message: '', state: 'info' });
      });
    });
  });

  describe('toast state types', () => {
    it('should render success toast', () => {
      hostComponent.toastState = { visible: true, message: 'Operation successful', state: 'success' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('Operation successful');
    });

    it('should render error toast', () => {
      hostComponent.toastState = { visible: true, message: 'Something went wrong', state: 'error' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('Something went wrong');
    });

    it('should render info toast', () => {
      hostComponent.toastState = { visible: true, message: 'Here is some information', state: 'info' };
      hostFixture.detectChanges();

      const messageSpan = hostFixture.nativeElement.querySelector('app-notification-toast span');
      expect(messageSpan.textContent.trim()).toBe('Here is some information');
    });
  });
});
