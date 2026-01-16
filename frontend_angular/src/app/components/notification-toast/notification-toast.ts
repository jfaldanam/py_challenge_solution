import { Component, input, output } from '@angular/core';
import { ToastState } from '../../shared/interfaces/notification-toast-state.model';

@Component({
  selector: 'app-notification-toast',
  imports: [],
  templateUrl: './notification-toast.html',
})
export class NotificationToast {
  state = input.required<ToastState>();
  clearToast = output<ToastState>();

  sendClearEvent() {
    this.clearToast.emit({ visible: false, message: "" });
  }
}
