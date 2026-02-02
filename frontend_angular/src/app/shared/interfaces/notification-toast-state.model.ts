export interface ToastState {
  visible: boolean,
  message: string,
  state: 'success' | 'error' | 'info',
}
