import { Component } from '@angular/core';

@Component({
  selector: 'app-forgot-password-modal',
  templateUrl: './forgot-password-modal.component.html',
  styleUrls: ['./forgot-password-modal.component.css']
})
export class ForgotPasswordModalComponent {
  email = '';
  showModal: boolean = true;
  
  onClose() {
    // Handle the close action here
    this.showModal = false;
    console.log('Modal closed');
  }
  onSubmit() {
    // Handle the submit action here
    console.log('Submit:', this.email);
  }
  onResetPassword() {
    // Handle reset password logic here
    console.log('Reset password:', this.email);
  }

}
