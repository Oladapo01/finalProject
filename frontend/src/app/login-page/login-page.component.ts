import { Component } from '@angular/core';
import { ModalService } from '../modal.service';

@Component({
  selector: 'app-login-page',
  templateUrl: './login-page.component.html',
  styleUrls: ['./login-page.component.css']
})
export class LoginPageComponent {
  user = {
    username: '',
    password: ''
  };

  constructor(private modalService: ModalService) {}

  onLogin() {
    // Handle login logic here
    console.log('Login:', this.user);
  }


  openForgotPasswordModal() {
    this.modalService.openForgotPasswordModal();
  }

  openCreateAccountModal() {
    this.modalService.openCreateAccountModal();
  }

  
  

}
