import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ModalService {
  private forgotPasswordModalOpen = new BehaviorSubject<boolean>(false);
  private createAccountModalOpen = new BehaviorSubject<boolean>(false);

  openForgotPasswordModal(){
    this.forgotPasswordModalOpen.next(true);
  }

  closeForgotPasswordModal(){
    this.forgotPasswordModalOpen.next(false);
  }

  isForgotPasswordModalOpen(){
    return this.forgotPasswordModalOpen.asObservable();
  }

  openCreateAccountModal(){
    this.createAccountModalOpen.next(true);
  }

  closeCreateAccountModal() {
    this.createAccountModalOpen.next(false);
  }

  isCreateAccountModalOpen() { 
    return this.createAccountModalOpen.asObservable();
  }

  constructor() { }
}
