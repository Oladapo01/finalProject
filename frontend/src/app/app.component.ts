import { Component, OnInit } from '@angular/core';
import  { ModalService } from './modal.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  isForgotPasswordModalOpen: Observable<boolean>;
  isCreateAccountModalOpen: Observable<boolean>;
  title = 'Homepage';


  constructor(private modalService: ModalService) {
    this.isForgotPasswordModalOpen = this.modalService.isForgotPasswordModalOpen();
    this.isCreateAccountModalOpen = this.modalService.isCreateAccountModalOpen();
  }

  openForgotPasswordModal() {
    this.modalService.openForgotPasswordModal();
  }

  openCreateAccountModal(){
    this.modalService.openCreateAccountModal();

  }

  ngOnInit() {

  }
}
