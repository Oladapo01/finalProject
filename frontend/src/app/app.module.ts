import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { AppComponent } from './app.component';
import { LoginPageComponent } from './login-page/login-page.component';
import { AppRoutingModule } from './app-routing.module';
import { ForgotPasswordModalComponent } from './forgot-password-modal/forgot-password-modal.component';
import { CreateAccountModalComponent } from './create-account-modal/create-account-modal.component';

@NgModule({
  declarations: [
    AppComponent,
    LoginPageComponent,
    ForgotPasswordModalComponent,
    CreateAccountModalComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    AppRoutingModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
