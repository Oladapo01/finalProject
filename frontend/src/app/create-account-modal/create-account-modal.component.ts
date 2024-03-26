import { Component, OnInit } from '@angular/core';
import { AuthService } from '../auth.service';

@Component({
  selector: 'app-create-account-modal',
  templateUrl: './create-account-modal.component.html',
  styleUrls: ['./create-account-modal.component.css']
})
export class CreateAccountModalComponent {
  genderOptions = ['Male', 'Female', 'Other'];
  interests = ['Reading', 'Writing', 'Math', 'Science', 'History', 
  'Art', 'Music', 'Sports', 'Cooking', 'Other', 
  'Hiking', 'Camping', 'Cycling', 'Birdwatching', 'Gardening',
  'Painting', 'Pottery', 'Knitting', 'Calligraphy', 'Origami', 
  'Photography', 'Dance', 'Yoga', 'Meditation', 'Other'];
  step: number = 1;
  totalSteps: number = 9;

  constructor(private authService: AuthService) { }


  

  username: string = '';
  email: string = '';
  password: string = '';
  first_name: string = '';
  last_name: string = '';
  date_of_birth: string = '';
  gender: string = '';
  selectedInterests: string[] = [];
  goals: string = '';
  language_proficiency: string = '';
  preferred_learning_style: string = '';
  accessibility_needs: string ='';
  showModal: boolean = true;

  

  onClose() {
    // Handle the close action here
    this.showModal = false;
    console.log('Modal closed');
  }


  nextStep() {
    if (this.step < this.totalSteps) {
      this.step++;
    }
  }
  prevStep() {
    if (this.step > 1) {
      this.step--;
    }
  }

  isSelected(interest: string): boolean {
    return this.selectedInterests.includes(interest);
  }


  onInterestChange(event: any, interest: string) {
    const isChecked = event.target.checked;
    if (isChecked) {
      if (this.selectedInterests.length >= 5) {
        alert('You can only select up to 5 interests');
        return;
      }
      this.selectedInterests.push(interest);
    } else {
      const index = this.selectedInterests.indexOf(interest);
      if (index !== -1) {
        this.selectedInterests.splice(index, 1);
      }
    }
  }

  onSubmit() {
    // Handle the submit action here
    const user = {
      username: this.username,
      email: this.email,
      password: this.password,
      first_name: this.first_name,
      last_name: this.last_name,
      date_of_birth: this.date_of_birth,
      gender: this.gender,
      interests: this.interests,
      goals: JSON.parse(this.goals),
      language_proficiency: this.language_proficiency,
      preferred_learning_style: JSON.parse(this.preferred_learning_style),
      accessibility_needs: JSON.parse(this.accessibility_needs)
    };
    this.authService.signup(user).subscribe({
      next: (response) => {
        alert(response.message);
        console.log('User created:',  response);
      }, 
      error: (error) => {
        alert(error.message);
        console.error('Error creating user:', error);
      }
    });
  }
  

}
