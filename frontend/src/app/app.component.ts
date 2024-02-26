import { Component, OnInit } from '@angular/core';
import { HelloWorldService } from './helloworld.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'frontend';
  message: string = 'Loading...';
  

  constructor(private helloWorldService: HelloWorldService) {}

  ngOnInit() {
    this.helloWorldService.getHelloWorld().subscribe(
      (response) => {
        // Assuming the response is in the format { message: "Hello, World!" }
        this.message = response.message;
      }, (error) => {
        console.error('There was an error!', error);
        this.message = 'Error fetching the message';
      }
    );
  }
}
