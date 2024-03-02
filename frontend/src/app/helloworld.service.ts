import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HelloWorldService {

  private backendUrl = 'http://localhost:5000/hello'; // Use the correct backend URL here

  constructor(private http: HttpClient) { }

  getHelloWorld(): Observable<any> {
    return this.http.get<any>(this.backendUrl);
  }
}
