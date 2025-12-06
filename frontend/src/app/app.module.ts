import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http'; 
import { AppComponent } from './app.component';
import { UploadComponent } from './upload/upload.component';
import { DemoComponent } from './demo/demo.component';
import { RouterModule } from '@angular/router';  
import { routes } from './app.routes';        

@NgModule({
    declarations: [
      AppComponent
    ],
    imports: [
      BrowserModule,
      HttpClientModule,
      UploadComponent,
      DemoComponent,
      RouterModule.forRoot(routes)  
    ],
    bootstrap: [AppComponent]
  })

export class AppModule { }
