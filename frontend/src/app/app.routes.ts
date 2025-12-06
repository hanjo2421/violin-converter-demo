// app.routes.ts
import { Routes } from '@angular/router';
import { DemoComponent } from './demo/demo.component';
import { UploadComponent } from './upload/upload.component';

export const routes: Routes = [
  { path: '', component: UploadComponent },
  { path: 'demo', component: DemoComponent }
];
