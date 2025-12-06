import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  standalone: true,
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
  imports: [HttpClientModule, CommonModule, FormsModule],
})
export class UploadComponent {
  selectedFile: File | null = null;
  selectedInstrument: string = 'violin';
  instruments: string[] = [
    'violin', 'cello', 'flute', 'trumpet', 'clarinet',
    'viola', 'bass', 'horn', 'trombone', 'piano'
  ];
  previewUrl: string | null = null;
  audioUrl: string | null = null;
  isProcessing: boolean = false;

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  onUpload() {
    if (!this.selectedFile) return;

    this.isProcessing = true;

    const formData = new FormData();
    formData.append('file', this.selectedFile);
    formData.append('instrument', this.selectedInstrument); 

    this.http.post<any>('http://localhost:5050/convert', formData).subscribe({
      next: (res) => {
        this.previewUrl = `http://localhost:5050${res.debugImage}`;
        this.audioUrl = `http://localhost:5050${res.audioFile}`;
        this.isProcessing = false;
      },
      error: (err) => {
        console.error('Upload failed:', err);
        this.isProcessing = false;
      }
    });
  }
}
