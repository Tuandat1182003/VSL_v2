import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common'; 

@Component({
  selector: 'app-root',
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  // Signals for reactive state
  isRecording = signal(false);
  isDragging = signal(false);
  statusText = signal('Ready to record or upload');
  transcript = signal('');
  errorMessage = signal('');

  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];

  // Drag and Drop support
  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging.set(false);
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      this.processAudio(file, file.name);
    } else {
      this.showError('Unsupported format. Please drop a valid audio file.');
    }
  }

  // Pointer events for hold-to-record or click-to-record
  private holdTimer: any;
  private isHold = false;

  async onPointerDown(e: PointerEvent) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();
    this.isHold = false;
    this.holdTimer = setTimeout(() => { this.isHold = true; }, 300);

    if (!this.isRecording()) {
      await this.startRecording();
    } else {
      this.stopRecording();
    }
  }

  onPointerUp(e: PointerEvent) {
    e.preventDefault();
    clearTimeout(this.holdTimer);
    if (this.isHold && this.isRecording()) {
      this.stopRecording();
    }
  }

  // Recording Logic using MediaRecorder API
  async startRecording() {
    this.errorMessage.set('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };

      this.mediaRecorder.onstop = () => {
        this.statusText.set('Processing...');
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        this.processAudio(audioBlob, 'recording.webm');
        stream.getTracks().forEach(track => track.stop()); // explicitly release mic
      };

      this.mediaRecorder.start();
      this.isRecording.set(true);
      this.statusText.set('Listening... (Release to stop)');
    } catch (err) {
      this.showError('Microphone permission denied or unavailable.');
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording()) {
      this.mediaRecorder.stop();
      this.isRecording.set(false);
    }
  }

  // API Integration with FastAPI
  private async processAudio(file: Blob, filename: string) {
    this.statusText.set('Transcribing...');
    this.transcript.set('');
    
    const formData = new FormData();
    formData.append('audio_file', file, filename);

    try {
      // Direct call to FastAPI server port 8000
      const response = await fetch('http://localhost:8000/api/transcribe', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Failed to transcribe audio.');
      }

      const result = await response.json();
      this.transcript.set(result.transcript);
      this.statusText.set('Transcription complete.');
    } catch (err: any) {
      this.showError(err.message || 'An error occurred during transcription.');
      this.statusText.set('Error');
    }
  }

  private showError(msg: string) {
    this.errorMessage.set(msg);
    setTimeout(() => this.errorMessage.set(''), 5000);
  }
}
