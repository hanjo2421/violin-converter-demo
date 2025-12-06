# Music Sheet → MIDI Converter

[📊 Project Presentation](https://docs.google.com/presentation/d/1QjSQmkDjjMqo_EFFUWHUEGhhmmyyr9ULu49mNYsomqo/edit?usp=sharing)

An end-to-end system for converting scanned violin sheet music into MIDI, developed as our **EE123 (Digital Signal Processing)** final project at UC Berkeley.  
This repository includes the **music sheet analysis and MIDI generation** stages of the pipeline.

---

## 🎵 Project Summary

Our project implements the first half of a full audio synthesis system: Music Sheet → MIDI → DDSP Parameters → Violin Audio

## 🔬 Technical Approach

### 1. Music Sheet → MIDI Conversion

### Sheet → MIDI Pipeline
- CLAHE contrast enhancement  
- Non-local Means denoising  
- Otsu thresholding  
- Staff detection (Hough Transform + fallback methods)  
- Notehead template matching  
- OCR for instrument identification  
- MIDI event sequencing  

### Frontend Interface
The web-based UI allows users to:
- Upload sheet music  
- View detected notes on the staff  
- Download generated MIDI files  
- Preview MIDI playback  

### 4. Frontend Interface

### Frontend Interface
The web-based UI allows users to:
- Upload sheet music  
- View detected notes on the staff  
- Download generated MIDI files  
- Preview MIDI playback  

## ⚙️ Why DDSP Code Is Not Included
The DDSP-based synthesis components rely on **proprietary research architectures**, **unreleased model weights**, and **confidential training pipelines**.  
For this reason, the repository focuses solely on the **Music Sheet → MIDI** stages, which are fully functional and reproducible on their own.


