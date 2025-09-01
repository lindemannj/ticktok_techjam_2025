# TikTokJam 2025 – Privacy Protector Demo

**Author:** Jonas Lindemann  
**Event:** TikTokJam 2025 Hackathon  

## Overview

This project was developed as part of the TikTokJam 2025 Hackathon. The challenge was to design a solution for automatically detecting and obscuring personal information in images. I participated solo and created a demo application that demonstrates a privacy-preserving workflow for images.

The application allows users to upload images, automatically detects faces and text, and then pixelates these sensitive elements to protect privacy. This solution showcases a practical approach to handling personal data in shared media while maintaining usability.

## Features

- **Face Detection:** Automatically identifies human faces in uploaded images.  
- **Text Detection:** Scans images for textual content, such as names, addresses, or other personal data.  
- **Automatic Redaction:** Detected faces and text are pixelated to anonymize sensitive information.  
- **Easy Demo:** A Streamlit frontend allows users to test the functionality with minimal setup.  

## Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Deployment:** Docker Compose (`docker-compose up` to start both backend and frontend)  
- **Detection Libraries:** OpenCV, Tesseract OCR, face_recognition

## Getting Started

1. Clone this repository:  
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

2. start the application:
   `docker-compose up` to start both backend and frontend
