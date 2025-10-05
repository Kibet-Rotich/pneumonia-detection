# Pneumonia Prediction AI

## Overview

This project is a deep learning-based web application that predicts Pneumonia from chest X-ray images. It includes:
- A CNN or transfer learning model (e.g., ResNet)
- Flask backend for inference
- A futuristic HTML/CSS/JS frontend
- Live upload mode and demo image mode

## Features

- Predict Pneumonia from uploaded X-ray images
- Demo images for testing without uploads
- API endpoints for predictions and demo image fetching
- Modal-based UI popup for demo predictions

## Project Structure

```
project/
│
├── app.py                # Flask backend
├── model/                # Saved model files
├── static/
│   ├── script.js
│   ├── style.css
│   └── demo_images/      # Demo images
├── templates/
│   └── index.html
└── README.md
```

## Setup Instructions

1. Clone the repository or download the code.

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your trained model in the `model/` directory.

4. Ensure demo images are placed in `static/demo_images/`.

5. Run the Flask app:
   ```bash
   python app.py
   ```

## API Endpoints

### POST /predict

Accepts an uploaded file or a demo_url.

Returns JSON:
```json
{
  "prediction": "Pneumonia or Normal",
  "confidence": 0.95
}
```

### GET /demo-images

Returns a list of demo image URLs.

## Deployment

You can deploy with platforms like:
- Render
- Railway
- Heroku (with buildpacks)
- Docker + cloud providers

## Future Enhancements

- Add YOLO-based detection
- Add loading spinners
- Provide mobile responsive UI
- Cloud deployment + domain name
