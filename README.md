# SkinGPT

An AI-powered skin condition analysis system using deep learning and computer vision for multi-task skin health assessment.

## Overview

SkinGPT is a deep learning system that analyzes 6-grid skin images to evaluate multiple skin health indicators. The system uses a transformer-based multi-modal architecture to process different imaging modalities and provide comprehensive skin condition assessments.

## Features

- Multi-modal image analysis (standard, UV, brown spots, redness, UV damage)
- Six-task evaluation: fine lines/wrinkles, pigmentation, pores, redness, UV damage, and overall texture
- Transformer-based cross-modal feature fusion
- Web-based interface for easy interaction
- AI-powered skincare recommendations

## Requirements

- Python 3.12+
- PyTorch 2.7.0
- CUDA 12.8 (for GPU acceleration)
- 8GB+ GPU memory recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SkinGPT.git
cd SkinGPT
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

3. Download or train the model weights and place them in the `models/` directory.

## Usage

### Running the Server

Start the Flask server:
```bash
python server.py
```

The server will start on `http://127.0.0.1:7860` by default.

### Using the Web Interface

1. Open `Page.html` in a web browser
2. Upload a 6-grid skin image (2 rows x 3 columns layout)
3. Click "Analyze" to process the image
4. Review the analysis results and recommendations

### Training the Model

To train the model on your own dataset:
```bash
python Training-6871.py
```

Ensure your dataset follows the expected structure with multi-modal images and labels in CSV format.

## Performance

- Accuracy: 68%+ on validation datasets
- Average inference time: <3 seconds per image
- Supported image formats: JPG, PNG
- Maximum image size: 20MB

## API Endpoints

- `GET /health` - Check server and model status
- `POST /analyze` - Analyze skin image
- `POST /ai_advice` - Get AI-powered skincare recommendations
- `GET /diag` - Diagnostic information

## Environment Variables

- `SKIN_MODEL_PATH` - Path to model weights (default: `models/super_multitask_skinnet.pth`)
- `DEEPSEEK_API_KEY` - API key for AI recommendations (optional)
- `HOST` - Server host (default: `127.0.0.1`)
- `PORT` - Server port (default: `7860`)


