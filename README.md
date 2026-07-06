# 🎭 Age, Gender & Emotion Detector

A real-time face analysis application using computer vision and deep learning. This Python application uses your webcam to detect faces and predict:

- **Age** (9 age ranges from 0-2 to 70+)
- **Gender** (Male/Female) 
- **Race/Ethnicity** (7 categories)
- **Emotion** (7 emotions via FER library)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎥 Real-time webcam processing
- 📸 Image file analysis support
- 🔄 Face tracking with temporal smoothing (reduces prediction flickering)
- 🎯 Multi-face detection and analysis
- 🧠 FairFace model for unbiased demographic predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- ~100MB disk space for models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sidhazzzzzz/VisageX.git
   cd VisageX
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model**
   ```bash
   python download_models.py
   ```

### Usage

**Webcam mode (default):**
```bash
python main.py
```

**Analyze an image:**
```bash
python main.py --image path/to/photo.jpg
```

**Controls:**
- Press `q` to quit the application

## 📁 Project Structure

```
age-gender-detector/
├── main.py              # Main application entry point
├── fairface_model.py    # FairFace model wrapper and face analysis
├── download_models.py   # Model download utility
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## 🔧 How It Works

1. **Face Detection**: Uses Haar Cascade classifiers for fast face detection
2. **Face Alignment**: Crops and pads detected faces for optimal analysis
3. **Attribute Prediction**: FairFace ResNet34 model predicts age, gender, and race
4. **Emotion Detection**: FER library analyzes facial expressions
5. **Temporal Smoothing**: Tracks faces across frames to smooth predictions

## 🧪 Model Information

This project uses the **FairFace** model, which is specifically trained to be fair and balanced across different demographic groups:

- **Architecture**: ResNet34
- **Training Data**: FairFace dataset (86,744 images, balanced across race and gender)
- **Paper**: [FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age](https://arxiv.org/abs/1908.04913)

## 📊 Accuracy

| Attribute | Accuracy |
|-----------|----------|
| Race      | ~72%     |
| Gender    | ~94%     |
| Age (±10) | ~61%     |

*Based on FairFace paper benchmarks*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FairFace](https://github.com/dchen236/FairFace) - Fair face attribute dataset and model
- [FER](https://github.com/justinshenk/fer) - Facial expression recognition
- [OpenCV](https://opencv.org/) - Computer vision library

## ⚠️ Disclaimer

This tool is for educational and research purposes. Face analysis predictions are estimates and may not always be accurate. Please use responsibly and be aware of privacy implications when analyzing faces.
