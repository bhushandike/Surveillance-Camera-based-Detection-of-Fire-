# Fire Detection System Project

## Overview
The Fire Detection System is designed to identify signs of fire and smoke in real-time using advanced computer vision techniques. This algorithm processes video streams from surveillance cameras to detect flames and alerts authorities when necessary.

## Features
- **Real-time detection**: Continuously analyzes camera footage to identify fire hazards.
- **Alerts**: Sends notifications to security personnel and local fire departments upon detection.
- **Integration**: Can be integrated with existing surveillance systems and fire alarm systems.

## Technology Stack
- **Programming Language**: Python (21.3%)
- **Web Interface**: HTML (78.7%)
- **Libraries**: OpenCV, NumPy, TensorFlow, Keras
- **Framework**: Flask for web application
- **Hardware**: Compatible with standard CCTV cameras

## Project Structure
```
├── app.py                 # Flask application for web interface
├── fire.py               # Core fire detection module
├── from tensorflow.keras.py  # Model training and utilities
├── images.jpg            # Sample detection images
├── templates/            # HTML templates for web interface
└── README.md            # This file
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/bhushandike/Surveillance-Camera-based-Detection-of-Fire-
   cd Surveillance-Camera-based-Detection-of-Fire-
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download or train the fire detection model (if not included).

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## How It Works
The system uses Convolutional Neural Networks (CNNs) to analyze video frames from surveillance cameras. It:

1. **Captures frames** from the video feed
2. **Preprocesses images** for model input
3. **Detects fire characteristics** such as:
   - Flame patterns
   - Smoke signatures
   - Heat distortions
4. **Classifies results** as fire or no-fire
5. **Triggers alerts** when fire is detected with high confidence

## Usage
1. Set up surveillance cameras in fire-prone areas
2. Configure camera feeds in the application
3. Monitor the real-time detection dashboard
4. Receive alerts when potential fire is detected

## Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: Video frames (RGB images)
- **Output**: Binary classification (Fire/No-Fire) with confidence score

## Performance
The system achieves high accuracy in detecting:
- Open flames
- Smoke
- Heat signatures
- Fire-related visual artifacts

## Limitations
- Performance depends on camera resolution and angle
- Requires adequate lighting conditions for optimal detection
- May generate false positives in certain conditions (e.g., bright reflections)

## Safety Disclaimer
⚠️ **Important**: This system is an auxiliary tool and **NOT a replacement** for:
- Professional fire detection systems
- Smoke detectors
- Fire sprinkler systems
- Emergency response protocols

Always maintain proper fire safety measures and professional fire detection equipment.

## Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.

## License
This project is provided as-is for research and educational purposes.

## Contact
For questions or suggestions, please open an issue on this repository.