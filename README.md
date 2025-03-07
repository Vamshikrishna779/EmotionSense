# EmotionSense-Speech Emotion Recognition

## Overview
This project aims to detect emotions from speech audio using machine learning techniques. The system extracts features from audio recordings, trains multiple classifiers, and utilizes an ensemble model to enhance accuracy. Additionally, a **Flask web application** has been developed to provide a user-friendly interface for emotion detection.

## Features
- **Audio-based Emotion Recognition**: Classifies speech into seven emotion categories.
- **Machine Learning Models**: Uses various classifiers to determine the most effective model.
- **Ensemble Learning**: Combines top-performing models for improved accuracy.
- **Flask Web App**: Provides a user interface for uploading audio files and getting predictions.
- **Data Visualization**: Displays feature distributions and model performance analysis.

## Technologies Used
- **Python**
- **Machine Learning Libraries**: Scikit-learn, NumPy, Pandas
- **Audio Processing**: Librosa
- **Web Development**: Flask, HTML, CSS, JavaScript
- **Model Persistence**: Joblib

## Dataset & Feature Extraction
- The dataset contains audio files categorized by emotions:
  - `OAF_Fear`, `OAF_Pleasant_surprise`, `OAF_Sad`, `OAF_angry`, `OAF_disgust`, `OAF_happy`, `OAF_neutral`.
- Features extracted: **MFCC (Mel Frequency Cepstral Coefficients)**.
- The extracted dataset is structured as a `(1400, 13)` feature matrix.

## Machine Learning Models
| Model                   | Accuracy (%) |
|-------------------------|-------------|
| RandomForest           | 91.43       |
| SVM                    | 87.86       |
| Logistic Regression    | 85.00       |
| K-Nearest Neighbors    | 86.79       |
| Decision Tree          | 77.86       |
| Naive Bayes            | 86.43       |
| Gradient Boosting      | 87.86       |

## Ensemble Model - `EmotionEnsembleClassifier`
- Combines **RandomForest, SVM, and Gradient Boosting** using majority voting.
- Achieved **91.07% accuracy**.

## Flask Web Application
- Allows users to upload audio files for real-time emotion detection.
- Backend processes audio, extracts MFCC features, and predicts emotion using the trained ensemble model.
- Built using **Flask, HTML, CSS, and JavaScript**.

## Installation & Setup
### 1. Clone the Repository
```sh
git clone https://github.com/your-username/speech-emotion-detection.git
cd speech-emotion-detection
```
### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```sh
pip install -r requirements.txt
```
### 4. Run the Flask App
```sh
python app.py
```
### 5. Access the Web Interface
Open your browser and go to: `http://localhost:5000/`

## Project Structure
```
├── app.py                # Flask application
├── models/               # Saved machine learning models
├── static/               # CSS, JavaScript files
├── templates/            # HTML files for web UI
├── data/                 # Dataset and processed features
├── requirements.txt      # Dependencies
├── README.md            # Documentation
```

## Future Enhancements
- Implement **Deep Learning (CNN/LSTM)** for better feature extraction.
- Improve **real-time processing** for live speech input.
- Enhance **dataset diversity** to cover more emotions.

## Contributors
- **G Vamshikrishna** - [GitHub](https://github.com/Vamshikrishna779)

## Contact
For any queries or collaboration opportunities:
- **GitHub**: [Vamshikrishna779](https://github.com/Vamshikrishna779)

## License
This project is licensed under the **MIT License**.
