# 🐶🐱 Dog vs Cat Image Classifier (MobileNetV2)

A complete Dog vs Cat classifier built using MobileNetV2. Everything is written in a clean, continuous, easy-to-read format without long paragraphs.

Project structure:
dog-cat-classifier/
├── models/mobilenetv2_best.h5
├── src/train.py
├── src/predict.py
├── data/training_set/
├── data/test_set/
├── requirements.txt
└── README.md

Dataset:
- Download from: https://www.kaggle.com/datasets/tongpython/cat-and-dog
- Extract into:
  data/training_set/
  data/test_set/
- Do NOT upload dataset to GitHub.

Setup:
git clone https://github.com/Nikhilyadav18/dog-cat-classifier.git  
cd dog-cat-classifier  
pip install -r requirements.txt  

Training:
- Run: python src/train.py
- Model gets saved automatically at: models/mobilenetv2_best.h5

Using Pretrained Model:
- Download from Google Drive or GitHub Releases
- Place file at: models/mobilenetv2_best.h5

Prediction:
python src/predict.py image.jpg  
- Outputs: Dog / Cat

Requirements:
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- h5py  
(Already listed in requirements.txt)

Notes:
- Keep models/ folder empty in repo except placeholder.
- Never upload dataset or large model files to GitHub.

Author:
Nikhil Yadav  
GitHub: https://github.com/Nikhilyadav18  
License: MIT
