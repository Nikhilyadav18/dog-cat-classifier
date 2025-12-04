# 🐶🐱 Dog vs Cat Image Classifier (MobileNetV2)

This project is a complete dog vs cat classifier built using MobileNetV2. Everything is explained in one continuous flow including dataset setup, installation, running files, training, prediction, requirements, and author information. No separate sections or splitting is used so the whole README stays in a single piece exactly as required.

The project folder looks like this:
dog-cat-classifier/
├── models/mobilenetv2_best.h5
├── src/train.py
├── src/predict.py
├── data/training_set/
├── data/test_set/
├── requirements.txt
└── README.md

To start using the project, first download the dog vs cat dataset from: https://www.kaggle.com/datasets/tongpython/cat-and-dog and after downloading, extract it inside the data folder so that your directory becomes:
data/training_set/
data/test_set/
This dataset must not be uploaded to GitHub because it is too large.

Once the dataset is ready, you need to install the project dependencies. Clone the repository and move inside it:
git clone https://github.com/Nikhilyadav18/dog-cat-classifier.git
cd dog-cat-classifier
and install the required packages with:
pip install -r requirements.txt

Now you can run the files of this project. To run the training file, use this command:
python src/train.py
This command trains the MobileNetV2 model using the images inside data/training_set and data/test_set. After training completes, the best model automatically gets saved at:
models/mobilenetv2_best.h5

If you do not want to train and prefer using a ready model, then download it from your Google Drive or GitHub Releases link and place the file exactly at:
models/mobilenetv2_best.h5

To run prediction, simply run the predict file with an image path. Use:
python src/predict.py image.jpg
The script loads the trained model, processes the image, and prints either Dog or Cat based on the model output. This is the same command you will run for any test image. Any image path will work as long as the file exists.

The important libraries used in this project are TensorFlow, OpenCV, NumPy, Matplotlib, and h5py. These are included in requirements.txt and installing them with pip ensures the training and prediction scripts work smoothly without errors.

This project was created by Nikhil Yadav. More work can be found here: https://github.com/Nikhilyadav18. The project is available under the MIT License.
