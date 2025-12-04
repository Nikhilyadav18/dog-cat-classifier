import tensorflow as tf
import numpy as np
import cv2
import sys

model = tf.keras.models.load_model("../models/mobilenetv2_best.h5")

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    if pred > 0.5:
        print("Prediction: Dog")
    else:
        print("Prediction: Cat")

if __name__ == "__main__":
    predict(sys.argv[1])
