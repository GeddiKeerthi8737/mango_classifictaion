# mango_classifictaion
# Import more advanced classifiers
import cv2
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = os.path.join('DATASET', 'FRUIT PHOTOS', '*.jpg')

# Load the images
images = [cv2.imread(file) for file in glob.glob(dataset_path)]

# Extract color histograms (features)
features = []

for image in images:
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate color histograms
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])

    # Normalize histograms
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
    hist_saturation = cv2.normalize(hist_saturation, hist_saturation).flatten()

    # Combine histograms into a feature vector
    feature = np.hstack([hist_hue, hist_saturation])
    features.append(feature)

# Ensure images were loaded correctly
if any(image is None for image in images):
    print("Some images failed to load. Check the dataset path or the image files.")
else:
    print("All images loaded successfully.")

# Assuming you've labeled your data manually:
# Adjust labels to match the number of images
labels = [0] * 36 + [1] * 36  # 0 for raw, 1 for ripened

# Ensure that features and labels match in length
if len(features) != len(labels):
    raise ValueError(f"Mismatch in number of features ({len(features)}) and labels ({len(labels)})")

# Shuffle the data (optional but recommended)
features, labels = shuffle(features, labels, random_state=42)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict whether a mango is raw or ripened
def predict_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract features (same as before)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()
    hist_saturation = cv2.normalize(hist_saturation, hist_saturation).flatten()
    feature = np.hstack([hist_hue, hist_saturation])

    # Predict using the trained model
    prediction = model.predict([feature])

    if prediction == 0:
        print("The mango is raw.")
    else:
        print("The mango is ripened.")

# Example usage
# Replace 'your_image_path.jpg' with the path to the image you want to test
# predict_image('your_image_path.jpg', model)
predict_image('green_mango.jpg', model)
predict_image('ripen_3.png',model)
predict_image('predict.JPG',model)
predict_image('red_mango.jpg',model)
