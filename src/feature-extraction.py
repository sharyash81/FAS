import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from numpy.fft import fft2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import tensorflow as tf

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_frequency_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = fft2(gray)
    f_transform = np.abs(f_transform)
    return f_transform.flatten()

def data_generator(label_file, batch_size=32, target_size=(224, 224)):
    data = pd.read_csv(label_file)
    total_samples = len(data)
    num_batches = (total_samples + batch_size - 1) // batch_size

    while True:
        for batch in range(num_batches):
            batch_data = data.iloc[batch * batch_size:(batch + 1) * batch_size]
            batch_frames = []
            batch_labels = []
            for _, row in batch_data.iterrows():
                frame_path = row['frame_path']
                label = row['label']
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame = cv2.resize(frame, target_size)
                    frame = frame / 255.0
                    batch_frames.append(frame)
                    batch_labels.append(label)
            yield np.array(batch_frames), np.array(batch_labels)

def load_and_extract_features_in_batches(label_file, batch_size=32, target_size=(224, 224)):
    data = pd.read_csv(label_file)
    features = []
    labels = []
    filenames = []
    total_samples = len(data)
    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch in range(num_batches):
        batch_data = data.iloc[batch * batch_size:(batch + 1) * batch_size]
        for _, row in batch_data.iterrows():
            frame_path = row['frame_path']
            label = row['label']
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Warning: Could not read frame {frame_path}")
                    continue
                frame = cv2.resize(frame, target_size)
                lbp_features = extract_lbp_features(frame)
                freq_features = extract_frequency_features(frame)
                combined_features = np.concatenate((lbp_features, freq_features))
                features.append(combined_features)
                labels.append(label)
                filenames.append(frame_path)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
    return np.array(features), np.array(labels), filenames

# Load and extract features in batches
label_file = '/content/drive/My Drive/CASIA_faceAntisp/labels.csv'
X_features, y_features, filenames = load_and_extract_features_in_batches(label_file)

# Check if features were extracted successfully
if len(X_features) == 0 or len(y_features) == 0:
    raise ValueError("No features extracted. Check the data and preprocessing steps.")

# Split the data
X_train_feat, X_test_feat, y_train_feat, y_test_feat, train_filenames, test_filenames = train_test_split(
    X_features, y_features, filenames, test_size=0.2, random_state=42
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_feat, y_train_feat)

# Predict and evaluate
y_pred_feat = clf.predict(X_test_feat)
print("Feature extraction method accuracy:", accuracy_score(y_test_feat, y_pred_feat))

# Save feature extraction predictions
def save_feature_predictions(filenames, y_pred, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "liveness_score", "liveness_score_crop", "liveness_score_frequency"])
        for filename, pred in zip(filenames, y_pred):
            writer.writerow([filename, pred, 0, pred])

save_feature_predictions(test_filenames, y_pred_feat, 'predictions_feature.csv')

