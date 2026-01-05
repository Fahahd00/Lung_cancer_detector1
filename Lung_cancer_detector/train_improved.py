import os
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import label
import pickle

IMG_DIR = "img"
MODEL_PATH = "lung_model.pkl"
SCALER_PATH = "scaler.pkl"

def extract_features(image_path):
    """Extract spot-focused features"""
    img = Image.open(image_path).convert('L')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    
    features = []
    
    # Basic stats
    features.append(np.mean(arr))
    features.append(np.std(arr))
    features.append(np.min(arr))
    features.append(np.max(arr))
    
    # Percentiles
    features.extend([np.percentile(arr, 25), np.percentile(arr, 50), 
                     np.percentile(arr, 75), np.percentile(arr, 90)])
    
    # Spot detection - dark pixels
    thresh = np.percentile(arr, 30)
    dark = arr < thresh
    features.append(np.sum(dark))
    features.append(np.sum(dark) / arr.size)
    
    # Connected components
    try:
        labeled, num = label(dark)
        features.append(num)
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            features.append(np.mean(sizes[1:]))
            features.append(np.max(sizes[1:]))
        else:
            features.append(0)
            features.append(0)
    except:
        features.extend([0, 0, 0])
    
    # Edge detection
    edges_x = np.abs(np.diff(arr, axis=0)).mean()
    edges_y = np.abs(np.diff(arr, axis=1)).mean()
    features.extend([edges_x, edges_y, edges_x + edges_y])
    
    # Histogram
    hist, _ = np.histogram(arr, bins=4, range=(0, 256))
    hist = hist / hist.sum()
    features.extend(hist)
    
    # Center analysis
    center = arr[50:174, 50:174].mean()
    features.append(center)
    features.append((arr[50:174, 50:174] < thresh).sum())
    
    # Variance and dark pixel count
    features.append(np.var(arr - arr.mean()))
    features.append(np.sum(arr < np.percentile(arr, 20)))
    
    return np.array(features)

print("Training model...")

images = []
labels = []

for filename in sorted(os.listdir(IMG_DIR)):
    filepath = os.path.join(IMG_DIR, filename)
    if not os.path.isfile(filepath):
        continue
    
    if "normal" in filename.lower():
        lab = 0
    elif "cancer" in filename.lower():
        lab = 1
    else:
        continue
    
    try:
        feat = extract_features(filepath)
        images.append(feat)
        labels.append(lab)
        print(f"  {filename}")
    except Exception as e:
        print(f"  ERROR: {filename} - {str(e)}")

print(f"\nDataset: {len(images)} images")
print(f"Healthy: {sum(1 for l in labels if l == 0)}")
print(f"Cancer: {sum(1 for l in labels if l == 1)}")

X = np.array(images)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=8, random_state=42)
model.fit(X_train, y_train)

print(f"Train: {model.score(X_train, y_train)*100:.2f}%")
print(f"Test: {model.score(X_test, y_test)*100:.2f}%")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel saved!")
