import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

IMG_DIR = "img"
MODEL_PATH = "lung_model.pkl"
SCALER_PATH = "scaler.pkl"

def extract_features_from_image(image_path):
    """استخراج 13 ميزة من الصورة"""
    img = Image.open(image_path).convert('L')
    
    # تغيير الحجم
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    features = []
    
    # 1-4: الإحصائيات العامة
    features.append(np.mean(img_array))
    features.append(np.std(img_array))
    features.append(np.min(img_array))
    features.append(np.max(img_array))
    
    # 5-12: إحصائيات الأرباع
    h, w = img_array.shape
    quadrants = [
        img_array[:h//2, :w//2],
        img_array[:h//2, w//2:],
        img_array[h//2:, :w//2],
        img_array[h//2:, w//2:]
    ]
    
    for quad in quadrants:
        features.append(np.mean(quad))
        features.append(np.std(quad))
    
    # 13: الحافات
    edges = np.abs(np.diff(img_array, axis=0)).mean() + np.abs(np.diff(img_array, axis=1)).mean()
    features.append(edges)
    
    return np.array(features)

def train():
    print("\n=== تدريب النموذج ===\n")
    
    if not os.path.exists(IMG_DIR):
        print(f"خطأ: مجلد {IMG_DIR} غير موجود!")
        return
    
    # جمع الصور
    images = []
    labels = []
    
    for filename in os.listdir(IMG_DIR):
        filepath = os.path.join(IMG_DIR, filename)
        if os.path.isfile(filepath):
            try:
                # تحديد الفئة
                if "normal" in filename.lower() or "healthy" in filename.lower():
                    label = 0
                elif "cancer" in filename.lower() or "tumor" in filename.lower():
                    label = 1
                else:
                    continue
                
                # استخراج الميزات
                features = extract_features_from_image(filepath)
                images.append(features)
                labels.append(label)
                print(f"✓ {filename} -> {'سليمة' if label == 0 else 'سرطان'}")
            except Exception as e:
                print(f"✗ خطأ في {filename}: {e}")
    
    if len(images) == 0:
        print("لا توجد صور للتدريب!")
        return
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\n📊 الإحصائيات:")
    print(f"إجمالي الصور: {len(X)}")
    print(f"صور سليمة: {np.sum(y == 0)}")
    print(f"صور سرطان: {np.sum(y == 1)}")
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # تطبيع البيانات
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # التدريب
    print("\n🔄 جاري التدريب...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # التقييم
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n✓ انتهى التدريب!")
    print(f"دقة التدريب: {train_score*100:.2f}%")
    print(f"دقة الاختبار: {test_score*100:.2f}%")
    
    # حفظ النموذج
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n💾 تم حفظ النموذج في {MODEL_PATH}")
    print(f"💾 تم حفظ المعايرة في {SCALER_PATH}")

if __name__ == "__main__":
    train()
