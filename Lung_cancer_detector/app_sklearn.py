import os
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# ===== تكوين الصفحة =====
st.set_page_config(
    page_title="تصنيف أمراض الرئة",
    page_icon="🫁",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .cancer {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 تحليل صور الرئة")
st.subheader("نموذج ذكاء اصطناعي للكشف عن سرطان الرئة")

MODEL_PATH = "lung_model.pkl"
SCALER_PATH = "scaler.pkl"

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    return None, None, False

def extract_features(image):
    """استخراج ميزات من الصورة"""
    # تحويل إلى رمادي
    img_array = np.array(image.convert('L'))
    
    # تغيير الحجم إلى 224x224
    from PIL import Image as PILImage
    img = PILImage.fromarray(img_array)
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # استخراج ميزات بسيطة
    features = []
    
    # إحصائيات عامة
    features.append(np.mean(img_array))
    features.append(np.std(img_array))
    features.append(np.min(img_array))
    features.append(np.max(img_array))
    
    # إحصائيات من أجزاء مختلفة
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
    
    # حافات
    edges = np.abs(np.diff(img_array, axis=0)).mean() + np.abs(np.diff(img_array, axis=1)).mean()
    features.append(edges)
    
    return np.array(features).reshape(1, -1)

# تحميل النموذج
model, scaler, model_loaded = load_model()

if not model_loaded:
    st.warning("⚠️ النموذج غير موجود")
    st.info("""
    **الخطوات:**
    1. ضع صور الرئة في مجلد `img/`
    2. استخدم أسماء: `normal_*.jpg` و `cancer_*.jpg`
    3. شغّل: `python train_model_sklearn.py`
    """)
else:
    st.success("✓ تم تحميل النموذج")
    
    # ===== واجهة الرفع =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "اختر صورة أشعة للرئة",
            type=["jpg", "jpeg", "png", "bmp"]
        )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L').convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        # استخراج الميزات
        features = extract_features(image)
        features_scaled = scaler.transform(features)
        
        # التنبؤ
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        class_names = ["سليمة ✓", "سرطان ✗"]
        
        with col2:
            st.markdown("### 📊 النتائج")
            
            if prediction == 0:
                confidence = probability[0] * 100
                st.markdown(
                    f"""
                    <div class="result-box healthy">
                    الرئة: <b>سليمة</b><br/>
                    الثقة: {confidence:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"😊 الحمد لله - الرئة سليمة بنسبة {confidence:.1f}%")
            else:
                confidence = probability[1] * 100
                st.markdown(
                    f"""
                    <div class="result-box cancer">
                    الرئة: <b>قد تحتوي على سرطان</b><br/>
                    الثقة: {confidence:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.error(f"⚠️ تحذير - قد تحتوي على سرطان بنسبة {confidence:.1f}%")
        
        st.markdown("---")
        st.markdown("### 📈 تحليل التفاصيل")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("احتمالية الرئة السليمة", f"{probability[0]*100:.1f}%")
        with col2:
            st.metric("احتمالية السرطان", f"{probability[1]*100:.1f}%")
        
        st.info("⚠️ **تحذير**: هذا النموذج للأغراض التعليمية فقط. استشر طبيب متخصص.")
    else:
        st.info("👈 يرجى رفع صورة أشعة للبدء")

# الشريط الجانبي
st.sidebar.markdown("### ⚙️ معلومات")
st.sidebar.write("""
- **النوع**: Random Forest Classifier
- **الميزات**: 13 ميزة من الصورة
- **الفئات**: سليمة / سرطان
""")
