import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# ===== تكوين الصفحة =====
st.set_page_config(
    page_title="تصنيف أمراض الرئة",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS مخصص
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
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

# ===== تحميل النموذج =====
MODEL_PATH = "lung_model.pth"

class LungClassifier(nn.Module):
    def __init__(self):
        super(LungClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = LungClassifier().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, True
    else:
        return model, False

# ===== الصفحة الرئيسية =====
st.title("🫁 تحليل صور الرئة")
st.subheader("نموذج ذكاء اصطناعي للكشف عن سرطان الرئة")

# تحميل النموذج
model, model_loaded = load_model()

if not model_loaded:
    st.error("⚠️ النموذج غير موجود! يرجى تدريب النموذج أولاً باستخدام `train_model.py`")
    st.info("الخطوات:")
    st.write("""
    1. ضع صور الرئة في مجلد `img/`
    2. استخدم أسماء الملفات: 
       - `normal_*.jpg` للصور السليمة
       - `cancer_*.jpg` لصور السرطان
    3. شغّل: `python train_model.py`
    """)
else:
    st.success("✓ تم تحميل النموذج بنجاح")
    
    # ===== واجهة الرفع =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "اختر صورة أشعة للرئة",
            type=["jpg", "jpeg", "png", "bmp"]
        )
    
    if uploaded_file is not None:
        # عرض الصورة
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        # ===== التنبؤ =====
        device = torch.device("cpu")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # معالجة الصورة
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # التنبؤ
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # النتائج
        class_names = ["سليمة ✓", "سرطان ✗"]
        
        with col2:
            st.markdown("### 📊 النتائج")
            
            if predicted_class == 0:
                st.markdown(
                    f"""
                    <div class="result-box healthy">
                    الرئة: <b>سليمة</b><br/>
                    الثقة: {confidence:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"😊 الحمد لله - الرئة سليمة بنسبة ثقة {confidence:.1f}%")
            else:
                st.markdown(
                    f"""
                    <div class="result-box cancer">
                    الرئة: <b>قد تحتوي على سرطان</b><br/>
                    الثقة: {confidence:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.error(f"⚠️ تحذير - قد تحتوي على سرطان بنسبة ثقة {confidence:.1f}%")
        
        # تفاصيل إضافية
        st.markdown("---")
        st.markdown("### 📈 تحليل التفاصيل")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("الرئة السليمة", f"{probabilities[0][0].item()*100:.1f}%")
        
        with col2:
            st.metric("احتمالية السرطان", f"{probabilities[0][1].item()*100:.1f}%")
        
        # تحذير طبي
        st.info(
            "⚠️ **تحذير**: هذا النموذج للأغراض التعليمية فقط. "
            "لا يمكن الاعتماد عليه للتشخيص الطبي الفعلي. "
            "يرجى استشارة طبيب متخصص للتشخيص الدقيق."
        )
    
    else:
        st.info("👈 يرجى رفع صورة أشعة للبدء")
        
        st.markdown("---")
        st.markdown("### ℹ️ معلومات")
        st.write("""
        - تم تدريب النموذج على صور أشعة سينية للرئة
        - يصنف الصور إلى: **سليمة** أو **قد تحتوي على سرطان**
        - استخدم صور عالية الجودة للحصول على أفضل النتائج
        """)

# الشريط الجانبي
st.sidebar.markdown("### ⚙️ معلومات النموذج")
st.sidebar.write("""
- **النوع**: CNN (Convolutional Neural Network)
- **حجم الإدخال**: 224×224 بكسل
- **الفئات**: 2 (سليمة / سرطان)
- **الإطار**: PyTorch + Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 التعليمات")
st.sidebar.write("""
1. رفّع صورة أشعة سينية للرئة
2. سينتظر النموذج معالجة الصورة
3. ستظهر النتيجة مع نسبة الثقة
4. استشر الطبيب دائماً
""")
