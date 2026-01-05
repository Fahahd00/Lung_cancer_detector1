# نظام كشف سرطان الرئة (Streamlit + PyTorch)

تطبيق ويب بسيط لتحليل صور أشعة الرئة وتصنيفها إلى سليمة أو مشتبه بوجود سرطان باستخدام نموذج CNN مبني بـ PyTorch، مع نسخة بديلة تعتمد على Scikit-learn لاستخراج ميزات يدوية. الواجهة تعمل عبر Streamlit.

## المزايا
- رفع صورة أشعة سينية للرئة والحصول على تنبؤ سريع مع نسبة الثقة.
- نموذجين جاهزين: شبكة عصبية التفافية (CNN) باستخدام PyTorch، أو مصنف RandomForest/GradientBoosting باستخدام ميزات يدوية.
- نصائح واضحة لتجهيز البيانات وتدريب النموذج وحفظه للتشغيل الفوري.
- يمكن توليد بيانات صناعية للتجارب السريعة عبر `generate_data.py`.

## المتطلبات
- Python 3.10 أو أحدث (يفضل 64 بت)
- حزم Python في [requirements.txt](requirements.txt):
  - torch, torchvision, Pillow, numpy, streamlit
- ذاكرة كافية لتحميل النموذج وتشغيله على المعالج (CPU).

## التشغيل السريع (تطبيق Streamlit)
1) تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```
2) التأكد من وجود ملف النموذج `lung_model.pth` في جذر المشروع (يمكنك تدريبه كما في القسم التالي).
3) تشغيل التطبيق:
```bash
streamlit run app.py
```
4) افتح الرابط المحلي الذي يظهر في الطرفية لرفع صورة والحصول على النتيجة.

## تدريب نموذج الـ CNN (PyTorch)
1) ضع صور التدريب في مجلد `img/` مع تسمية الملفات بحيث تحتوي على الكلمات `normal` أو `healthy` للصور السليمة، و`cancer` أو `tumor` للصور المريضة.
2) شغل سكربت التدريب:
```bash
python train_model.py
```
3) بعد انتهاء التدريب سيتولد ملف `lung_model.pth` في جذر المشروع ويستعمله التطبيق تلقائيا.

## تدريب نموذج بديل (Scikit-learn)
1) تأكد من وجود الصور في `img/` بالتسمية نفسها.
2) شغل:
```bash
python train_model_sklearn.py
```
3) سيتولد `lung_model.pkl` و`scaler.pkl` ويستخدمهما التطبيق [app_sklearn.py](app_sklearn.py).
4) لتجربة نموذج محسن بمزايا أكثر، يمكن استخدام [train_improved.py](train_improved.py) بنفس خطوات الصور.

## توليد بيانات صناعية للتجربة
إذا لم تتوفر لديك بيانات، يمكن توليد 30 صورة صناعية (15 سليمة، 15 بسرطان) عبر:
```bash
python generate_data.py
```
سيتم إنشاء مجلد `img/` من الصفر بالصور الصناعية.

## بنية المشروع
- [app.py](app.py): واجهة Streamlit الرئيسية باستخدام نموذج PyTorch.
- [app_sklearn.py](app_sklearn.py): واجهة Streamlit لنسخة Scikit-learn.
- [app_simple.py](app_simple.py): صفحة جاهزية مبسطة تعرض خطوات الإعداد.
- [train_model.py](train_model.py): تدريب CNN وحفظ `lung_model.pth`.
- [train_model_sklearn.py](train_model_sklearn.py): تدريب RandomForest وحفظ `lung_model.pkl` و`scaler.pkl`.
- [train_improved.py](train_improved.py): تدريب GradientBoosting بمزايا إضافية.
- [generate_data.py](generate_data.py): توليد بيانات صناعية للتجارب.
- [requirements.txt](requirements.txt): المتطلبات البرمجية.

## نشر المشروع على GitHub
1) أنشئ مستودعاً جديداً في حسابك GitHub دون ملفات افتراضية.
2) من داخل مجلد المشروع شغل الأوامر التالية (مع استبدال الرابط بعنوان مستودعك):
```bash
git init
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git add .
git commit -m "Initial commit: lung cancer detector"
git push -u origin main
```
3) بعد الدفع، احصل على رابط المستودع مثل: `https://github.com/<USERNAME>/<REPO>`.

## نشر التطبيق أونلاين (Streamlit Community Cloud)
1) ارفع الكود إلى GitHub كما في الخطوات السابقة.
2) ادخل إلى https://streamlit.io/cloud وأنشئ تطبيقاً جديداً، اختر المستودع وملف `app.py` كنقطة تشغيل، وحدد فرع `main`.
3) تأكد من تفعيل الخيار لإعداد بيئة افتراضية تلقائياً، وسيتم تثبيت ما في `requirements.txt`.
4) بعد الإطلاق سيظهر لك رابط عام للتطبيق يمكنك مشاركته.

## ملاحظات مهمة
- النموذج للتجارب التعليمية وليس بديلاً عن التشخيص الطبي. يجب الرجوع لطبيب مختص لأي قرار طبي.
- جودة النتائج تعتمد بشدة على جودة ودقة بيانات التدريب وتوازن الفئات.
- لتحديث النموذج، أعد التدريب بعد إضافة بيانات جديدة ثم أعد نشر الملف الناتج.
