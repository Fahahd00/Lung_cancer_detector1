import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle

# تحديد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# مسار مجلد الصور
IMG_DIR = "img"
MODEL_PATH = "lung_model.pth"
CLASS_NAMES = ["Normal", "Cancer"]  # 0: سليمة، 1: سرطان

# تحويل الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Dataset مخصص
class LungDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # افترض أن الصور منظمة في مجلدات:
        # img/normal/ و img/cancer/
        # أو استخدم أسماء الملفات (مثل normal_1.jpg, cancer_1.jpg)
        
        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            if os.path.isfile(filepath):
                # تحديد الفئة من اسم الملف
                if "normal" in filename.lower() or "healthy" in filename.lower():
                    label = 0  # سليمة
                elif "cancer" in filename.lower() or "tumor" in filename.lower():
                    label = 1  # سرطان
                else:
                    continue  # تخطي الملفات غير المصنفة
                
                self.images.append(filepath)
                self.labels.append(label)
        
        print(f"تم العثور على {len(self.images)} صورة")
        print(f"سليمة: {self.labels.count(0)}, سرطان: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # فتح الصورة
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# نموذج CNN بسيط
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
            nn.Linear(256, 2)  # 2 classes: Normal, Cancer
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train():
    print("\n=== تدريب النموذج ===\n")
    
    # التحقق من وجود مجلد الصور
    if not os.path.exists(IMG_DIR):
        print(f"خطأ: مجلد {IMG_DIR} غير موجود!")
        return
    
    # إنشاء dataset
    dataset = LungDataset(IMG_DIR, transform=transform)
    
    if len(dataset) == 0:
        print("لا توجد صور للتدريب!")
        return
    
    # تقسيم البيانات
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # إنشاء النموذج
    model = LungClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # التدريب
    epochs = 20
    best_val_acc = 0
    
    for epoch in range(epochs):
        # التدريب
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # التحقق
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # حفظ أفضل نموذج
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ تم حفظ النموذج (Best Val Acc: {best_val_acc:.2f}%)")
        
        print()
    
    print(f"\n✓ انتهى التدريب! أفضل دقة: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
