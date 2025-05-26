import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SafetyGearDataset
from model import AttributeRecognitionModel
from torchvision import transforms

# Konfigurasi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './dataset'
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Transformasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
dataset = SafetyGearDataset(data_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, dan optimizer
model = AttributeRecognitionModel().to(device)
# (Opsional) Muat bobot pre-trained
# model.load_state_dict(torch.load('models/pretrained_model.pth'), strict=False)
criterion = nn.BCEWithLogitsLoss()  # Untuk multilabel classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loop pelatihan
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}')

# Simpan model
torch.save(model.state_dict(), 'models/safety_gear_model.pth')
print('Model disimpan ke models/safety_gear_model.pth')