# Gerekli kütüphaneleri ekliyoruz
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Modelimizi tanımlıyoruz(3 katmalı modele geçildi)
class MYNet(nn.Module):
    def __init__(self):
        super(MYNet, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Boyutu 128 * 3 * 3 olarak güncellendi
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), 2))  # Boyut: (batch_size, 32, 14, 14)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Boyut: (batch_size, 64, 7, 7)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Boyut: (batch_size, 128, 3, 3)

        x = x.view(-1, 128 * 3 * 3)  # Düzleştir
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Eğitim fonksiyonu
def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad() # Önceki gradyanları sıfırla
        output = model(data)  # Modelin tahmini
        loss = F.nll_loss(output, target)  # Kayıp hesaplama
        loss.backward()  # Geri yayılım
        optimizer.step()

        running_loss += loss.item()  # Toplam kaybı güncelle
        _, predicted = output.max(1) # En yüksek olasılıklı tahmini al
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
 # Eğitim sonunda doğruluğu ve kaybı yazdır
    print(f'Epoch {epoch}: Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')
# Test fonksiyonu
def test(model, device, test_loader):
    model = model.to(device)
    model.eval()# Modeli değerlendirme moduna al
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Değerlendirme sırasında gradyan hesaplaması yapılmaz
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Toplam kaybı hesapla
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
# Ana fonksiyon
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
# Veri dönüştürme işlemleri
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),# Stanart Normalizasyon
    ])
 # MNIST veri setini indirme ve yükleme
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)# Kendi donanım gücünüze göre Worker sayısını ve batch boyutunu değiştirebilirsiniz.
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
# Modeli başlat
    model = MYNet().to(device)
     # Optimizatör(Adam daha iyi diye gözlemledim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Eğitim döngüsü
    for epoch in range(1, 21):  # 20 epoch boyunca eğitim
        train(model, device, train_loader, optimizer, epoch)# Eğitim fonksiyonunu çağır
        test(model, device, test_loader) # Test fonksiyonunu çağır
# Modeli kaydet
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model kaydedildi")

# Burası sadece Windows'ta gereklidir
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

