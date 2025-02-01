import torch
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Ana fonksiyonunuzu oluşturuyoruz
def main():
    # Modeli yükleyeceğiz
    class MYNet(nn.Module):
        def __init__(self):
            super(MYNet, self).__init__()
            self.conv = nn.Conv2d(1, 10, 5)
            self.conv1 = nn.Conv2d(10, 20, 5)
            self.conv1_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv(x), 2))
            x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # CUDA destekliyse GPU'yu, değilse CPU'yu kullan
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli yükleme
    model = MYNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()  # Modeli değerlendirme moduna al

    # Kendi yüklediğiniz resmi alıyoruz
    img_path = 'your_image.png'
    img = Image.open(img_path).convert('L')  # Resmi gri tonlamaya dönüştür

    # Resmi uygun boyuta getiriyoruz (28x28)
    img = img.resize((28, 28))

    # Resmi tensor'a dönüştürüyoruz
    transform = ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)  # 1x1x28x28 boyutunda olmalı

    # Modelin tahmin yapması
    output = model(img_tensor)
    prediction = output.argmax(dim=1, keepdim=True).item()

    print(f'Prediction: {prediction}')

    # Görselleştiriyoruz
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted: {prediction}')
    plt.show()

# Bu kısım sadece Windows'ta gereklidir
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support() 
    main()
