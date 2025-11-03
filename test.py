import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from model import MYNet
import argparse

# Ana fonksiyonunuzu oluşturuyoruz
def main():
    # Komut satırı argümanlarını işleme
    parser = argparse.ArgumentParser(description='MNIST Rakam Tanıma Test Betiği')
    parser.add_argument('image_path', type=str, help='Test edilecek görselin yolu')
    args = parser.parse_args()

    # CUDA destekliyse GPU'yu, değilse CPU'yu kullan
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli yükleme
    model = MYNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    model.eval()  # Modeli değerlendirme moduna al

    # Kendi yüklediğiniz resmi alıyoruz
    img_path = args.image_path
    img = Image.open(img_path).convert('L')  # Resmi gri tonlamaya dönüştür
    img = ImageOps.invert(img) # Renkleri tersine çevir

    # Resmi uygun boyuta getiriyoruz (28x28)
    img = img.resize((28, 28))

    # Resmi tensor'a dönüştürüyoruz ve normalleştiriyoruz
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
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
