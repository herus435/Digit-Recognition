# Digit-Recognition
MNIST Rakam Tanıma için Evrişimli Sinir Ağları (CNN) Modeli
Bu kod, PyTorch kullanarak MNIST rakam veri setini sınıflandırmak için bir CNN modeli uygular.

## Gereksinimler
- Python 3.x
- Gerekli kütüphaneler `requirements.txt` dosyasında listelenmiştir.

## Kurulum
Gerekli kütüphaneleri yüklemek için proje kök dizininde aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```

## Nasıl Çalıştırılır

### 1. Modeli Eğitme
Modeli eğitmek ve `mnist_model.pth` dosyasını oluşturmak için aşağıdaki komutu çalıştırın:
```bash
python Eğitim.py
```
Bu komut, MNIST veri setini indirecek, modeli 20 epoch boyunca eğitecek ve eğitilmiş modelin ağırlıklarını kaydedecektir.

### 2. Kendi Görselinizi Test Etme
Eğitilmiş modeli kullanarak kendi rakam görselinizi test etmek için `test.py` betiğini kullanabilirsiniz. Komut satırından test etmek istediğiniz görselin yolunu belirtmeniz yeterlidir:
```bash
python test.py <path_to_your_image.png>
```
Örneğin:
```bash
python test.py testPhotos/test_digit_3.png
```
Betiği çalıştırdığınızda, modelin tahmini komut satırına yazdırılacak ve görselin kendisi bir `matplotlib` penceresinde gösterilecektir.

## Test Görselleri İçin Öneriler
- Yüklediğiniz resim dosyasında rakamın arka planının temiz ve **beyaz** olmasına dikkat edin.
- Görselde rakam dışında bir nesne bulunmamalıdır.
- Rakamın etrafında yeterince kenar boşluğu bırakılmış olmalıdır.

**Not:** Kod, beyaz arka plan üzerine siyah rakamlı görsellerle en iyi sonucu verecek şekilde ayarlanmıştır. Eğer farklı formatta bir görsel kullanıyorsanız, `test.py` dosyasındaki görüntü işleme adımlarını düzenlemeniz gerekebilir.
