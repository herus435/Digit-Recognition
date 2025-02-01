# Digit-Recognition
MNIST Rakam Tanıma için  Convolutional Neural Networks (CNN) Modeli  
Bu kod, PyTorch kullanarak MNIST rakam veri setini sınıflandırmak için (CNN) modeli uygulamaktadır.  

## Gereksinimler
Python 3.x (Bu projede 3.12 kullanılmıştır!)  
PyTorch kütüphanesi (pytorch)(2.5.1)  
Numpy kütüphanesi (numpy)(1.26.4)  

## Nasıl Çalıştırılır
Gerekli kütüphaneleri yükleyin: pip install torch torchvision numpy  
Kodu bir Python dosyasına kaydedin (örn. mnist.py)  
Kodu çalıştırın: python mnist.py  
Kod, MNIST veri setini indirecek, modeli eğitecek ve mnist_model.pth dosyasına kaydedecektir.  
Eğittiğiniz modeli kullanmak için (test.py) dosyasındaki kodu kullanıp kendi örneklerinizi test edebilirsiniz.  
Yapmanız gereken (img_path = 'your_image.png') kısmına kendi png dosya yolunuzu eklemektir.  


## Öneriler
Yüklediğiniz resim dosyasında rakamın arka planının temiz ve beyaz olmasına,  
Rakam harici birşeyin bulunmamasına,  
Kenar boşluklarının yeterince bırakılmış olmasına dikkat edin lütfen.  
