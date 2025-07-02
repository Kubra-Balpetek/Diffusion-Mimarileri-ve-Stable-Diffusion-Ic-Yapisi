import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt



# Basit U-Net modeli
def unet_model(input_size=(128, 128, 3)):
    inputs = keras.layers.Input(input_size)
    
    # Encoder
    c1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D()(c1)
    
    c2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D()(c2)
    
    c3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D()(c3)
    
    # Bottleneck
    b = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = keras.layers.Conv2D(512, 3, activation='relu', padding='same')(b)
    
    # Decoder
    u3 = keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(b)
    concat3 = keras.layers.concatenate([u3, c3])
    c6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(concat3)
    c6 = keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c6)
    
    u2 = keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    concat2 = keras.layers.concatenate([u2, c2])
    c7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
    c7 = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c7)
    
    u1 = keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    concat1 = keras.layers.concatenate([u1, c1])
    c8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    c8 = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c8)
    
    outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(c8)
    
    model = keras.Model(inputs, outputs)
    return model

# Modeli oluştur
model = unet_model()

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Örnek segmentasyon için dummy veri (gerçek veri yerine)
# Gerçekte burada kendi datasetini kullanmalısın.
def load_sample_image(path, size=(128, 128)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

# Örnek kullanım

# Buraya segmentasyon yapmak istediğin resmin yolunu yaz
img_path = "Diffusion_Mimarileri/trafik.jpg"

image = load_sample_image(img_path)

# Model giriş boyutu (batch, yükseklik, genişlik, kanal)
input_img = np.expand_dims(image, axis=0)  # 1 adet görüntü

# Model tahmini (bu örnek için eğitim yok, sadece ileri geçiş)
seg_mask = model.predict(input_img)[0, :, :, 0]

# Segmentasyon maskesini 0-1 aralığında alıyoruz, eşikleyelim
seg_mask_bin = (seg_mask > 0.1).astype(np.uint8)

# Sonucu görselleştir
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title("Orijinal Görsel")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Segmentasyon Maskesi (Sürekli)")
plt.imshow(seg_mask, cmap='gray')
plt.axis('off')

"""
plt.subplot(1,3,3)
plt.title("Segmentasyon Maskesi (Eşiklenmiş)")
plt.imshow(seg_mask_bin, cmap='gray')
plt.axis('off')
"""
plt.show()


