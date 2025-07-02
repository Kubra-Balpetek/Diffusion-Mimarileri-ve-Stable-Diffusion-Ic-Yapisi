import numpy as np
from PIL import Image
import os

os.makedirs("Diffusion_Mimarileri/forward_noise_steps", exist_ok=True)

img = Image.open("Diffusion_Mimarileri/yemek.jpeg").convert("RGB").resize((256, 256))
img_np = np.array(img) / 255.0  # 0-1 arası normalize

num_steps = 10  
beta = 0.05     

for step in range(1, num_steps + 1):
    noise = np.random.normal(0, beta * step, img_np.shape)
    noisy_img = img_np + noise
    noisy_img = np.clip(noisy_img, 0, 1)  # 0-1 arası tut
    noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
    out_img = Image.fromarray(noisy_img_uint8)
    out_img.save(f"Diffusion_Mimarileri/forward_noise_steps/noise_step_{step}.png")
    print(f"Adım {step}: Diffusion_Mimarileri/forward_noise_steps/noise_step_{step}.png kaydedildi.")

print("Tüm adımlar tamamlandı.")
