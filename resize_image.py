from PIL import Image
import os

def resize_images(folder):
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img = img.resize((128, 128))  
            img.save(img_path)
            
resize_images('dataset/red')
resize_images('dataset/yellow')
resize_images('dataset/green')
resize_images('dataset/other')



