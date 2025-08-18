import os
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

BLURRED_IMGS_PATH = 'data_new/blr-img/images'
NORMAL_IMGS_PATH = 'data_new/nrm-img/images'

# Create a dataset with blurred images
def blur(images):
    blurred_imgs = [gaussian_filter(image, sigma=1) for image in images]
    return blurred_imgs

# Show images_aug in a grid
def show_grid(images, cols=4):
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def process_image(image_path, image_size):
    # Using OpenCV to read and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB
    img = cv2.resize(img, image_size)  # Resize image using OpenCV
    return img / 255.0  # Normalize to [0, 1]

# Save the blurred images to a new directory
def save_images(images, start_range, directory_path):
    curr_index = start_range
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for img in images:
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(directory_path, f"image_{curr_index}.png"))
        curr_index += 1

def create_blurred_images(root_dir, img_dirs, image_size=(500, 375),num_imgs=100,batch_size=50,max_workers=4):
    img_number = 0
    for img_dir in img_dirs:
        last_i_batch = 0
        for i in range(0,num_imgs+1,batch_size+1):
            if i > 0:
                # Complete path for class
                class_dir = f'{root_dir}/{img_dir}'
                # Process images in batches
                img_paths = os.listdir(class_dir)[last_i_batch:i]
                # Get list of image paths
                image_paths = [os.path.join(class_dir, filename) for filename in img_paths if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                print(image_paths)
            
                # Use ProcessPoolExecutor to process images in parallel
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Map process_image to the image paths
                    images = list(tqdm(executor.map(process_image, image_paths, [image_size]*len(image_paths)), total=len(image_paths)))
                images = np.array(images)
                # Ensure images are of type uint8 before augmentation
                images = (images * 255).astype(np.uint8)  # Convert normalized images back to uint8
                blurred_imgs = blur(images)

                # Save the blurred images to a new directory
                img_number = img_number + last_i_batch
                save_images(blurred_imgs, img_number, os.path.join(BLURRED_IMGS_PATH))
                save_images(images, img_number, os.path.join(NORMAL_IMGS_PATH))
                
                print(f"Processed images from {last_i_batch} to {i} for {img_dir}")
                print(f"Current number of images: {img_number}")
                last_i_batch = i

            




