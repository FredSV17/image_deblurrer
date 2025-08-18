from preprocess.blur_generator import create_blurred_images

root_dir = "data_new/raw-img"
img_dirs = ["cane","cavallo","elefante","farfalla","gallina","gatto","mucca","pecora","ragno","scoiattolo"]

image_data = create_blurred_images(root_dir,img_dirs, num_imgs=200, batch_size=50, show_img=False)
