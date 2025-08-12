from deblurrer_wgans import train_wgan

IMG_NRML_DIR = "data_new/nrm-img"
IMG_BLUR_DIR = "data_new/blr-img"

train_wgan(IMG_NRML_DIR, IMG_BLUR_DIR, 5, True)
 