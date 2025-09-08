from GAN_model.gan_training import GAN, train_wgan
from data_loader import DataLoaderCreator
from GAN_model.model_args import args

IMG_NRML_DIR = "data/gopro_deblur/sharp"
IMG_BLUR_DIR = "data/gopro_deblur/blur"
SAVED_MODEL_PATH = "results/saved_model"

if __name__=="__main__":
    model = GAN(SAVED_MODEL_PATH,"cuda")
    dtl = DataLoaderCreator(IMG_NRML_DIR, IMG_BLUR_DIR)
    train_wgan(model, dtl, args['save_by_epoch'], True)
 