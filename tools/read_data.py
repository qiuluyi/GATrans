import cv2
import numpy as np
from PIL import Image
import os

# p = './data_slice_DRIVE/label_train'
# p = './CHASE_new/big_img'
p = './CHASE/training/1st_manual'
imgs = os.listdir(p)
for i in range(len(imgs)):
    img = np.asarray(Image.open(os.path.join(p, imgs[i])).convert('L'))
    print(set(img.flatten()))
p = './DRIVE/training/images/21_training.tif'
img = np.asarray(Image.open(p))
img = np.stack((img[:, :, 1], img[:, :, 1], img[:, :, 1]), axis=-1)
print(img.shape)