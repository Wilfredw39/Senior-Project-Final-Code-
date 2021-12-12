from typing import ChainMap
import numpy as np
import os
import cv2
import skimage
from skimage.transform import resize
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import segmentation_models as sm


APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # project abs path

SIZE_X = 128
SIZE_Y = 128
SIZE_Z = 1

CLASSIFICATION_MODEL_PATH = r"..\Models\Classification Models\MBLY-Net\experiments\exp#8\weights\0.933.h5"


SEGMENTATION_MODEL_PATH = r"..\Models\Segmentatoin Models\FPN-Binary model\experiments\exp#1\weights\First_Mulit_UNet.h5"
WEIGHTS_PATH = r"..\Models\Segmentatoin Models\FPN-Binary model\class_weights.npy"


DECODE_DICT = {0: "Glioma II", 1: "Glioma III", 2: "Glioma VI",
               3: "Meningioma", 4: "No tumor", 5: "Pituitary"}
LABEL_DESCRIPTION = {
    "Glioma II": r"Grade II diffuse gliomas, often referred as low-grade gliomas, are slow-growing tumors and hold a better prognosis than grade III-IV diffuse gliomas, which are high-grade gliomas, and progress more rapidly. Finding information about prognoses and survival rates is a personal decision.",
    "Glioma III": r"is considered a more malignant evolution of a previously lower grade astrocytoma, which has acquired more aggressive features, including a higher pace of growth and more invasion into the brain. Histologically, it displays a higher degree of cellular abnormalities, and evidence of cell proliferation (mitoses), in comparison to grade 2 tumors. Surgery is never considered curative for these tumors, and needs to be followed by radiation and almost always chemotherapy.",
    "Glioma VI": r"is the most malignant, aggressive and common (60%) form of astrocytomas. Histologically, it is characterized by very abnormal-appearing cells, proliferation, areas of dead tissue and formation of new vessels. GBM can present either as a malignant progression from a previously existing lower grade astrocytoma (usually in 10% of cases) or originate directly as a grade 4 tumor (90% of cases). The former scenario is most common in younger patients, while the latter is most common after age 60.",
    "Pituitary": r"Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too much of the hormones that regulate important functions of your body. Some pituitary tumors can cause your pituitary gland to produce lower levels of hormones.",
    "Meningioma": r"A meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. Specifically, the tumor forms on the three layers of membranes that are called meninges. These tumors are often slow-growing. As many as 90% are benign (not cancerous).",
    "No tumor": "The Model didn't detect any tumor."
}


def get_custom_obj():
    class_weights = np.load(WEIGHTS_PATH)
    dice_loss = sm.losses.DiceLoss(class_weights)
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics_obj = {"dice_loss_plus_1focal_loss": total_loss,
                   "iou_score": sm.metrics.IOUScore(threshold=0.5),
                   "f1-score": sm.metrics.FScore(threshold=0.5)}
    return metrics_obj


def imgResize(img):
    wt, ht = SIZE_X, SIZE_Y
    h, w = img.shape
    f = min(wt / w, ht / h)
    tx = (wt - w * f) / 2
    ty = (ht - h * f) / 2

    # map image into target image
    M = np.float32([[f, 0, tx], [0, f, ty]])
    target = np.ones([ht, wt]) * 255
    img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target,
                         borderMode=cv2.BORDER_TRANSPARENT)
    return img


def preprocessing(path):
    img = cv2.imread(path)[:, :, 0]
    img = imgResize(img)
    img = np.array(img).reshape(-1, SIZE_X, SIZE_Y)
    img = np.expand_dims(img, axis=3)
    img = img/255
    return img


def rgb_img(path):
    img = cv2.imread(path)[:, :, 0]
    img = imgResize(img)
    return np.stack((img,)*3, axis=-1)


def predict_class(tensor):
    model = load_model(CLASSIFICATION_MODEL_PATH)
    predictions = model.predict(tensor)
    predictions_encoded = np.argmax(predictions, axis=1)
    predictions = np.vectorize(DECODE_DICT.get)(predictions_encoded)
    return predictions[0]


def segment_tumor(tensor):
    custom_obj = get_custom_obj()
    model = load_model(SEGMENTATION_MODEL_PATH, custom_objects=custom_obj)
    predictions = (model.predict(tensor))
    tumor_mask = np.argmax(predictions, axis=3)[0, :, :]
    return tumor_mask


def Draw_Mask(mri, mask):
    import cv2 as cv
    sample = np.array(np.squeeze(mask), dtype=np.uint8)
    contours, hier = cv.findContours(
        sample, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    sample_over_gt = cv.drawContours(
        mri, contours, -1, [255, 0, 0], thickness=1)
    return sample_over_gt


def predict(path):
    target = os.path.join(APP_ROOT, 'static/Prediction_imgs')

    mir_file_name = os.path.split(path)[-1]
    pre_file_name = mir_file_name.replace("MRI", "Pred")
    prediction_file_path = os.path.join(target, pre_file_name)

    ten_img = preprocessing(path)
    colored_img = rgb_img(path)
    label = predict_class(ten_img)

    if label != "No tumor":
        mask = segment_tumor(ten_img)
        prediction_img = Draw_Mask(colored_img, mask)
        plt.imsave(prediction_file_path, prediction_img, cmap=cm.gray)
    else:
        plt.imsave(prediction_file_path, ten_img[0, :, :, 0], cmap=cm.gray)

    return {"label": label, "prediction_path": pre_file_name, "description": LABEL_DESCRIPTION[label]}
