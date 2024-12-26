import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import random
from torchvision import transforms
from torchvision.transforms import functional as F

def transform_JPEGcompression(image, compress_range = (30, 100)):
    assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
    image = Image.fromarray(image)
    jpegcompress_value = random.randint(compress_range[0], compress_range[1])
    out = BytesIO()
    image.save(out, 'JPEG', quality=jpegcompress_value)
    out.seek(0)
    rgb_image = Image.open(out)
    
    return np.array(rgb_image)


def transform_gaussian_noise(img, mean = 0.0, var = 10.0):
    img = np.array(img)
    height, width, channels = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma,(height, width, channels))
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)
    
    return noisy


def _motion_blur(img, kernel_size):
    kernel_v = np.zeros((kernel_size, kernel_size)) 
    kernel_h = np.copy(kernel_v) 
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    
    # Normalize. 
    kernel_v /= kernel_size 
    kernel_h /= kernel_size 
    if np.random.uniform() > 0.5:
        blurred = cv2.filter2D(img, -1, kernel_v) 
    else:
        blurred = cv2.filter2D(img, -1, kernel_h)

    return blurred


def _unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
        
    return sharpened


def _increase_contrast(img, kernel_size):
    lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=np.random.uniform(0.001, 4.0), tileGridSize=(kernel_size,kernel_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final


def transform_random_blur(img):
    img = np.array(img)
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19])
    
    if flag >= 0.75:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), np.random.uniform(0.0, 2.0))
    elif flag >= 0.5:
        img = _motion_blur(img, kernel_size)
    elif flag >= 0.4:
        img = cv2.blur(img, (kernel_size, kernel_size))
    elif flag >= 0.2:
        img = _unsharp_mask(img, kernel_size = kernel_size)
    elif flag >= 0.0:
        img = _increase_contrast(img, kernel_size)
        
    return img


def transform_adjust_gamma(image, lower = 0.2, upper = 2.0):
    image = np.array(image)
    gamma = np.random.uniform(lower, upper)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)


def transform_to_gray(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return gray


def transform_crop(image, crop_range = 0.3):
    height, width, _ = image.shape
    x1 = random.randint(0, int(crop_range/2*width))
    y1 = random.randint(0, int(crop_range/2*height))
    x2 = random.randint(int((1-crop_range/2)*width), width)
    y2 = random.randint(int((1-crop_range/2)*height), height)
    crop = cv2.resize(image[y1:y2, x1:x2], (width, height))
    
    return crop


def transform_resize(image, resize_range = (24, 112), target_size = 112):
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = np.random.randint(resize_range[0], resize_range[1])
    inter = random.choice([cv2.INTER_AREA,
                        cv2.INTER_CUBIC,
                        cv2.INTER_LANCZOS4,
                        cv2.INTER_LINEAR,
                        cv2.INTER_NEAREST])
    resize_image = cv2.resize(image, (resize_value, resize_value), interpolation = inter)
    
    return cv2.resize(resize_image, (target_size, target_size), interpolation = inter)


def transform_color_jiter(sample, brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1):
    sample = Image.fromarray(sample)
    photometric = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            photometric.get_params(photometric.brightness, photometric.contrast,
                                                  photometric.saturation, photometric.hue)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            sample = F.adjust_brightness(sample, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            sample = F.adjust_contrast(sample, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            sample = F.adjust_saturation(sample, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            sample = F.adjust_hue(sample, hue_factor)

    return np.array(sample)


def transform_resize_padding(sample, target_size = 224):
    sample = np.array(sample)
    height, width, _ = sample.shape
    scale = min(target_size/height, target_size/width)
    new_width = int(width*scale)
    new_height = int(height*scale)
    img = np.zeros((target_size, target_size, 3), dtype = np.uint8)
    img[:new_height, :new_width] = cv2.resize(sample, (new_width, new_height))
    
    return img


def random_augment(sample):
    sample = transform_resize_padding(sample, target_size = 224)

    # Input is RGB image
    if np.random.uniform() < 0.2:
        sample = transform_crop(sample)

    # Blur augmentation
    if np.random.uniform() < 0.3:
        sample = transform_random_blur(sample)

    # Downscale augmentation
    if np.random.uniform() < 0.2:
        sample = transform_resize(sample, resize_range = (32, 224), target_size = 224)

    # Color augmentation
    if np.random.uniform() < 0.5:
        sample = transform_adjust_gamma(sample)
    if np.random.uniform() < 0.3:
        sample = transform_color_jiter(sample)

    # Noise augmentation
    if np.random.uniform() < 0.15:
        sample = transform_gaussian_noise(sample, mean = 0.0, var = 10.0)

    # Gray augmentation
    if np.random.uniform() < 0.2:
        sample = transform_to_gray(sample)
    
    # JPEG augmentation
    if np.random.uniform() < 0.3:
        sample = transform_JPEGcompression(sample, compress_range = (20, 100))

    return sample

def just_resize(sample):
    sample = transform_resize_padding(sample, target_size = 224)
    return sample