import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

def normalize_image(image_path, max_width, max_height, output_folder):
    image = cv2.imread(image_path)
    output_path = "{}.png".format(image_path[:-4])
    output_path = os.path.join(output_folder,output_path.split("/")[-1])

    im_height, im_width, channels = image.shape
    x_factor = float(max_width)/float(im_width)
    y_factor = float(max_height)/float(im_height)

    min_factor = min(x_factor, y_factor)
    if min_factor < 1.0:
        image = cv2.resize(image, (0,0), fx=min_factor, fy=min_factor)

    im_height, im_width, channels = image.shape
    final_image = np.zeros((max_height, max_width, channels))
    final_image[:,:] = (255, 255, 255)

    # Compute variable possible positions
    y_variation = max_height - im_height
    x_variation = max_width - im_width

    x_pos = 0
    y_pos = 0
    if x_variation > 0:
        x_pos = np.random.randint(x_variation, size=1)[0]
    if y_variation > 0:
        y_pos = np.random.randint(y_variation, size=1)[0]

    # Insert image into big one
    final_image[y_pos:im_height+y_pos,x_pos:im_width+x_pos] = image
    cv2.imwrite(output_path, final_image)

MAX_WIDTH = 400
MAX_HEIGHT = 40
NUM_MAX = 1000
OUTPUT_FOLDER = "data/normalized_{}_{}/".format(MAX_WIDTH, MAX_HEIGHT)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

files = glob.glob("data/original/*.png")
size = len(files)

with tqdm(total=size) as pbar:
    for i, image_path in enumerate(files):
        if i < NUM_MAX:
            normalize_image(image_path, MAX_WIDTH, MAX_HEIGHT, OUTPUT_FOLDER)
        pbar.update()
    pbar.close()


    
