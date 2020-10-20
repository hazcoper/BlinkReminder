import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import time

vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)

blink_counter = 0
think_blink_count = 0
think_noblink_count = 0
while rval:
    # cv2.imshow("preview", frame)
    rval, frame = vc.read()   #read the frame
    # key = cv2.waitKey(20)

    image_array = np.asarray(frame)
    image_array = cv2.resize(image_array, dsize=size, interpolation=cv2.INTER_CUBIC)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    if prediction[0][1] > 0.97:
        #Means the program thinks there was a blink, but I want three blinks consecutive to count as a blink
        think_blink_count += 1
    else:
        think_blink_count = 0
        think_noblink_count += 1

    
    if think_blink_count > 2 and think_noblink_count > 20:
        blink_counter += 1
        print(f"There was a blink! - {blink_counter}");
        think_blink_count = 0
        think_noblink_count = 0
    



# run the inference
