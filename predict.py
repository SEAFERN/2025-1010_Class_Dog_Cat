import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

if len(sys.argv) != 2:
    print("Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

model = tf.keras.models.load_model('image_classification_model.h5')
img_height, img_width = 150, 150
img = image.load_img(image_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)[0][0]
                                          
if predictions > 0.5:
    print(f"The image is classified as: Dog with confidence {predictions:.2f}")
else:
    print(f"The image is classified as: Cat with confidence {1 - predictions:.2f}")