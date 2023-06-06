# 내가 그린 손글씨 인식하기.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

im = Image.open('number.png')
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert('L'))
print(img.shape)
print(img)

plt.imshow(img, cmap='Greys')
plt.show()

data = img.reshape([1, 784])
data = data / 255.0

mymodel = tf.keras.models.load_model("tf22model.hdf5")
pred = mymodel.predict(data)
print('pred : ', pred)
print('pred : ', np.argmax(pred,1))
