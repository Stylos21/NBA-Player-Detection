import json
from keras.optimizers import Adam
import cv2
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential

ANCHOR_BOXES = 13
DIMS_PER_BOX = 5
IMAGE_SOURCE = './data/'

xs = []
ys = []
model = Sequential()
model.add(Conv2D(64, (3,  3), input_shape=(90, 122, 3), strides=(2, 2), activation="sigmoid"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(32, (1,  1), activation="sigmoid"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="softmax"))
model.add(Dense(ANCHOR_BOXES * DIMS_PER_BOX, activation="sigmoid"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-5), metrics=['accuracy'])

model.load_weights('model.h5')
image = cv2.imread('getty_475927916_86483.jpg') / 255.
img_to_display = cv2.resize(image, (490, 360))
img = cv2.resize(image, (490 // 4, 360 // 4))
img = np.reshape(img, (-1, 90, 122, 3))
pred = model.predict(img)[0]
print(pred)
i = 1
while i < len(pred):
    box = pred[i]
    print(pred[i-1])
    if pred[i-1] > 0.91:
        cv2.rectangle(img_to_display,(int(pred[i] * 490), int(pred[i + 1] * 360)),(int((pred[i] * 490) +( pred[i + 2] * 490)), int((pred[i + 1] * 360) + (pred[i + 3] * 360))),(0,0,255),5)
    i += 5
    # print(box)
# if pred[0][0] > 0.5:
cv2.imshow('HSV image', img_to_display); cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)


print(pred[0])
