from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import cv2
import json

ANCHOR_BOXES = 13
DIMS_PER_BOX = 5
IMAGE_SOURCE = './data/'

with open("./data.json", 'r') as f:
    DATA = json.loads(f.read())


xs = []
ys = []
for data in DATA:
    boxes = []
    image = cv2.imread(f"{IMAGE_SOURCE}/{data['dir']}/{data['clip']}/{data['file']}") / 255.
    img = cv2.resize(image, (490 // 4, 360 // 4))
    xs.append(np.array(img))
    for anchor in data['players']:
        # print(len(anchor))
        # print(anchor)
        # if image.shape == (360, 490, 3):
        boxes.append(1)
        print(anchor, image.shape)
        boxes.append(anchor[0] / (image.shape[0] / (490 // 4)) / (490 // 4))
        boxes.append(anchor[1] / (image.shape[1] / (360 // 4)) / (360 // 4))
        boxes.append(anchor[2] / (image.shape[0] / (490 // 4)) / (490 // 4))
        boxes.append(anchor[3] / (image.shape[1] / (360 // 4)) / (360 // 4))
        # else:
        #     boxes.append(1)
        #     boxes.append(anchor[0] / (image.shape[0] / 490) // (490 // 4))
        #     boxes.append(anchor[1] / (image.shape[1] / 360) // (360 // 4))
        #     boxes.append(anchor[2] / (image.shape[0] / 490) // (490 // 4))
        #     boxes.append(anchor[3] / (image.shape[1] / 360) // (360 // 4))
        # for value in anchor:
            # print(value)
        # boxes.append(value)
    if len(boxes) < ANCHOR_BOXES * DIMS_PER_BOX:
        for _ in range(ANCHOR_BOXES * DIMS_PER_BOX - len(boxes)):
            boxes.append(0)
    # if len(boxes) != 50:
    # if  np.array(boxes).shape  !=  (60,):
        # print(np.array(boxes).shape)
    
    ys.append(np.array(boxes))

print(ys)
print(len(xs), len(ys))
model = Sequential()
model.add(Conv2D(64, (3,  3), input_shape=(90, 122, 3), strides=(2, 2), activation="sigmoid"))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, (1,  1), activation="sigmoid"))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation="softmax"))
model.add(Dense(ANCHOR_BOXES * DIMS_PER_BOX, activation="sigmoid"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-5), metrics=['accuracy'])
model.fit(np.array(xs), np.array(ys), epochs=100, batch_size=8)
model.save("model.h5")
model.load_weights('model.h5')
image = cv2.imread('data/zYgYQAWrDmw/clip_30/01.png') / 255.
image = cv2.resize(image, (490 // 4, 360 // 4))
image = np.reshape(image, (-1, 90, 122, 3))
print(xs[0].shape, image.shape)
print(model.predict(image))
