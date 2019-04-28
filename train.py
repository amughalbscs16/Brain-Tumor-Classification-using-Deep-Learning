import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
#print(os.listdir('./data/'))
noExamples = len(os.listdir('./data'))
data = np.zeros((noExamples,512,512))
labels = np.zeros((noExamples,1))
for itemId, file in zip(range(0,noExamples),os.listdir('./data')):
    with h5py.File('./data/'+file, 'r') as f:
        data[itemId,:,:] = f['cjdata']['image']
        labels[itemId] = f['cjdata']['label']

x_train,x_test = data[0:700,:,:]/255,data[700:,:,:]/255
y_train,y_test = labels[0:700], labels[700:]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(512, 512)),
  tf.keras.layers.Conv2D(32,(3,3),(1,2),padding="valid"),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print()
model.evaluate(x_test, y_test)

#print(data[0,200:210,200:210])
#plt.imshow(data[0,:,:])
#plt.show()
#print(data.shape)
#print(labels.shape)
#print(labels)
