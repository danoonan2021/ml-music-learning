import os
import numpy as np
from tensorflow import keras
import librosa
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


DATA_PATH = 'vocab.json'
EPOCHS = 64

# load the processed data from json and split the data for the model
def split_train_test_data(features, labels):
    
    print(len(features), len(labels))

    # Split the data into training, validation, and testing parts
    permutations = np.random.permutation(300)
    features = np.array(features)[permutations]
    labels = np.array(labels)[permutations]

    features_train = features[0:180]
    labels_train = labels[0:180]

    features_validation = features[180:240]
    labels_validation = features[180:240]

    features_test = features[240:300]
    labels_test = labels[240:300]

    return features_train, labels_train, features_validation, labels_validation,features_test, labels_test

def save_model(model, accuracy):

    model_name = ('model_' + str(EPOCHS) + "_" + str(accuracy[1]))

    model.save(model_name)

with open(DATA_PATH, 'r') as fp:
    data = json.load(fp)
    
# convert to numpy array
features = np.array(data["mfcc"])
labels = np.array(data["labels"])

# splits the data into 80% train, 20% test
# features_train, labels_train, features_validation, labels_validation, features_test, labels_test = split_train_test_data(features, labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

# build network topology
model = keras.Sequential([
# input layer
keras.layers.Flatten(input_shape=(features.shape[1], features.shape[2])),
# 1st dense layer
keras.layers.Dense(512, activation='relu'),
# 2nd dense layer
keras.layers.Dense(256, activation='relu'),
# 3rd dense layer
keras.layers.Dense(64, activation='relu'),
# output layer
keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_summ = model.summary()
print(f"Model Summary: \n---------\n{model_summ}")

# train model
history = model.fit(features_train.tolist(), labels_train.tolist(), verbose=1, batch_size=32, epochs=EPOCHS)

stats = model.evaluate(x=features_test.tolist(), y=labels_test.tolist(), verbose=0)

print("----------")
print(f"Model Evaluation stats (loss, accuracy): {stats}")
# Show model statistics
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

# Show the plot in a window that stays open
plt.show()

save_model(model, stats)