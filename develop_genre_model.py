import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import math

DATA_PATH = 'vocab.json'
EPOCHS = 64
NUM_OF_GENRES = 10

def save_model(model, accuracy):
    # Generate a custom name for the model
    model_name = ("model_" + str(EPOCHS) + "_" + str(math.floor(accuracy[1] * 100)) + str(math.floor(accuracy[0] * 100000)) + ".pb")
    # Save model with custom name
    model.save(model_name)

def build_model(input_shape):
    # instantiate model
    model = keras.Sequential()
    # add 1st convoluted input layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='valid'))
    model.add(keras.layers.MaxPooling2D(2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    # add 2nd convoluted layer
    model.add(keras.layers.Conv2D(128, (2,2), activation='relu', padding='valid'))
    model.add(keras.layers.MaxPooling2D(2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    # add 3rd convoluted layer
    model.add(keras.layers.Conv2D(128, (2,2), activation='relu', padding='valid'))
    model.add(keras.layers.MaxPooling2D(2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    # add hidden layer
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # add another hidden layer
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # output 
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def load_split_data(test_size=0.2, val_size=0.2):
    # Load data from json
    with open(DATA_PATH, 'r') as fp:
        data = json.load(fp)
    
    # Convert to numpy array
    features = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    # Splits the data into 60% train, 20% test, 20% validation
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
    features_train, features_val, labels_train, labels_val = train_test_split(features_train, labels_train, test_size=val_size, stratify=labels_train)

    # Add a new dimension to the input_shape (2D -> 3D)
    num_frames = features_train.shape[1]
    num_mfcc = features_train.shape[2]

    # Reshape your data to add a channel dimension (in this case, 1 channel)
    features_train = features_train.reshape(features_train.shape[0], num_frames, num_mfcc, 1)
    features_val = features_val.reshape(features_val.shape[0], num_frames, num_mfcc, 1)
    features_test = features_test.reshape(features_test.shape[0], num_frames, num_mfcc, 1)

    return features_train, features_test, features_val, labels_train, labels_test, labels_val

def plot_model_stats(stats):
    print("----------")
    print(f"Model Evaluation stats (loss, accuracy): {stats}")

    # Show the accuracy and loss plots in separate windows
    figure, axis = plt.subplots(1, 2, figsize=(12, 6))

    # For Accuracy Plot
    axis[0].plot(history.history['accuracy'])
    axis[0].set_title("Model Accuracy")
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Accuracy')

    # For Loss Plot
    axis[1].plot(history.history['loss'])
    axis[1].set_title("Model Loss")
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()


# split the data into 60% train, 20% test, 20% validation
features_train, features_test, features_val, labels_train, labels_test, labels_val = load_split_data(0.2, 0.2)

labels_train = to_categorical(labels_train, num_classes=NUM_OF_GENRES)
labels_val = to_categorical(labels_val, num_classes=NUM_OF_GENRES)
labels_test = to_categorical(labels_test, num_classes=NUM_OF_GENRES)

# build network topology
model = build_model((features_train.shape[1], features_train.shape[2], 1))

earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, verbose=1)

# compile model
model.compile(optimizer='adam',
                loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
# model summary
model_summ = model.summary()
print(f"Model Summary: \n---------\n{model_summ}")

# train model
history = model.fit(features_train.tolist(), labels_train.tolist(), validation_data=(features_val.tolist(), labels_val.tolist()), verbose=1, batch_size=32, epochs=EPOCHS, callbacks=[earlystop])

# Get the loss and accuracy of the model
stats = model.evaluate(x=features_test.tolist(), y=labels_test.tolist(), verbose=0)
plot_model_stats(stats)

# Save the model
save_model(model, stats)