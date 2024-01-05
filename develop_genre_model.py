import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


DATA_PATH = 'vocab.json'
EPOCHS = 64

def save_model(model, accuracy):

    model_name = ('model_' + str(EPOCHS) + "_" + str(accuracy[1]))

    model.save(model_name)

def build_model(input_shape):
    # instantiate model
    model = keras.Sequential()

    # add 1st convoluted input layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # add 2nd convoluted layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # add 3rd convoluted layer
    model.add(keras.layers.Conv2D(64, (2,2), activation='sigmoid'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # add hidden layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
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
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)
    features_train, features_val, labels_train, labels_val = train_test_split(features_train, labels_train, test_size=val_size)

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

    # Show model statistics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy/loss')
    plt.ylabel('accuracy(blue), accuracy(yellow)')
    plt.xlabel('epoch')

    # Show the plot in a window that stays open
    plt.show()


# split the data into 60% train, 20% test, 20% validation
features_train, features_test, features_val, labels_train, labels_test, labels_val = load_split_data(0.2, 0.2)

# build network topology
model = build_model((features_train.shape[1], features_train.shape[2], 1))

# compile model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
# model summary
model_summ = model.summary()
print(f"Model Summary: \n---------\n{model_summ}")

# train model
history = model.fit(features_train.tolist(), labels_train.tolist(), validation_data=(features_val.tolist(), labels_val.tolist()), verbose=1, batch_size=32, epochs=EPOCHS)

# Get the loss and accuracy of the model
stats = model.evaluate(x=features_test.tolist(), y=labels_test.tolist(), verbose=0)
plot_model_stats(stats)

# Save the model
save_model(model, stats)