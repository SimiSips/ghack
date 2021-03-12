# Import libaries
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import splitfolders
from tensorflow.keras.callbacks import TensorBoard
import time 

# Create a tensorboard callback
NAME = "DeepHospitalCNN-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Use split folders library to split into train and test set
splitfolders.ratio("dataset", "new_data", ratio=(0.9, 0.1))

# Load Images
train_dir = "new_data/train"
val_dir = "new_data/val"

train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                vertical_flip=True,
                                horizontal_fip=True,
                                fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                   class_mode='nearest',
                                                   target_size=(150, 150))

val_generator = val_datagen.flow_from_directory(directory=val_dir,
                                               class_mode='nearest',
                                               target_size=(150, 150))
                                            
# Check the classes index
print(train_generator.class_indices)
print(val_generator.class_indices)


# Define a machine learning model
model = tf.keras.models.Sequential([
    # first convolution with 64 units
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    # first 2x2 maxpooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # second convolution with 64 units
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    # second 2x2 maxpooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # third convolution with 128 units
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    # third 2x2 maxpooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # fourth convolution with 128 units
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    # fourth 2x2 maxpooling
    tf.keras.layers.MaxPooling2D(2, 2),
    # flatten into a DNN
    tf.keras.layers.Flatten(),
    # Dense layer with 512 units
    tf.keras.layers.Dense(512, activation="relu"),
    # output layer with 4 classes
    tf.keras.layers.Dense(4, activation="softmax")
])

# Compile the model
model.compile(optimizer="rmsprop", 
            loss="categorical_crossentropy", 
            metrics=[tf.keras.layers.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy'])

# Summary of the model
model.summary()

# Train the model
model.fit(train_generator, epochs=25, validation_data=val_generator, callbacks=[tensorboard])

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(val_generator)

f1score = 2*((precision * recall) / (precision + recall))
print("Accuray: " + str(accuracy))
print("Loss: " + str(loss))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(f1score))