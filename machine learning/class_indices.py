# Import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Load images
train_dir = "new_data/train"
val_dir = "new_data/val"

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                     class_mode="categorical",
                                                     target_size=(150, 150))

val_generator = val_datagen.flow_from_directory(directory=val_dir,
                                                     class_mode="categorical",
                                                     target_size=(150, 150))

# Check class indices
print(train_generator.class_indices)
print(val_generator.class_indices)