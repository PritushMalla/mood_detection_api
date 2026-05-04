# Import required libraries
import os                      # for working with folders/files
import numpy as np            # for numerical operations

# Import deep learning tools from Keras
from keras.models import Sequential                 # allows building model layer-by-layer
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # helps load & preprocess images


# -----------------------------
# 📁 DATASET PATHS
# -----------------------------

# Folder where training images are stored
train_dir = "data/train"

# Folder where validation/test images are stored
val_dir = "data/test"


# -----------------------------
# 🔄 IMAGE PREPROCESSING
# -----------------------------

# This rescales pixel values from (0–255) → (0–1)
# Helps model learn better and faster
train_datagen = ImageDataGenerator(rescale=1./255)

# Same preprocessing for validation data
val_datagen = ImageDataGenerator(rescale=1./255)


# Load training images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,                # path to training data
    target_size=(48, 48),     # resize all images to 48x48 pixels
    batch_size=64,            # number of images processed at once
    color_mode="grayscale",   # convert images to black & white (1 channel)
    class_mode="categorical"  # because we have multiple emotion classes
)

# Load validation images (same settings)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)


# -----------------------------
# 🧠 BUILDING CNN MODEL
# -----------------------------

# Create a sequential model (layer-by-layer)
model = Sequential()


# First Convolution Layer
# 32 filters → detects basic features like edges, lines
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))

# MaxPooling reduces image size but keeps important info
model.add(MaxPooling2D(pool_size=(2,2)))


# Second Convolution Layer
# 64 filters → detects more complex patterns (eyes, mouth)
model.add(Conv2D(64, (3,3), activation='relu'))

# Again reduce size
model.add(MaxPooling2D(pool_size=(2,2)))


# Flatten converts 2D feature maps → 1D vector
# Needed before feeding into Dense layers
model.add(Flatten())


# Fully connected layer (decision making)
model.add(Dense(128, activation='relu'))

# Dropout randomly turns off 50% neurons during training
# Helps prevent overfitting (model memorizing instead of learning)
model.add(Dropout(0.5))


# Output layer
# 7 neurons → one for each emotion class
# Softmax → gives probability for each class
model.add(Dense(7, activation='softmax'))


# -----------------------------
# ⚙️ COMPILING MODEL
# -----------------------------

model.compile(
    optimizer='adam',                  # how the model learns (adjusts weights)
    loss='categorical_crossentropy',   # used for multi-class classification
    metrics=['accuracy']               # measure performance
)


# -----------------------------
# 🎯 TRAINING MODEL
# -----------------------------

model.fit(
    train_generator,                   # training data
    validation_data=val_generator,     # validation data
    epochs=30                         # number of times model sees full dataset
)


# -----------------------------
# 💾 SAVE MODEL
# -----------------------------

# Save trained model into a file
model.save("emotion_model.h5")

print("✅ Model saved as emotion_model.h5")