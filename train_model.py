import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Input


# Dataset path
train_dir = "chest_xray/train"
val_dir = "chest_xray/test"
test_dir = "chest_xray/test"

# Image size
IMG_SIZE = 224
BATCH_SIZE = 32

# Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize
    rotation_range=20,        # rotate
    zoom_range=0.2,           # zoom
    horizontal_flip=True      # flip
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load Images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("Classes:", train_generator.class_indices)

model = Sequential([
    Input(shape=(224,224,3)),
    Conv2D(32, (3,3), activation='relu'),


    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)
model.save("medical_model.h5")
print("Model saved successfully!")
model.summary()