import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = tf.keras.Sequential([
    # Add convolutional layers with ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Flatten the output and add a dense layer with softmax activation
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Set up data augmentation for the validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Set the batch size and number of epochs

#1st run
# batch_size = 1
# epochs = 10

#2nd run
batch_size = 3
epochs = 10

# Load the training and validation data using the data generators
train_data = train_datagen.flow_from_directory(
    'D:\Sayyam AQI Model\Dataset\TRAIN',
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='categorical')

val_data = val_datagen.flow_from_directory(
    'D:\Sayyam AQI Model\Dataset\TEST',
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples//batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_data.samples//batch_size)

model_json = model.to_json()
with open("air_quality_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model
model.save('air_quality_model.h5')