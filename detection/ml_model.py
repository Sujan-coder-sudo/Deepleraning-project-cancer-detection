import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from sklearn.model_selection import train_test_split

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def preprocess_with_generator(base_dir, img_size=(50, 50), batch_size=32):
    """
    Use ImageDataGenerator to preprocess and load data in batches.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Reserve 20% of the data for validation
    )
    
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model():
    base_dir = './data'
    img_size = (50, 50)
    batch_size = 32

    print("Starting data preprocessing...")
    
    try:
        train_gen, val_gen = preprocess_with_generator(base_dir, img_size, batch_size)
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    model = create_model((img_size[0], img_size[1], 3))
    
    print("Starting model training...")
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    
    test_loss, test_acc = model.evaluate(val_gen, verbose=2)
    print(f"Test accuracy: {test_acc}")
    
    model.save('cancer_detection_model.h5')
    print("Model saved successfully")

def predict_cancer(image_path):
    model = keras.models.load_model('cancer_detection_model.h5')
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(50, 50))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    prediction = model.predict(img_array)
    
    return "Malignant" if prediction[0][0] > 0.5 else "Benign"

if __name__ == "__main__":
    train_model()
