import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def preprocess_data(base_dir, img_size=(50, 50)):
    images = []
    labels = []
    
    print(f"Base directory: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                img_path = os.path.join(root, file)
                print(f"Processing file: {img_path}")
                
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    
                    # Determine label based on directory name
                    if 'class1' in root.lower() or '1' in file.split('_')[2]:
                        labels.append(1)
                    else:
                        labels.append(0)
                    
                    print(f"Successfully processed {img_path}")
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    continue
    
    if not images:
        raise ValueError("No valid images found in the dataset")
    
    print(f"Total images processed: {len(images)}")
    return np.array(images), np.array(labels)

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

def train_model():
    data_dir = './data'
    
    print("Starting data preprocessing...")
    print(f"Looking for images in: {os.path.abspath(data_dir)}")
    
    try:
        images, labels = preprocess_data(data_dir)
        print(f"Found {len(images)} images with {sum(labels)} positive cases")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    model = create_model(X_train.shape[1:])
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    print("Starting model training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
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

