import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None

    def load_data(self):
        # Load MNIST dataset
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Normalize pixel values to range [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")

        return (x_train, y_train), (x_test, y_test)

    def build_model(self):
        # Build CNN architecture
        print("\nBuilding CNN model...")

        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),

            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # Third convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),

            # Fully connected layers
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile model with optimizer and loss function
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("Model built successfully!")
        return model

    def train(self, x_train, y_train, x_test, y_test, epochs=5):
        # Train the model
        print(f"\nTraining model for {epochs} epochs...")

        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=128,
            verbose=1
        )

        print("Training completed!")
        return self.history

    def evaluate(self, x_test, y_test):
        # Evaluate model performance on test set
        print("\nEvaluating model...")
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Test Accuracy: {test_acc * 100:.2f}%')
        print(f'Test Loss: {test_loss:.4f}')

        return test_loss, test_acc

    def save_model(self, filepath='models/mnist_model.h5'):
        # Save trained model to disk
        print(f"\nSaving model to {filepath}...")
        self.model.save(filepath)
        print("Model saved successfully!")