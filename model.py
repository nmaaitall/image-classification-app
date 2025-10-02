import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib


class CatsVsDogsClassifier:
    def __init__(self, img_size=150):
        self.img_size = img_size
        self.model = None
        self.history = None
        self.train_generator = None
        self.validation_generator = None

    def prepare_data(self, data_dir):
        # Convert to Path object and ensure it's a valid path
        data_dir = pathlib.Path(data_dir)

        # Verify the directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        train_dir = data_dir / 'train'
        validation_dir = data_dir / 'validation'

        # Verify subdirectories exist
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not validation_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

        print("Preparing data generators...")
        print(f"Train directory: {train_dir}")
        print(f"Validation directory: {validation_dir}")

        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation set
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        # Create generators - convert Path to string
        self.train_generator = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='binary'
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            str(validation_dir),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='binary'
        )

        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")

        return self.train_generator, self.validation_generator

    def build_model(self):
        # Build CNN model
        print("\nBuilding CNN model...")

        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(self.img_size, self.img_size, 3)),
            keras.layers.MaxPooling2D((2, 2)),

            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # Fourth convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),

            # Flatten and dense layers
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("Model built successfully!")
        return model

    def train(self, epochs=20):
        # Train the model
        print(f"\nTraining model for {epochs} epochs...")

        # Callbacks for better training
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )

        # Train
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        print("Training completed!")
        return self.history

    def evaluate(self):
        # Evaluate model on validation set
        print("\nEvaluating model...")
        val_loss, val_acc = self.model.evaluate(self.validation_generator, verbose=0)
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")

        return val_loss, val_acc

    def save_model(self, filepath='../models/cats_vs_dogs_model.h5'):
        # Save trained model
        print(f"\nSaving model to {filepath}...")
        self.model.save(filepath)
        print("Model saved successfully!")