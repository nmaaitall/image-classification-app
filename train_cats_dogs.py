from model import CatsVsDogsClassifier
import tensorflow as tf
import pathlib
import zipfile
import os

print("=" * 60)
print("DOWNLOADING CATS VS DOGS DATASET")
print("=" * 60)

# Download the dataset
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file(
    'cats_and_dogs.zip',
    origin=dataset_url,
    cache_dir='.',
    cache_subdir='data'
)

print(f"Downloaded to: {zip_path}")

# Extract the zip file
data_dir = pathlib.Path(zip_path).parent / 'cats_and_dogs_filtered'

# If directory doesn't exist, extract it
if not data_dir.exists():
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_to = pathlib.Path(zip_path).parent
        zip_ref.extractall(extract_to)
    print("Extraction complete!")
else:
    print("Data already extracted")

# Verify the structure
train_dir = data_dir / 'train'
val_dir = data_dir / 'validation'

print(f"\nData directory: {data_dir}")
print(f"Train directory exists: {train_dir.exists()}")
print(f"Validation directory exists: {val_dir.exists()}")

if not train_dir.exists() or not val_dir.exists():
    print("\nERROR: Required directories not found!")
    print("Contents of data directory:")
    for item in data_dir.parent.iterdir():
        print(f"  - {item}")
    exit(1)

print("\n" + "=" * 60)
print("CATS VS DOGS CLASSIFICATION PROJECT")
print("=" * 60)

# Initialize the cats vs dogs classifier
classifier = CatsVsDogsClassifier(img_size=150)

# Load and prepare the cats and dogs dataset
classifier.prepare_data(data_dir)

# Build the CNN model for cats vs dogs classification
classifier.build_model()

# Display model architecture
print("\nCats vs Dogs Model Architecture:")
classifier.model.summary()

# Train the cats vs dogs model (15 epochs)
print("\nTraining Cats vs Dogs classifier...")
classifier.train(epochs=15)

# Evaluate cats vs dogs model performance
classifier.evaluate()

# Save the trained cats vs dogs model
classifier.save_model('../models/cats_vs_dogs_model.h5')

print("\n" + "=" * 60)
print("CATS VS DOGS PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)