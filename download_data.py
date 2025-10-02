import tensorflow as tf
import os
import pathlib


def download_and_prepare_data():
    # Download the dataset from TensorFlow
    print("Downloading Cats vs Dogs dataset...")

    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

    # Download and extract
    data_dir = tf.keras.utils.get_file(
        'cats_and_dogs.zip',
        origin=dataset_url,
        extract=True
    )

    data_dir = pathlib.Path(data_dir).parent / 'cats_and_dogs_filtered'

    print(f"\nData downloaded to: {data_dir}")

    # Show data structure
    train_dir = data_dir / 'train'
    validation_dir = data_dir / 'validation'

    # Count images
    train_cats = len(list((train_dir / 'cats').glob('*.jpg')))
    train_dogs = len(list((train_dir / 'dogs').glob('*.jpg')))
    val_cats = len(list((validation_dir / 'cats').glob('*.jpg')))
    val_dogs = len(list((validation_dir / 'dogs').glob('*.jpg')))

    print("\nDataset Statistics:")
    print(f"Training cats: {train_cats}")
    print(f"Training dogs: {train_dogs}")
    print(f"Validation cats: {val_cats}")
    print(f"Validation dogs: {val_dogs}")
    print(f"Total images: {train_cats + train_dogs + val_cats + val_dogs}")

    return str(data_dir)


if __name__ == "__main__":
    data_path = download_and_prepare_data()
    print(f"\nData ready at: {data_path}")