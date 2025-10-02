from mnist_model import MNISTClassifier
import os

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

print("=" * 60)
print("MNIST DIGIT CLASSIFICATION PROJECT")
print("=" * 60)

# Initialize the classifier
classifier = MNISTClassifier()

# Load and prepare the dataset
(x_train, y_train), (x_test, y_test) = classifier.load_data()

# Build the CNN model
classifier.build_model()

# Display model architecture
print("\nModel Architecture:")
classifier.model.summary()

# Train the model with 5 epochs for quick testing
classifier.train(x_train, y_train, x_test, y_test, epochs=5)

# Evaluate model performance
classifier.evaluate(x_test, y_test)

# Save the trained model
classifier.save_model('../models/mnist_model.h5')

# Generate visualizations
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

from visualize import plot_training_history, plot_predictions, plot_confusion_matrix

# Plot training history
plot_training_history(classifier.history.history)

# Plot sample predictions
plot_predictions(classifier.model, x_test, y_test, num_samples=10)

# Plot confusion matrix
plot_confusion_matrix(classifier.model, x_test, y_test, num_samples=1000)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)