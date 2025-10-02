import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def plot_training_history(history_dict):
    # Plot training and validation accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history_dict['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved to results/training_history.png")
    plt.show()


def plot_predictions(model, x_test, y_test, num_samples=10):
    # Make predictions on test samples
    predictions = model.predict(x_test[:num_samples])

    # Create subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        # Display image
        axes[i].imshow(x_test[i].reshape(28, 28), cmap='gray')

        # Get predicted and true labels
        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]
        confidence = predictions[i][pred_label] * 100

        # Color code: green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'

        axes[i].set_title(f'Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}',
                          color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('MNIST Predictions - Sample Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../results/prediction_samples.png', dpi=300, bbox_inches='tight')
    print("Prediction samples plot saved to results/prediction_samples.png")
    plt.show()


def plot_confusion_matrix(model, x_test, y_test, num_samples=1000):
    # Make predictions
    predictions = model.predict(x_test[:num_samples])
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = y_test[:num_samples]

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set labels
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - MNIST Classification', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to results/confusion_matrix.png")
    plt.show()