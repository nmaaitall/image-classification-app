import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import pathlib


def plot_training_history(history_dict, save_path='../results/cats_dogs_training_history.png'):
    # Plot training and validation accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Cats vs Dogs - Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history_dict['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Cats vs Dogs - Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


def plot_sample_predictions(model, data_dir, num_samples=8):
    # Load some validation images
    data_dir = pathlib.Path(data_dir)
    val_dir = data_dir / 'validation'

    # Get sample images
    cat_images = list((val_dir / 'cats').glob('*.jpg'))[:num_samples // 2]
    dog_images = list((val_dir / 'dogs').glob('*.jpg'))[:num_samples // 2]
    sample_images = cat_images + dog_images

    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, img_path in enumerate(sample_images):
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]

        # Determine predicted class
        if prediction > 0.5:
            pred_class = 'Dog'
            confidence = prediction * 100
        else:
            pred_class = 'Cat'
            confidence = (1 - prediction) * 100

        # Get true class from filename
        true_class = 'Cat' if 'cat' in str(img_path).lower() else 'Dog'

        # Color code: green if correct, red if wrong
        color = 'green' if pred_class == true_class else 'red'

        # Display image
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {pred_class} ({confidence:.1f}%)\nTrue: {true_class}',
                          color=color, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Cats vs Dogs - Sample Predictions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../results/cats_dogs_predictions.png', dpi=300, bbox_inches='tight')
    print("Prediction samples plot saved to results/cats_dogs_predictions.png")
    plt.show()


def plot_training_summary(history_dict):
    # Create comprehensive training summary
    fig = plt.figure(figsize=(16, 10))

    # Accuracy over time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history_dict['accuracy'], label='Training', linewidth=2, marker='o')
    ax1.plot(history_dict['val_accuracy'], label='Validation', linewidth=2, marker='s')
    ax1.set_title('Model Accuracy Progress', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss over time
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history_dict['loss'], label='Training', linewidth=2, marker='o')
    ax2.plot(history_dict['val_loss'], label='Validation', linewidth=2, marker='s')
    ax2.set_title('Model Loss Progress', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Final metrics
    ax3 = plt.subplot(2, 2, 3)
    final_train_acc = history_dict['accuracy'][-1] * 100
    final_val_acc = history_dict['val_accuracy'][-1] * 100
    metrics = ['Training\nAccuracy', 'Validation\nAccuracy']
    values = [final_train_acc, final_val_acc]
    colors = ['#2ecc71', '#3498db']
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Final Model Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 100])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Training info
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    info_text = f"""
    TRAINING SUMMARY

    Total Epochs: {len(history_dict['accuracy'])}

    Final Training Accuracy: {final_train_acc:.2f}%
    Final Validation Accuracy: {final_val_acc:.2f}%

    Final Training Loss: {history_dict['loss'][-1]:.4f}
    Final Validation Loss: {history_dict['val_loss'][-1]:.4f}

    Best Validation Accuracy: {max(history_dict['val_accuracy']) * 100:.2f}%
    (Epoch {np.argmax(history_dict['val_accuracy']) + 1})
    """

    ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Cats vs Dogs Classification - Complete Training Report',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('../results/cats_dogs_complete_report.png', dpi=300, bbox_inches='tight')
    print("Complete training report saved to results/cats_dogs_complete_report.png")
    plt.show()