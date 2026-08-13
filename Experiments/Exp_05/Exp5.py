import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical
# Load and prepare MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize and reshape
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
# Build CNN
model = Sequential([
 # First Convolutional Block
 Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
 MaxPooling2D((2, 2)),

 # Second Convolutional Block
 Conv2D(64, (3, 3), activation='relu'),
 MaxPooling2D((2, 2)),

 # Third Convolutional Block
 Conv2D(64, (3, 3), activation='relu'),

 # Fully Connected Layers
 Flatten(),
 Dense(64, activation='relu'),
 Dropout(0.5),
 Dense(10, activation='softmax')
])
# Compile
model.compile(
 optimizer=Adam(learning_rate=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy']
)
# Print summary
model.summary()
# Train
history = model.fit(
 X_train, y_train,
 epochs=20,
 batch_size=128,
 validation_split=0.1,
 verbose=1
)
# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
# Visualize training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid()
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.show()
# Visualize sample predictions
predictions = model.predict(X_test[:10])
predicted_labels = np.ar