import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load and prepare data
iris = load_iris()
X = iris.data
y = to_categorical(iris.target, 3) # One-hot encode
# Split data
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Build Multilayer Network
model = Sequential([
 Dense(64, activation='relu', input_dim=4),
 Dropout(0.2),
 Dense(32, activation='relu'),
 Dropout(0.2),
 Dense(16, activation='relu'),
 Dense(3, activation='softmax') # 3 classes
])
# Compile
model.compile(
 optimizer=Adam(learning_rate=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy']
)
# Print architecture
model.summary()
# Train
history = model.fit(
 X_train, y_train,
 epochs=200,
 batch_size=8,
 validation_split=0.2,
 verbose=1
)
# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
# Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss - Multilayer Network')
axes[0].legend()
axes[0].grid()
# Plot accuracy
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy - Multilayer Network')
axes[1].legend()
axes[1].gr
