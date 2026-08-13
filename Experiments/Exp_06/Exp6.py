import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import VGG16, ResNet50, MobileNetV2
from keras.applications.vgg16 import preprocess_input
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Resize for VGG16 (requires 224×224)
from tensorflow.image import resize
X_train_resized = resize(X_train, (224, 224))
X_test_resized = resize(X_test, (224, 224))
# Normalize
X_train_resized = preprocess_input(X_train_resized)
X_test_resized = preprocess_input(X_test_resized)
# Split training data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
 X_train_resized, y_train, test_size=0.2, random_state=42
)
print("="*60)
print("TRANSFER LEARNING APPROACH")
print("="*60)
# Load pre-trained VGG16
base_model = VGG16(
 weights='imagenet',
 include_top=False,
 input_shape=(224, 224, 3)
)
# Freeze base model layers
base_model.trainable = False
# Add custom top layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model_transfer = Model(inputs=base_model.input, outputs=predictions)
# Compile
model_transfer.compile(
 optimizer=Adam(learning_rate=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy']
)
print(model_transfer.summary())
# Train (transfer learning)
print("\nTraining with transfer learning...")
history_transfer = model_transfer.fit(
 X_train_split, y_train_split,
 epochs=20,
 batch_size=32,
 validation_data=(X_val_split, y_val_split),
 verbose=1
)
# Evaluate
test_loss_transfer, test_acc_transfer = model_transfer.evaluate(
 X_test_resized, y_test, verbose=0
)
print(f"Transfer Learning Test Accuracy: {test_acc_transfer:.4f}")
print("\n" + "="*60)
print("TRAINING FROM SCRATCH (For Comparison)")
print("="*60)
# Build similar model from scratch
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
model_scratch = Sequential([
 Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
 MaxPooling2D((2, 2)),
 Conv2D(64, (3, 3), activation='
        
    