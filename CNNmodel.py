import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- CONFIGURATION DES CHEMINS ---
# Remplacez par les chemins réels sur votre PC si nécessaire
TRAIN_DIR = r'C:\Users\PC\Desktop\train' 
TEST_DIR = r'C:\Users\PC\Desktop\test'
IMG_SIZE = (80, 80)
BATCH_SIZE = 32

# --- 1. GÉNÉRATEURS DE DONNÉES (LOCAL) ---
# Normalisation et Augmentation pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Utilise 20% des images d'entraînement pour la validation
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset='validation'
)

# --- 2. CONSTRUCTION DU MODÈLE ---
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(80, 80, 1)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    AveragePooling2D(pool_size=(3, 3)),
    Dropout(0.5),
    
    Flatten(),
    Dense(90, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. ENTRAÎNEMENT ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

print("Début de l'entraînement...")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# --- 4. ÉVALUATION FINALE ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Prédictions
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)
y_true = test_generator.classes

# Affichage des résultats
print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_true, y_pred, target_names=['Fermé', 'Ouvert']))

# Matrice de Confusion
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fermé', 'Ouvert'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confusion")
plt.show()