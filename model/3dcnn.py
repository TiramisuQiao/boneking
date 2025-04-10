from tensorflow.python.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

#Build a 3D convolutional neural network model
def get_model(width=60, height=60, depth=60):
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=8, kernel_size=3, activation="elu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=4, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Conv3D(filters=2, kernel_size=3, activation="elu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=2,padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="elu")(x)
    x = layers.Dense(units=64, activation="elu")(x)
    x = layers.Dense(units=32, activation="elu")(x)
    outputs = layers.Dense(units=1, activation="linear")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


#1. 3D CNN model for elastic modulus predicting

# Generate random data for matrix and E
num_samples = 100  # Number of samples to generate
matrix_shape = (num_samples, 60, 60, 60)
matrix = np.random.rand(*matrix_shape)
E_values = np.random.rand(num_samples) * 100  # Random elastic modulus values

X = matrix.reshape(len(E_values), 60, 60, 60, 1)
y = E_values

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build model.
model = get_model(width=60, height=60, depth=60)

#Training
optimizer = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=["mean_absolute_error"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # Reduced patience for faster demonstration
mc = ModelCheckpoint("model/3dCNN_E_random.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
print("\n--- Training model for Elastic Modulus (with random data) ---")
history_e = model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=50, callbacks=[es,mc])


#2. 3D CNN model for yield strength predicting

# Generate random data for matrix and yield
num_samples_yield = 100  # Number of samples to generate
matrix_yield = np.random.rand(*matrix_shape) # Use the same shape
yield_values = np.random.rand(num_samples_yield) * 50 # Random yield strength values

X2 = matrix_yield.reshape(len(yield_values), 60, 60, 60, 1)
y2 = yield_values

#Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=1)
# Build model.
model2 = get_model(width=60, height=60, depth=60)
#Training
optimizer2 = keras.optimizers.Adam(learning_rate=0.005)
model2.compile(optimizer=optimizer2, loss='mean_squared_error', metrics=["mean_absolute_error"])
es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) # Reduced patience
mc2 = ModelCheckpoint("model/3dCNN_Y_random.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
print("\n--- Training model for Yield Strength (with random data) ---")
history_y = model2.fit(X_train2, y_train2, validation_data=(X_test2, y_test2),  batch_size=16, epochs=50, callbacks=[es2,mc2])

print("\nTraining with random data completed. Check 'model' directory for saved models.")