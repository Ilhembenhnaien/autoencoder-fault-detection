import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# --- Load data ---
df = pd.read_excel("C:/Users/21653/Downloads/Autoencoder/datatrainauto.xlsx")

# Drop the 'label' column if it exists
if 'label' in df.columns:
    df = df.drop(columns=['label'])

data = df.values  # Convert to numpy array

# --- Parameters ---
sequence_length = 5
n_features = data.shape[1]

# --- Build flattened sequences ---
sequences = []
for i in range(len(data) - sequence_length + 1):
    seq = data[i:i + sequence_length].flatten()  # concatenate all theta values over 5 time steps
    sequences.append(seq)
sequences = np.array(sequences)

# --- Normalization ---
scaler = StandardScaler()
sequences_scaled = scaler.fit_transform(sequences)

# --- Autoencoder model ---
input_dim = sequences_scaled.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),   # input_dim = 30
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')  # output = 30
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Training ---
history = model.fit(
    sequences_scaled, sequences_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# --- Plotting the loss curve ---
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve (Autoencoder)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")  # Save the figure
plt.show()

# --- Compute anomaly threshold ---
reconstructions = model.predict(sequences_scaled)
mse_train = np.mean(np.square(sequences_scaled - reconstructions), axis=1)
threshold = np.mean(mse_train) + 3 * np.std(mse_train)
print(f"Automatic MSE Threshold: {threshold:.6f}")

# --- Save outputs ---
output_dir = "C:/Users/21653/Downloads/Autoencoder/final"
os.makedirs(output_dir, exist_ok=True)
model.save(f"{output_dir}/autoencoder_dense_model.h5")
joblib.dump(scaler, f"{output_dir}/scaler_dense.save")
np.save(f"{output_dir}/mse_threshold.npy", threshold)
