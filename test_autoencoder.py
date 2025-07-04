import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

# --- Paramètres ---
sequence_length = 5
model_dir = "C:/Users/21653/Downloads/Autoencoder/final"

# --- Chargement du scaler ---
scaler = joblib.load(f"{model_dir}/scaler_dense.save")
expected_features = scaler.mean_.shape[0]
n_features = expected_features // sequence_length

# --- Chargement des données de test ---
df_test = pd.read_excel("C:/Users/21653/Downloads/Autoencoder/test_ilhem.xlsx")
print("Nombre de lignes dans le fichier test :", len(df_test))

# --- Extraction des données et labels ---
data_test = df_test.iloc[:, :n_features].values
labels = df_test.iloc[:, -1].values  # La dernière colonne contient les labels (0=normal, 1=anomalie)

# --- Création des séquences et des labels associés ---
sequences_test = []
labels_seq_test = []

for i in range(len(data_test) - sequence_length + 1):
    seq = data_test[i:i + sequence_length].flatten()
    sequences_test.append(seq)
    
    # Label = 1 si au moins un des points dans la séquence est anormal
    window_labels = labels[i:i + sequence_length]
    label_seq = 1 if np.any(window_labels == 1) else 0
    labels_seq_test.append(label_seq)

sequences_test = np.array(sequences_test)
labels_seq_test = np.array(labels_seq_test)

if sequences_test.shape[0] == 0:
    print("Pas assez de données pour créer des séquences.")
    exit()

# --- Normalisation ---
sequences_test_scaled = scaler.transform(sequences_test)

# --- Chargement du modèle et du seuil ---
autoencoder = load_model(f"{model_dir}/autoencoder_dense_model.h5", compile=False)
threshold = np.load(f"{model_dir}/mse_threshold.npy")

# --- Reconstruction ---
reconstructions = autoencoder.predict(sequences_test_scaled)
mse = np.mean(np.square(sequences_test_scaled - reconstructions), axis=1)
anomalies = mse > threshold

# --- Affichage des résultats ---
print("\nRésultats de détection :")
print("start_index\tmse\t\tanomaly\t\tend_index")
for i in range(len(mse)):
    print(f"{i}\t\t{mse[i]:.8f}\t{anomalies[i]}\t\t{i + sequence_length - 1}")

# --- Matrice de confusion ---
cm = confusion_matrix(labels_seq_test, anomalies, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- Tableau de performance ---
precision = precision_score(labels_seq_test, anomalies)
recall = recall_score(labels_seq_test, anomalies)
f1 = f1_score(labels_seq_test, anomalies)
accuracy = accuracy_score(labels_seq_test, anomalies)

df_metrics = pd.DataFrame({
    "Threshold": [threshold],
    "Precision": [precision],
    "Recall": [recall],
    "F1-measure": [f1],
    "Accuracy": [accuracy]
})

print("\nTableau de performance :")
print(df_metrics)
df_metrics.to_csv("performance_table_result.csv", index=False)

