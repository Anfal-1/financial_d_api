"""
Script to create sample autoencoder model and scaler for testing
This creates files compatible with the fraud detection API
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def generate_transaction_data(n_samples=10000, n_features=30):
    """Generate synthetic transaction data"""
    print(f"Generating {n_samples} samples with {n_features} features...")
    
    # Generate normal transactions (95% of data)
    n_normal = int(n_samples * 0.95)
    n_fraud = n_samples - n_normal
    
    # Normal transactions - centered around 0 with small variance
    normal_data = np.random.normal(0, 1, (n_normal, n_features))
    
    # Add some correlation between features for normal transactions
    for i in range(n_features - 1):
        normal_data[:, i+1] += 0.3 * normal_data[:, i] + np.random.normal(0, 0.5, n_normal)
    
    # Fraudulent transactions - different distribution
    fraud_data = np.random.normal(0, 1, (n_fraud, n_features))
    
    # Make fraud data more extreme and with different patterns
    fraud_data *= np.random.uniform(2, 4, (n_fraud, n_features))
    fraud_data += np.random.normal(0, 2, (n_fraud, n_features))
    
    # Add some outlier patterns to fraud data
    outlier_mask = np.random.random((n_fraud, n_features)) < 0.3
    fraud_data[outlier_mask] *= np.random.uniform(3, 6, np.sum(outlier_mask))
    
    # Labels (0 = normal, 1 = fraud)
    normal_labels = np.zeros(n_normal)
    fraud_labels = np.ones(n_fraud)
    
    # Combine data
    all_data = np.vstack([normal_data, fraud_data])
    all_labels = np.hstack([normal_labels, fraud_labels])
    
    # Shuffle the data
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    all_labels = all_labels[indices]
    
    print(f"Generated {n_normal} normal and {n_fraud} fraudulent transactions")
    return all_data, all_labels, normal_data

def create_autoencoder(input_dim):
    """Create autoencoder model"""
    print(f"Creating autoencoder with input dimension: {input_dim}")
    
    # Encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu')  # Bottleneck layer
    ])
    
    # Decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(input_dim, activation='linear')  # Output layer
    ])
    
    # Combine encoder and decoder
    autoencoder = tf.keras.Sequential([encoder, decoder])
    
    # Compile the model
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder

def main():
    # Generate synthetic data
    all_data, all_labels, normal_data = generate_transaction_data(n_samples=10000, n_features=30)
    
    # Create and fit scaler on all data (normal + fraud)
    print("Creating and fitting scaler...")
    scaler = StandardScaler()
    scaled_all_data = scaler.fit_transform(all_data)
    
    # Use only normal data for training autoencoder
    normal_indices = all_labels == 0
    normal_scaled = scaled_all_data[normal_indices]
    
    print(f"Training data shape: {normal_scaled.shape}")
    
    # Split normal data for training and validation
    train_data, val_data = train_test_split(normal_scaled, test_size=0.2, random_state=42)
    
    # Create autoencoder
    autoencoder = create_autoencoder(normal_scaled.shape[1])
    
    # Print model summary
    print("\nAutoencoder Architecture:")
    autoencoder.summary()
    
    # Train the autoencoder
    print("\nTraining autoencoder...")
    history = autoencoder.fit(
        train_data, train_data,
        epochs=100,
        batch_size=64,
        validation_data=(val_data, val_data),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    )
    
    # Save the model
    model_path = 'model/autoencoder_model.h5'
    autoencoder.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save the scaler
    scaler_path = 'model/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    
    # Test on normal data
    normal_test_scaled = scaler.transform(normal_data[:100])
    normal_reconstructions = autoencoder.predict(normal_test_scaled, verbose=0)
    normal_errors = np.mean(np.square(normal_test_scaled - normal_reconstructions), axis=1)
    
    # Test on fraud data
    fraud_indices = all_labels == 1
    fraud_data_sample = all_data[fraud_indices][:100]
    fraud_test_scaled = scaler.transform(fraud_data_sample)
    fraud_reconstructions = autoencoder.predict(fraud_test_scaled, verbose=0)
    fraud_errors = np.mean(np.square(fraud_test_scaled - fraud_reconstructions), axis=1)
    
    print(f"\nReconstruction Error Statistics:")
    print(f"Normal transactions - Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
    print(f"Fraud transactions - Mean: {np.mean(fraud_errors):.6f}, Std: {np.std(fraud_errors):.6f}")
    
    # Suggest threshold
    threshold_95 = np.percentile(normal_errors, 95)
    threshold_99 = np.percentile(normal_errors, 99)
    
    print(f"\nSuggested thresholds:")
    print(f"95th percentile of normal errors: {threshold_95:.6f}")
    print(f"99th percentile of normal errors: {threshold_99:.6f}")
    print(f"Current API threshold: 0.01")
    
    # Calculate performance metrics at different thresholds
    thresholds = [0.005, 0.01, threshold_95, threshold_99, 0.05]
    
    print(f"\nPerformance at different thresholds:")
    print(f"{'Threshold':<12} {'Normal FP Rate':<15} {'Fraud Detection Rate':<20}")
    print("-" * 50)
    
    for thresh in thresholds:
        normal_fp_rate = np.mean(normal_errors > thresh)
        fraud_detection_rate = np.mean(fraud_errors > thresh)
        print(f"{thresh:<12.6f} {normal_fp_rate:<15.3f} {fraud_detection_rate:<20.3f}")
    
    print(f"\nModel and scaler created successfully!")
    print(f"Files created:")
    print(f"- {model_path}")
    print(f"- {scaler_path}")
    print(f"\nYou can now run the Flask API with these files.")

if __name__ == "__main__":
    main()
