# [ai] #03 - AI Deep Learning & Neural Networks
![deep-learning](https://raw.githubusercontent.com/oujisan/OuVault/main/img/py-ai.png)

## Introduction to Deep Learning

---

Deep Learning adalah subset dari Machine Learning yang menggunakan artificial neural networks dengan banyak layer (hidden layers) untuk mempelajari representasi data yang kompleks. Teknik ini terinspirasi dari cara kerja otak manusia dengan neuron-neuron yang saling terhubung.

### Mengapa Deep Learning Powerful?

- **Automatic Feature Learning**: Tidak perlu manual feature engineering
- **Handle Complex Data**: Gambar, suara, teks, video
- **Scalability**: Performa meningkat dengan data yang lebih besar
- **State-of-the-art Results**: Hasil terbaik di banyak domain AI

## Neural Networks Fundamentals

---

### 1. Perceptron (Single Neuron)

Unit dasar dari neural network, menerima input dan menghasilkan output.

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def fit(self, X, y):
        # Initialize weights dan bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Training
        for i in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Forward pass
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                
                # Update weights dan bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear_output)
        return predictions

# Contoh penggunaan: AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate output

perceptron = Perceptron()
perceptron.fit(X, y)

predictions = perceptron.predict(X)
print("AND Gate Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Expected: {y[i]}, Predicted: {predictions[i]}")
```

### 2. Multi-Layer Perceptron (MLP)

Neural network dengan satu atau lebih hidden layers.

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate non-linearly separable data
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), 
                   activation='relu', 
                   solver='adam',
                   max_iter=1000, 
                   random_state=42)

# Train model
mlp.fit(X_train, y_train)

# Evaluate
train_accuracy = mlp.score(X_train, y_train)
test_accuracy = mlp.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Original Data')

plt.subplot(1, 3, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.title('Training Data')

plt.subplot(1, 3, 3)
y_pred = mlp.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.title('Predictions')

plt.tight_layout()
plt.show()
```

## Deep Learning dengan TensorFlow/Keras

---

### 1. Setup dan Basic Neural Network

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set random seed untuk reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load dan preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values ke range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data untuk fully connected layer
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

print(f"Training data shape: {x_train_flat.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")
```

### 2. Building Deep Neural Network

```python
# Create model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
history = model.fit(x_train_flat, y_train,
                   epochs=10,
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
```

### 3. Visualizing Training Process

```python
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions
predictions = model.predict(x_test_flat[:10])
predicted_classes = np.argmax(predictions, axis=1)

# Visualize predictions
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'True: {y_test[i]}, Pred: {predicted_classes[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## Convolutional Neural Networks (CNN)

---

CNN adalah arsitektur yang sangat efektif untuk mengolah data gambar.

### 1. CNN untuk Image Classification

```python
# Reshape data untuk CNN (menambahkan channel dimension)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile CNN
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train CNN
cnn_history = cnn_model.fit(x_train_cnn, y_train,
                           epochs=5,
                           batch_size=32,
                           validation_split=0.2,
                           verbose=1)

# Evaluate CNN
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_test_accuracy:.4f}")
```

### 2. Understanding CNN Layers

```python
def visualize_conv_layers(model, image, layer_names):
    """Visualize feature maps dari convolutional layers"""
    
    # Create model yang output intermediate layers
    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(image.reshape(1, 28, 28, 1))
    
    plt.figure(figsize=(15, 10))
    
    for layer_idx, activation in enumerate(activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        n_cols = n_features // 4
        
        for i in range(min(16, n_features)):  # Show max 16 features
            plt.subplot(len(activations), 4, layer_idx * 4 + i % 4 + 1)
            plt.imshow(activation[0, :, :, i], cmap='viridis')
            plt.title(f'Layer {layer_idx+1}, Feature {i+1}')
            plt.axis('off')
            
            if i >= 15:
                break
    
    plt.tight_layout()
    plt.show()

# Visualize untuk satu gambar
sample_image = x_test[0]
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2']
# visualize_conv_layers(cnn_model, sample_image, layer_names)
```

## Recurrent Neural Networks (RNN)

---

RNN cocok untuk data sequential seperti teks, time series, dan speech.

### 1. Simple RNN untuk Text Classification

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = [
    "Saya sangat suka film ini",
    "Film ini sangat bagus dan menghibur",
    "Cerita yang menarik dan akting yang hebat",
    "Film ini membosankan sekali",
    "Tidak recommend film ini",
    "Saya tidak suka dengan jalan ceritanya"
]

labels = [1, 1, 1, 0, 0, 0]  # 1: positive, 0: negative

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

print("Original texts:")
for i, text in enumerate(texts):
    print(f"{i}: {text}")

print(f"\nSequences: {sequences}")
print(f"Padded sequences: {padded_sequences}")

# Build RNN model
rnn_model = keras.Sequential([
    layers.Embedding(1000, 16, input_length=10),
    layers.SimpleRNN(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train (dengan data kecil ini hanya untuk demo)
rnn_model.fit(padded_sequences, labels, epochs=50, verbose=0)

# Test predictions
predictions = rnn_model.predict(padded_sequences)
print("\nPredictions:")
for i, (text, pred) in enumerate(zip(texts, predictions)):
    sentiment = "Positive" if pred[0] > 0.5 else "Negative"
    print(f"{text} -> {sentiment} ({pred[0]:.3f})")
```

### 2. LSTM untuk Time Series Prediction

```python
# Generate sample time series data
def create_time_series():
    time = np.arange(0, 100, 0.1)
    amplitude = np.sin(time) + 0.5 * np.sin(time * 3) + np.random.normal(0, 0.1, len(time))
    return amplitude

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Prepare data
ts_data = create_time_series()
seq_length = 20
X, y = create_sequences(ts_data, seq_length)

# Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape untuk LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
lstm_model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    layers.LSTM(50, return_sequences=False),
    layers.Dense(25),
    layers.Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
lstm_history = lstm_model.fit(X_train, y_train,
                             epochs=20,
                             batch_size=32,
                             validation_split=0.2,
                             verbose=1)

# Make predictions
y_pred = lstm_model.predict(X_test)

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
plt.title('Time Series Prediction dengan LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()
```

## Advanced Deep Learning Concepts

---

### 1. Transfer Learning

Menggunakan model yang sudah dilatih untuk task baru.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classifier
transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes untuk contoh
])

transfer_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

print("Transfer Learning Model Summary:")
transfer_model.summary()
```

### 2. Autoencoders

Neural network untuk dimensionality reduction dan data compression.

```python
# Build Autoencoder
input_dim = 784  # 28x28 pixels

# Encoder
encoder = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu')
])

# Decoder
decoder = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

# Autoencoder
autoencoder = keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder (menggunakan data MNIST yang sudah ada)
autoencoder.fit(x_train_flat, x_train_flat,
                epochs=10,
                batch_size=32,
                validation_data=(x_test_flat, x_test_flat),
                verbose=1)

# Test reconstruction
reconstructed = autoencoder.predict(x_test_flat[:10])

# Visualize original vs reconstructed
plt.figure(figsize=(15, 6))
for i in range(10):
    # Original
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 3. Generative Adversarial Networks (GAN) - Konsep

```python
# Simple GAN architecture (conceptual)
class SimpleGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='tanh'),
            layers.Reshape((28, 28, 1))
        ])
        return model
    
    def build_discriminator(self):
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def generate_fake_samples(self, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_images = self.generator.predict(noise)
        return generated_images

# Instantiate GAN
gan = SimpleGAN()
print("GAN Generator Summary:")
gan.generator.summary()
print("\nGAN Discriminator Summary:")
gan.discriminator.summary()
```

## Optimization dan Best Practices

---

### 1. Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

def create_model(neurons=64, dropout_rate=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Dense(neurons, activation='relu', input_shape=(784,)),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create KerasClassifier
keras_classifier = KerasClassifier(model=create_model, epochs=5, batch_size=32, verbose=0)

# Define parameter grid
param_grid = {
    'model__neurons': [32, 64, 128],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'model__learning_rate': [0.001, 0.01, 0.1]
}

# Random search
random_search = RandomizedSearchCV(keras_classifier, param_grid, 
                                 n_iter=5, cv=3, verbose=1, random_state=42)

# Note: Ini akan memakan waktu lama untuk dataset besar
# random_search.fit(x_train_flat[:1000], y_train[:1000])  # Menggunakan subset untuk demo
```

### 2. Callbacks untuk Training Optimization

```python
# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train dengan callbacks
model_with_callbacks = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model_with_callbacks.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

history_callbacks = model_with_callbacks.fit(
    x_train_flat, y_train,
    epochs=50,  # Epochs tinggi, tapi early stopping akan menghentikan jika perlu
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

## Practical Project: Image Classification with CNN

---

```python
# Complete project: Fashion MNIST classification
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess data
x_train_fashion = x_train_fashion.astype('float32') / 255.0
x_test_fashion = x_test_fashion.astype('float32') / 255.0

# Add channel dimension
x_train_fashion = x_train_fashion.reshape(-1, 28, 28, 1)
x_test_fashion = x_test_fashion.reshape(-1, 28, 28, 1)

# Build improved CNN
fashion_cnn = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile
fashion_cnn.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train
fashion_history = fashion_cnn.fit(
    x_train_fashion, y_train_fashion,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
fashion_test_loss, fashion_test_accuracy = fashion_cnn.evaluate(x_test_fashion, y_test_fashion, verbose=0)
print(f"\nFashion MNIST Test Accuracy: {fashion_test_accuracy:.4f}")

# Make predictions dan visualize
fashion_predictions = fashion_cnn.predict(x_test_fashion[:20])
fashion_predicted_classes = np.argmax(fashion_predictions, axis=1)

plt.figure(figsize=(15, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_test_fashion[i].reshape(28, 28), cmap='gray')
    true_label = class_names[y_test_fashion[i]]
    pred_label = class_names[fashion_predicted_classes[i]]
    confidence = np.max(fashion_predictions[i])
    
    color = 'green' if y_test_fashion[i] == fashion_predicted_classes[i] else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.2f})', 
              color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Key Takeaways

---

1. **Deep Learning is powerful**: Mampu menangani data kompleks dengan minimal preprocessing
2. **Architecture matters**: Pilih arsitektur yang sesuai dengan jenis data dan masalah
3. **Data is king**: Model yang baik membutuhkan data yang berkualitas dan cukup banyak
4. **Regularization penting**: Dropout, BatchNormalization mencegah overfitting
5. **Hyperparameter tuning**: Eksperimen dengan berbagai parameter untuk hasil optimal
6. **Monitor training**: Gunakan callbacks dan validation untuk mengoptimalkan training

Pada file selanjutnya, kita akan membahas Computer Vision dan aplikasinya dalam dunia nyata!