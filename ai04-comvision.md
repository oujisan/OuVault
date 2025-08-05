 # [ai] #04 - AI Computer Vision 
![computer-vision](https://raw.githubusercontent.com/oujisan/OuVault/main/img/py-ai.png)
## Introduction to Computer Vision

---

Computer Vision adalah bidang AI yang memungkinkan komputer untuk "melihat" dan memahami konten visual seperti gambar dan video. Teknologi ini meniru cara kerja sistem visual manusia untuk menginterpretasi dan menganalisis informasi visual.

### Aplikasi Computer Vision di Dunia Nyata:

- **Autonomous Vehicles**: Self-driving cars
- **Medical Imaging**: Diagnosis penyakit dari X-ray, MRI
- **Security**: Face recognition, surveillance systems
- **Retail**: Product recognition, inventory management
- **Social Media**: Photo tagging, content moderation
- **Manufacturing**: Quality control, defect detection

## Image Processing Fundamentals

---

### 1. Working with Images in Python

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Membaca gambar
def load_image(image_path):
    # Menggunakan OpenCV
    img_cv2 = cv2.imread(image_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    # Menggunakan PIL
    img_pil = Image.open(image_path)
    
    return img_cv2_rgb, img_pil

def display_image(image, title="Image"):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generate sample image untuk demo
def create_sample_image():
    # Membuat gambar sample dengan shapes
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Menambahkan shapes
    cv2.rectangle(img, (50, 50), (150, 100), (255, 0, 0), -1)  # Red rectangle
    cv2.circle(img, (100, 150), 30, (0, 255, 0), -1)  # Green circle
    cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 3)  # Blue line
    
    return img

# Membuat sample image
sample_img = create_sample_image()
display_image(sample_img, "Sample Image with Shapes")
```

### 2. Basic Image Operations

```python
def basic_image_operations(image):
    """Demonstrasi operasi dasar pada gambar"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)
    axes[0, 2].imshow(equalized, cmap='gray')
    axes[0, 2].set_title('Histogram Equalized')
    axes[0, 2].axis('off')
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    axes[1, 0].imshow(blurred)
    axes[1, 0].set_title('Gaussian Blur')
    axes[1, 0].axis('off')
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('Edge Detection')
    axes[1, 1].axis('off')
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    axes[1, 2].imshow(morphed, cmap='gray')
    axes[1, 2].set_title('Morphological Close')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gray, edges

# Apply basic operations
gray_img, edge_img = basic_image_operations(sample_img)
```

### 3. Image Filtering dan Enhancement

```python
def apply_filters(image):
    """Berbagai filter untuk image enhancement"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Mean filter
    mean_filtered = cv2.blur(image, (5, 5))
    axes[0, 1].imshow(mean_filtered, cmap='gray')
    axes[0, 1].set_title('Mean Filter')
    axes[0, 1].axis('off')
    
    # Median filter
    median_filtered = cv2.medianBlur(image, 5)
    axes[0, 2].imshow(median_filtered, cmap='gray')
    axes[0, 2].set_title('Median Filter')
    axes[0, 2].axis('off')
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    axes[0, 3].imshow(bilateral, cmap='gray')
    axes[0, 3].set_title('Bilateral Filter')
    axes[0, 3].axis('off')
    
    # Sobel X
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    axes[1, 0].imshow(np.abs(sobelx), cmap='gray')
    axes[1, 0].set_title('Sobel X')
    axes[1, 0].axis('off')
    
    # Sobel Y
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    axes[1, 1].imshow(np.abs(sobely), cmap='gray')
    axes[1, 1].set_title('Sobel Y')
    axes[1, 1].axis('off')
    
    # Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    axes[1, 2].imshow(np.abs(laplacian), cmap='gray')
    axes[1, 2].set_title('Laplacian')
    axes[1, 2].axis('off')
    
    # Sharpening
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharp)
    axes[1, 3].imshow(sharpened, cmap='gray')
    axes[1, 3].set_title('Sharpened')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# Apply filters ke grayscale image
apply_filters(gray_img)
```

## Feature Detection dan Description

---

### 1. Corner Detection

```python
def detect_corners(image):
    """Deteksi corners menggunakan Harris corner detector"""
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Harris corner detection
    corners_harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Dilate corner image to enhance corner points
    corners_harris = cv2.dilate(corners_harris, None)
    
    # Create result image
    result = image.copy()
    result[corners_harris > 0.01 * corners_harris.max()] = [255, 0, 0]
    
    # Shi-Tomasi corner detection
    corners_shi_tomasi = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners_shi_tomasi = np.int0(corners_shi_tomasi)
    
    result2 = image.copy()
    for corner in corners_shi_tomasi:
        x, y = corner.ravel()
        cv2.circle(result2, (x, y), 3, (0, 255, 0), -1)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(result)
    axes[1].set_title('Harris Corners')
    axes[1].axis('off')
    
    axes[2].imshow(result2)
    axes[2].set_title('Shi-Tomasi Corners')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return corners_harris, corners_shi_tomasi

# Detect corners
harris_corners, shi_tomasi_corners = detect_corners(sample_img)
```

### 2. SIFT (Scale-Invariant Feature Transform)

```python
def sift_features(image):
    """Ekstraksi SIFT features"""
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(img_with_keypoints)
    axes[1].set_title(f'SIFT Keypoints ({len(keypoints)} detected)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return keypoints, descriptors

# Extract SIFT features
keypoints, descriptors = sift_features(sample_img)
print(f"Detected {len(keypoints)} SIFT keypoints")
if descriptors is not None:
    print(f"Descriptor shape: {descriptors.shape}")
```

## Object Detection

---

### 1. Template Matching

```python
def template_matching(image, template):
    """Template matching untuk mencari objek dalam gambar"""
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    # Get template dimensions
    h, w = template_gray.shape
    
    # Apply template matching
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find locations with good match
    threshold = 0.8
    locations = np.where(result >= threshold)
    
    # Draw rectangles around matched regions
    result_img = image.copy()
    for pt in zip(*locations[::-1]):
        cv2.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    
    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(template)
    axes[1].set_title('Template')
    axes[1].axis('off')
    
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title('Matching Result')
    axes[2].axis('off')
    
    axes[3].imshow(result_img)
    axes[3].set_title('Detected Objects')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result, locations

# Demo template matching dengan creating template dari sample image
template = sample_img[50:100, 50:100]  # Extract red rectangle sebagai template
matches, match_locations = template_matching(sample_img, template)
```

### 2. Contour Detection dan Analysis

```python
def contour_analysis(image):
    """Analisis contour untuk deteksi bentuk"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create result image
    result = image.copy()
    
    # Analyze each contour
    contour_info = []
    for i, contour in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip small contours
        if area < 100:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine shape based on number of vertices
        vertices = len(approx)
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif vertices > 4:
            shape = "Circle" if vertices > 8 else "Polygon"
        else:
            shape = "Unknown"
        
        # Draw contour and label
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        cv2.putText(result, f"{shape}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        contour_info.append({
            'shape': shape,
            'area': area,
            'perimeter': perimeter,
            'vertices': vertices,
            'center': (x + w//2, y + h//2)
        })
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title('Threshold')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('Detected Shapes')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return contour_info

# Analyze contours
shape_info = contour_analysis(sample_img)
print("Detected shapes:")
for i, info in enumerate(shape_info):
    print(f"Shape {i+1}: {info['shape']}, Area: {info['area']:.1f}, Vertices: {info['vertices']}")
```

## Deep Learning untuk Computer Vision

---

### 1. CNN untuk Image Classification

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Create sample dataset for classification
def create_shape_dataset(n_samples=1000):
    """Generate dataset dengan berbagai shapes"""
    
    images = []
    labels = []
    
    for i in range(n_samples):
        # Create blank image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Randomly choose shape type
        shape_type = np.random.randint(0, 3)
        
        if shape_type == 0:  # Rectangle
            x1, y1 = np.random.randint(5, 30, 2)
            x2, y2 = x1 + np.random.randint(10, 25), y1 + np.random.randint(10, 25)
            cv2.rectangle(img, (x1, y1), (min(x2, 60), min(y2, 60)), 
                         (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)), -1)
            labels.append(0)
            
        elif shape_type == 1:  # Circle
            center = (np.random.randint(15, 50), np.random.randint(15, 50))
            radius = np.random.randint(8, 20)
            cv2.circle(img, center, radius, 
                      (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)), -1)
            labels.append(1)
            
        else:  # Triangle
            points = np.random.randint(10, 55, (3, 2))
            cv2.fillPoly(img, [points], 
                        (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)))
            labels.append(2)
        
        images.append(img)
    
    return np.array(images), np.array(labels)

# Generate dataset
X, y = create_shape_dataset(1000)
X = X.astype('float32') / 255.0  # Normalize

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Classes: Rectangle(0), Circle(1), Triangle(2)")

# Visualize sample data
class_names = ['Rectangle', 'Circle', 'Triangle']
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(X[i])
    plt.title(f'{class_names[y[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### 2. Build dan Train CNN Model

```python
# Build CNN model untuk shape classification
def build_shape_classifier():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create dan train model
shape_model = build_shape_classifier()
shape_model.summary()

# Train model
history = shape_model.fit(X_train, y_train,
                         epochs=20,
                         batch_size=32,
                         validation_split=0.2,
                         verbose=1)

# Evaluate model
test_loss, test_accuracy = shape_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

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
```

### 3. Model Evaluation dan Predictions

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Make predictions
y_pred = shape_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualize predictions
plt.figure(figsize=(15, 10))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_test[i])
    
    true_label = class_names[y_test[i]]
    pred_label = class_names[y_pred_classes[i]]
    confidence = np.max(y_pred[i])
    
    color = 'green' if y_test[i] == y_pred_classes[i] else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.2f})', 
              color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Transfer Learning untuk Computer Vision

---

### 1. Using Pre-trained Models

```python
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=True)

# Function untuk predict dengan pre-trained model
def predict_with_pretrained(image, model, preprocess_func, decode_func):
    """Predict menggunakan pre-trained model"""
    
    # Resize image untuk model input
    img_resized = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_func(img_array)
    
    # Make prediction
    predictions = model.predict(img_preprocessed)
    decoded_predictions = decode_func(predictions, top=3)[0]
    
    return decoded_predictions

# Demo dengan sample image (dalam praktik, gunakan real images)
# predictions = predict_with_pretrained(sample_img, vgg16_model, preprocess_input, decode_predictions)
# print("Top 3 predictions:")
# for i, (imagenet_id, label, score) in enumerate(predictions):
#     print(f"{i+1}. {label}: {score:.4f}")

print("Pre-trained VGG16 model loaded successfully!")
print("Model input shape:", vgg16_model.input_shape)
print("Model output shape:", vgg16_model.output_shape)
```

### 2. Fine-tuning Pre-trained Model

```python
def create_transfer_learning_model(base_model_name='VGG16', num_classes=3):
    """Create transfer learning model"""
    
    # Load base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create transfer learning model
transfer_model = create_transfer_learning_model('VGG16', 3)
transfer_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

print("Transfer Learning Model:")
transfer_model.summary()

# Train transfer learning model
transfer_history = transfer_model.fit(X_train, y_train,
                                    epochs=10,
                                    batch_size=32,
                                    validation_split=0.2,
                                    verbose=1)

# Evaluate transfer learning model
transfer_test_loss, transfer_test_accuracy = transfer_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTransfer Learning Test Accuracy: {transfer_test_accuracy:.4f}")
```

## Advanced Computer Vision Applications

---

### 1. Object Detection dengan YOLO (Conceptual)

```python
# Conceptual YOLO implementation struktur
class SimpleYOLO:
    """Simplified YOLO architecture untuk educational purposes"""
    
    def __init__(self, input_shape=(416, 416, 3), num_classes=80):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = self.generate_anchors()
        
    def generate_anchors(self):
        """Generate anchor boxes untuk different scales"""
        anchors = [
            [(10, 13), (16, 30), (33, 23)],      # Small objects
            [(30, 61), (62, 45), (59, 119)],    # Medium objects  
            [(116, 90), (156, 198), (373, 326)] # Large objects
        ]
        return anchors
    
    def build_backbone(self):
        """Build backbone network (simplified Darknet)"""
        backbone = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=self.input_shape),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(512, 3, activation='relu', padding='same'),
        ])
        return backbone
    
    def build_detection_head(self, input_tensor):
        """Build detection head untuk bounding box dan class prediction"""
        # Each anchor predicts: x, y, w, h, confidence, class_probs
        output_channels = len(self.anchors[0]) * (5 + self.num_classes)
        
        x = layers.Conv2D(output_channels, 1, activation='sigmoid')(input_tensor)
        return x
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.5):
        """Non-maximum suppression untuk menghilangkan duplicate detections"""
        # Simplified NMS implementation
        indices = []
        sorted_indices = np.argsort(scores)[::-1]
        
        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            indices.append(current)
            
            if len(sorted_indices) == 1:
                break
                
            # Calculate IoU dengan remaining boxes
            ious = self.calculate_iou(boxes[current], boxes[sorted_indices[1:]])
            
            # Keep boxes dengan IoU < threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]
        
        return indices
    
    def calculate_iou(self, box1, boxes):
        """Calculate Intersection over Union"""
        # Simplified IoU calculation
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / union

# Initialize YOLO model
yolo_model = SimpleYOLO()
print("YOLO Model initialized!")
print(f"Input shape: {yolo_model.input_shape}")
print(f"Number of classes: {yolo_model.num_classes}")
print(f"Anchor boxes: {yolo_model.anchors}")
```

### 2. Image Segmentation

```python
def simple_segmentation(image):
    """Simple image segmentation menggunakan K-means clustering"""
    
    # Reshape image untuk clustering
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = 4  # Number of clusters
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 dan reshape ke original shape
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)
    
    # Create individual segment masks
    segments = []
    for i in range(k):
        mask = (labels.flatten() == i).reshape(image.shape[:2])
        segments.append(mask)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(segmented_image)
    axes[0, 1].set_title('Segmented')
    axes[0, 1].axis('off')
    
    # Show individual segments
    for i in range(min(4, k)):
        row, col = (0, 2) if i < 1 else (1, i-1)
        axes[row, col].imshow(segments[i], cmap='gray')
        axes[row, col].set_title(f'Segment {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return segmented_image, segments

# Apply segmentation
segmented_img, segments = simple_segmentation(sample_img)
```

## Practical Project: Real-time Face Detection

---

```python
def face_detection_project():
    """Complete face detection project menggunakan OpenCV"""
    
    # Load pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_faces_in_image(image):
        """Detect faces dalam single image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        result = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes dalam face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return result, len(faces)
    
    def create_face_dataset():
        """Create sample face-like data untuk demo"""
        faces = []
        
        for i in range(10):
            # Create simple face-like shape
            face = np.zeros((100, 100, 3), dtype=np.uint8)
            
            # Face outline (circle)
            cv2.circle(face, (50, 50), 40, (200, 180, 150), -1)
            
            # Eyes
            cv2.circle(face, (35, 40), 5, (0, 0, 0), -1)
            cv2.circle(face, (65, 40), 5, (0, 0, 0), -1)
            
            # Nose
            cv2.line(face, (50, 45), (50, 55), (150, 120, 100), 2)
            
            # Mouth
            cv2.ellipse(face, (50, 65), (10, 5), 0, 0, 180, (100, 50, 50), 2)
            
            faces.append(face)
        
        return faces
    
    # Create demo dataset
    demo_faces = create_face_dataset()
    
    # Display results
    plt.figure(figsize=(15, 6))
    for i, face in enumerate(demo_faces[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(face)
        plt.title(f'Demo Face {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Face detection project initialized!")
    print("In real application, this would work with:")
    print("- Webcam feed untuk real-time detection")
    print("- Image files untuk batch processing")
    print("- Video files untuk video analysis")
    
    return face_cascade, eye_cascade

# Initialize face detection project
face_detector, eye_detector = face_detection_project()
```

## Performance Optimization

---

### 1. Model Optimization Techniques

```python
def optimize_model_performance():
    """Teknik optimasi untuk computer vision models"""
    
    # 1. Model Quantization
    print("1. Model Quantization:")
    print("   - Reduce model size dengan converting weights ke lower precision")
    print("   - TensorFlow Lite untuk mobile deployment")
    
    # 2. Model Pruning
    print("\n2. Model Pruning:")
    print("   - Remove unnecessary connections dalam neural network")
    print("   - Maintain accuracy sambil reduce computation")
    
    # 3. Knowledge Distillation
    print("\n3. Knowledge Distillation:")
    print("   - Train smaller student model dari larger teacher model")
    print("   - Transfer knowledge untuk maintain performance")
    
    # 4. Image Preprocessing Optimization
    def optimize_image_pipeline(images):
        """Optimized image preprocessing pipeline"""
        
        # Batch processing
        processed_images = []
        batch_size = 32
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Vectorized operations
            batch = np.array(batch, dtype=np.float32)
            batch = batch / 255.0  # Normalization
            
            processed_images.extend(batch)
        
        return np.array(processed_images)
    
    print("\n4. Image Pipeline Optimization:")
    print("   - Batch processing untuk efficiency")
    print("   - Vectorized operations")
    print("   - Memory management")
    
    return optimize_image_pipeline

# Demonstrate optimization concepts
optimization_func = optimize_model_performance()
```

### 2. GPU Acceleration

```python
def check_gpu_availability():
    """Check GPU availability untuk deep learning"""
    
    print("GPU Availability Check:")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        # Set memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs available, using CPU")
    
    # Mixed precision training
    print("\nMixed Precision Training:")
    print("- Use both float16 dan float32 untuk faster training")
    print("- Maintain numerical stability")
    
    return len(gpus) > 0

# Check GPU availability
has_gpu = check_gpu_availability()
```

## Key Takeaways

---

1. **Computer Vision Pipeline**: Image acquisition → Preprocessing → Feature extraction → Analysis → Decision
2. **Classical vs Deep Learning**: Traditional methods masih relevan untuk specific tasks
3. **CNN Architecture**: Fundamental untuk modern computer vision applications
4. **Transfer Learning**: Leverage pre-trained models untuk faster development
5. **Real-time Processing**: Optimization critical untuk production applications
6. **Multi-modal Integration**: Combine computer vision dengan NLP, audio, etc.

Pada file selanjutnya, kita akan membahas Natural Language Processing (NLP) dan bagaimana AI memahami bahasa manusia!