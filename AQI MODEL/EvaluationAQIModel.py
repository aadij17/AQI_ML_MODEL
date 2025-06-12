import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\AQI ML Model\\air_quality_model.h5')

# Load and preprocess the test image
img = cv2.imread('D:\AQI ML Model\\test_image.jpeg', 1)
img = np.expand_dims(img, axis=0)

# Make a prediction using the model
pred = model.predict(img)

print('###### Predicted AQI: ',pred)

# Get the predicted class label
label = np.argmax(pred)

# Map the predicted class label to the corresponding air quality level
if label == 0:
    air_quality = 'Poor'
elif label == 1:
    air_quality = 'Very poor'
else:
    air_quality = 'Severe'

# Print the predicted air quality level
print('Predicted air quality level:', air_quality)


#accuracy
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\AQI ML Model\\air_quality_model.h5')

# Define the test dataset directory
# test_dir = 'D:\AQI ML Model\Dataset\TEST'
test_dir = "D:\AQI ML Model\Dataset\TRAIN"

# Get the class names and corresponding label indices
class_names = sorted(os.listdir(test_dir))
label_indices = {class_name: i for i, class_name in enumerate(class_names)}

# Initialize the lists to store the true labels and predicted labels
true_labels = []
pred_labels = []

# Loop over the test images and predict the air quality level
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        label = np.argmax(pred)
        true_labels.append(label_indices[class_name])
        pred_labels.append(label)

# Calculate the accuracy of the model
acc = sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)

# Print the accuracy
print('Test accuracy:', acc)

#VISUALIZATION
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\AQI ML Model\\air_quality_model.h5')

# Define the test dataset directory
# test_dir = 'D:\AQI ML Model\Dataset\TEST'
test_dir = "D:\AQI ML Model\\Dataset\\TRAIN"
# Get the class names and corresponding label indices
class_names = sorted(os.listdir(test_dir))
label_indices = {class_name: i for i, class_name in enumerate(class_names)}

# Initialize the lists to store the true labels and predicted labels
true_labels = []
pred_labels = []

# Loop over the test images and predict the air quality level
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        label = np.argmax(pred)
        true_labels.append(label_indices[class_name])
        pred_labels.append(label)

# Calculate the accuracy of the model
acc = sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)

# Print the accuracy
print('Test accuracy:', acc)

# Compute and plot the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Plot a bar chart of the predicted labels
plt.figure(figsize=(6, 4))
plt.bar(class_names, [pred_labels.count(i) for i in range(len(class_names))])
plt.title('Predicted Labels')
plt.xlabel('Air Quality Level')
plt.ylabel('Number of Images')
plt.show()


# CLASSIFICATION REPORT
from sklearn.metrics import classification_report

# Print the classification report
target_names = class_names
print(classification_report(true_labels, pred_labels, target_names=target_names))