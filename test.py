import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math
from gtts import gTTS
import os

# Load the trained TensorFlow model
model_path = '/Users/ghost/Documents/gradproject/finalproject2/detect_Signs/Model_path'  # Update this path to your model
model = tf.saved_model.load(model_path)
serving_fn = model.signatures["serving_default"]

# Inspect the model's output to see available keys
print(serving_fn.structured_outputs)

# Parameters
offset = 20
model_input_size = 64  # Update based on the input size of your model
imgSize = model_input_size  # Ensures the processed image size matches the model's input size
labels = ["Hello", "i love you", "An A", "Deserve", "I", "yes", "Thank you", "A", "B", "C", "D", "E", "L", "W", "O"]  # Update as per your classes

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas for the input image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop and resize the region of interest (ROI)
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size == 0:  # Skip if the crop is empty
            continue

        aspectRatio = h / w

        if aspectRatio > 1:  # Tall image
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:  # Wide image
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Preprocess the image for model inference
        imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        imgWhite = imgWhite / 255.0  # Normalize pixel values to [0, 1]
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
        imgWhite = tf.convert_to_tensor(imgWhite, dtype=tf.float32)  # Convert to float32

        # Predict using the loaded model
        predictions = serving_fn(inputs=imgWhite)  # Explicitly pass the 'inputs' argument
        
        # Check the model's actual output layer names
        print(predictions.keys())  # Print the available output keys
        
        # Update this line with the correct key from the printed keys
        predictions = predictions['output_0']  # Update "output_0" based on the key printed above
        predictions = predictions.numpy()
        
        # Print the shape of predictions to debug
        print(f"Prediction shape: {predictions.shape}")
        
        # Get the index of the highest prediction
        index = np.argmax(predictions)

        # Check if index is within the range of labels
        if index < len(labels):
            # Display prediction on the output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x + w + offset, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            print(f"Prediction: {labels[index]}, Confidence: {predictions[0][index]:.2f}")

            # Convert the prediction to speech
            tts = gTTS(text=labels[index], lang='en', slow=False)  # Language can be adjusted
            tts.save("prediction1.mp3")
            os.system("mpg321 prediction.mp3")  # Play the audio file (you might need to adjust based on OS)
        else:
            print(f"Prediction index {index} out of range. Adjust the model or labels.")
            
    # Display images
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
