# sign-to-text-speech-AI-model

An AI model built to bridge the gap between the disabled community and everyone else. Making signing a lot easier to understand and communication a lot smoother.

A Sign Language to Text and Speech AI Model is a system designed to translate sign language gestures into written text or spoken words. This type of AI model is developed to help bridge the communication gap between individuals who use sign language and those who rely on spoken or written language.

# How It Works🧑🏽‍💻🧑🏽‍💻

## Data Collection and Preprocessing:

    The model is trained on a dataset containing videos or images of sign language gestures.
    Each gesture is labeled with its corresponding meaning in text or speech.

## Gesture Recognition:

    A computer vision model (e.g., using OpenCV or deep learning frameworks like TensorFlow or PyTorch) processes the video or image input.
    Techniques like hand tracking, keypoint detection, or pose estimation (e.g., Mediapipe) are used to identify the movement of hands, fingers, and other relevant body parts.

## Feature Extraction:

    The model extracts key features from the input, such as finger shapes, positions, and motion.
    Classification:

    A machine learning or deep learning model (e.g., CNN, RNN, or Transformer) classifies the gesture into its corresponding sign.

## Text Generation:

    The recognized gesture is converted into text (e.g., "Hello," "Thank you").

## Speech Generation (Optional):🗣🗣

    A text-to-speech (TTS) system converts the text into audible speech using tools like Google Text-to-Speech or custom TTS models.

## Use Cases:

    Communication Tools: Helping people with hearing or speech impairments communicate more effectively with those who do not know sign language.

    Education: Assisting in teaching sign language or improving learning materials.

    Customer Service: Allowing businesses to serve deaf or hard-of-hearing customers more inclusively.

    Healthcare: Enhancing communication in medical settings where understanding is critical.

## Technologies Involved:

    Computer Vision: For gesture and movement detection.

    Natural Language Processing (NLP): For mapping gestures to text or speech.

    Speech Synthesis: To generate spoken language.

    Machine Learning Frameworks: TensorFlow, PyTorch, Mediapipe, etc.

# HOW TO IMPLEMENT THIS PROJECT

## 🦾🦾 PREREQUISITES

All the dependencies and required libraries are included in the file **requirements.txt** [Click Here](https://drive.google.com/file/d/10YNWv3xTxgsMRLE18pzTExG9k54rB6T8/view?usp=drive_link)

## 🚀 Installation

1.  Start and fork the repository.

2.  Clone the repo:

        git clone https://github.com/Henry-Edet/sign-to-text-speech-AI-model.git

3.  Change your directory to the cloned repo and create a Python virtual environment named 'whatever name you want'

        python -m venv whatever name you want

    if you installed virtualenv, run:

        virtualenv whatever name you want

    Activate the Virtual Environment

        source myenv/bin/activate

    On Windows:

        myenv\Scripts\activate

4.  Now, run the following command in your Terminal/Command Prompt to install the libraries required
    pip install -r requirements.txt

## 💡 Is it Working?? //(This should be done first before creating virtual environment, locate your project directory, create virttual environment before running "code .")//

1.  In terminal window run:

        cd /Users/...

2.  Navigate to your directory and run if you're using vscode:

        code .

3.  To train the model, open the [train_py](https://drive.google.com/file/d/12y8ZHrRsNz-irH7qmUKeecfaQNS_k-i8/view?usp=drive_link) file in VScode and run.

4.  To detect ASL Gestures in real-time video streams run the [test.py](https://drive.google.com/file/d/1lpK6y24jwx27JNjDhDqCvC6W4DbWi6Gk/view?usp=drive_link) file.
