"use client"; // Enable client-side interactivity

import { useState, useRef, useEffect } from 'react';

export default function Home() {
  const videoRef = useRef(null); // Reference for the video element
  const [text, setText] = useState(''); // State for converted text
  const [isCameraOn, setIsCameraOn] = useState(false); // State for camera status

  // Function to start the camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraOn(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Failed to access camera. Please allow camera permissions.');
    }
  };

  // Function to stop the camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
  };

  // Function to convert sign language to text (mock implementation)
  const convertSignToText = () => {
    // Replace this with your actual AI model logic
    const mockText = "Hello, how are you?";
    setText(mockText);
    speakText(mockText);
  };

  // Function to convert text to speech
  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  // Cleanup camera on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="container">
      <h1>Sign Language to Text & Speech</h1>

      {/* Camera Section */}
      <div className="camera-section">
        <video ref={videoRef} autoPlay playsInline muted className="camera-feed"></video>
        <div className="camera-controls">
          {!isCameraOn ? (
            <button onClick={startCamera} className="camera-button">
              Start Camera
            </button>
          ) : (
            <button onClick={stopCamera} className="camera-button">
              Stop Camera
            </button>
          )}
          <button onClick={convertSignToText} disabled={!isCameraOn} className="action-button">
            Convert Sign to Text
          </button>
        </div>
      </div>

      {/* Text Display Section */}
      <div className="text-section">
        <h2>Converted Text:</h2>
        <p>{text}</p>
      </div>
    </div>
  );
}