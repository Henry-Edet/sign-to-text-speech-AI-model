import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
} from "react-native";
import { Camera, useCameraDevices } from "react-native-vision-camera";

const SignToTextApp = () => {
  const [permission, setPermission] = useState(null);
  const [recognizedText, setRecognizedText] = useState("");
  const devices = useCameraDevices();
  const device = devices.back;

  useEffect(() => {
    (async () => {
      const newCameraPermission = await Camera.requestCameraPermission();
      setPermission(newCameraPermission);
    })();
  }, []);

  const handleTranslate = () => {
    // Placeholder function to simulate translation
    setRecognizedText("Hello, how are you?");
  };

  if (permission !== "authorized") {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required to use this app.
        </Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraContainer}>
        {device && (
          <Camera style={styles.camera} device={device} isActive={true} />
        )}
      </View>
      <View style={styles.outputContainer}>
        <Text style={styles.outputText}>
          {recognizedText || "Sign language will appear here..."}
        </Text>
      </View>
      <TouchableOpacity style={styles.button} onPress={handleTranslate}>
        <Text style={styles.buttonText}>Convert to Speech</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
    alignItems: "center",
    justifyContent: "center",
  },
  cameraContainer: {
    width: "90%",
    height: "50%",
    borderRadius: 20,
    overflow: "hidden",
    backgroundColor: "#000",
  },
  camera: {
    flex: 1,
  },
  outputContainer: {
    marginTop: 20,
    padding: 15,
    width: "90%",
    backgroundColor: "#fff",
    borderRadius: 10,
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  outputText: {
    fontSize: 18,
    textAlign: "center",
    color: "#333",
  },
  button: {
    marginTop: 20,
    padding: 15,
    backgroundColor: "#007AFF",
    borderRadius: 10,
    width: "80%",
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  permissionText: {
    fontSize: 16,
    color: "#333",
    textAlign: "center",
  },
});

export default SignToTextApp;
