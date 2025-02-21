import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ActivityIndicator,
  Alert,
} from "react-native";
import { Camera, useCameraDevices } from "react-native-vision-camera";

const SignToTextApp = () => {
  const [permission, setPermission] = useState(null);
  const [recognizedText, setRecognizedText] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [cameraType, setCameraType] = useState("back");
  const devices = useCameraDevices();
  const device = cameraType === "back" ? devices.back : devices.front;

  // Check if running in Expo Go
  const isExpoGo = Constants?.appOwnership === "expo";

  // Request camera permission
  useEffect(() => {
    (async () => {
      try {
        const newCameraPermission = await Camera.requestCameraPermission();
        setPermission(newCameraPermission);
      } catch (error) {
        Alert.alert("Error", "Failed to request camera permission.");
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  // Handle translation (placeholder for sign language recognition)
  const handleTranslate = async () => {
    setIsLoading(true);
    try {
      // Simulate API call or model processing
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setRecognizedText("Hello, how are you?");
    } catch (error) {
      Alert.alert("Error", "Failed to translate sign language.");
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  // Switch between front and back cameras
  const switchCamera = () => {
    setCameraType((prev) => (prev === "back" ? "front" : "back"));
  };

  // Show warning if running in Expo Go
  if (isExpoGo) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>
          This app is not supported in Expo Go. Please use a development build
          or EAS Build.
        </Text>
      </SafeAreaView>
    );
  }

  // Show loading state
  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading...</Text>
      </SafeAreaView>
    );
  }

  // Show permission request message
  if (permission !== "authorized") {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required to use this app.
        </Text>
      </SafeAreaView>
    );
  }

  // Main UI
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraContainer}>
        {device ? (
          <Camera style={styles.camera} device={device} isActive={true} />
        ) : (
          <Text style={styles.errorText}>No camera device found.</Text>
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
      <TouchableOpacity style={styles.switchButton} onPress={switchCamera}>
        <Text style={styles.switchButtonText}>
          Switch to {cameraType === "back" ? "Front" : "Back"} Camera
        </Text>
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
  switchButton: {
    marginTop: 10,
    padding: 10,
    backgroundColor: "#34C759",
    borderRadius: 10,
    width: "80%",
    alignItems: "center",
  },
  switchButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  permissionText: {
    fontSize: 16,
    color: "#333",
    textAlign: "center",
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: "#333",
  },
  errorText: {
    fontSize: 16,
    color: "#FF3B30",
    textAlign: "center",
    marginTop: 20,
  },
});

export default SignToTextApp;
