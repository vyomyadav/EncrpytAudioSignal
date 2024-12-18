# **Audio Encryption and Decryption GUI**

## **Overview**

This project is a Python-based GUI application for encrypting and decrypting audio files. The audio is converted into a compressed encrypted image, ensuring secure storage and transmission. It supports real-time audio recording or uploading `.wav` files, and the encrypted audio can be decrypted back to its original form with high fidelity.

---

## **Features**

- **Record Audio**: Record real-time audio (default 10 seconds).  
- **Upload Audio**: Encrypt pre-existing `.wav` files.  
- **Encrypt Audio**: Save the encrypted audio as an image file with metadata.  
- **Decrypt Audio**: Recover the original audio from the image and metadata.  
- **Visualize Waveforms**: Compare original and decrypted audio waveforms and spectrograms.  
- **Compression**: Reduces audio file size by 20-50% during encryption.  

---

## **Dependencies**

Install the following Python libraries:
- **Tkinter** (standard library for GUI)  
- **numpy**  
- **matplotlib**  
- **scipy**  
- **wave** (standard library)  
- **pyaudio**  
- **OpenCV** (only for reading and writing the encrypted image)

---

## **Notes**

- The application currently supports only `.wav` audio files as input.  
- The system works seamlessly with both **mono** and **stereo** audio formats.  
- The encrypted image and metadata file must both be stored securely for successful decryption.  
- Real-time audio recording duration is **10 seconds** by default but can be adjusted in the code.  
