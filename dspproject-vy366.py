import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import wave
import pyaudio
from scipy.io.wavfile import write, read
from scipy.signal import spectrogram
import cv2


class AudioEncryptDecryptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Encryption & Decryption")
        self.audio_file = None
        self.encrypted_image = None
        self.metadata_file = None
        self.decrypted_audio = "decrypted_audio.wav"
        self.key = None

        # UI Elements
        self.select_button = tk.Button(root, text="Select Audio File", command=self.select_audio)
        self.select_button.pack(pady=10)

        self.play_original_button = tk.Button(root, text="Play Original Audio", command=self.play_original, state=tk.DISABLED)
        self.play_original_button.pack(pady=10)

        self.encrypt_button = tk.Button(root, text="Encrypt Audio", command=self.encrypt_audio, state=tk.DISABLED)
        self.encrypt_button.pack(pady=10)

        self.upload_image_button = tk.Button(root, text="Upload Encrypted Image", command=self.upload_encrypted_image)
        self.upload_image_button.pack(pady=10)

        self.upload_metadata_button = tk.Button(root, text="Upload Metadata", command=self.upload_metadata)
        self.upload_metadata_button.pack(pady=10)

        self.decrypt_button = tk.Button(root, text="Decrypt Audio", command=self.decrypt_audio, state=tk.DISABLED)
        self.decrypt_button.pack(pady=10)

        self.play_decrypted_button = tk.Button(root, text="Play Decrypted Audio", command=self.play_decrypted, state=tk.DISABLED)
        self.play_decrypted_button.pack(pady=10)

        self.plot_button = tk.Button(root, text="Show Waveforms", command=self.plot_waveforms, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

    def select_audio(self):
        self.audio_file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.audio_file:
            self.play_original_button.config(state=tk.NORMAL)
            self.encrypt_button.config(state=tk.NORMAL)
            messagebox.showinfo("File Selected", "Audio file loaded successfully!")

    def upload_encrypted_image(self):
        self.encrypted_image = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
        if self.encrypted_image:
            messagebox.showinfo("File Selected", "Encrypted image file loaded successfully!")
            self.decrypt_button.config(state=tk.NORMAL)

    def upload_metadata(self):
        self.metadata_file = filedialog.askopenfilename(filetypes=[("Metadata files", "*.npz")])
        if self.metadata_file:
            messagebox.showinfo("File Selected", "Metadata file loaded successfully!")
            self.decrypt_button.config(state=tk.NORMAL)

    def play_audio(self, file):
        chunk = 1024
        wf = wave.open(file, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                         channels=wf.getnchannels(),
                         rate=wf.getframerate(),
                         output=True)
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def play_original(self):
        if self.audio_file:
            self.play_audio(self.audio_file)

    def encrypt_audio(self):
        rate, data = read(self.audio_file)
        is_stereo = len(data.shape) > 1
        if is_stereo and data.shape[1] > 2:
            data = data[:, :2]
            messagebox.showwarning("Channel Reduction", "Audio has more than 2 channels. Only the first 2 channels will be used.")
        elif not is_stereo:
            data = np.expand_dims(data, axis=1)

        data = data.astype(np.float32)
        max_val = np.max(np.abs(data))
        normalized_data = data / max_val

        flattened_data = normalized_data.flatten()
        size = int(np.ceil(np.sqrt(len(flattened_data))))
        padded_data = np.zeros(size * size, dtype=np.float32)
        padded_data[:len(flattened_data)] = flattened_data
        square_matrix = padded_data.reshape((size, size))

        self.key = np.random.uniform(1.0, 10.0)
        encrypted_matrix = square_matrix * self.key

        # Save encrypted data as a 16-bit grayscale image
        encrypted_image = np.clip(encrypted_matrix * 65535, 0, 65535).astype('uint16')
        cv2.imwrite("encrypted_audio.png", encrypted_image)

        # Save metadata
        np.savez("metadata.npz", key=self.key, max_val=max_val, original_shape=data.shape, audio_length=len(flattened_data))

        self.decrypt_button.config(state=tk.NORMAL)
        self.plot_button.config(state=tk.NORMAL)
        messagebox.showinfo("Encryption Complete", "Audio encrypted and saved as an image.")

    def decrypt_audio(self):
      if not self.encrypted_image or not self.metadata_file:
          messagebox.showerror("Error", "Please upload both the encrypted image and metadata file!")
          return

      # Load metadata
      metadata = np.load(self.metadata_file)
      key = metadata["key"]
      max_val = metadata["max_val"]
      original_shape = metadata["original_shape"]
      audio_length = metadata["audio_length"]

      # Load encrypted image
      encrypted_image = cv2.imread(self.encrypted_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535

      # Decrypt the audio signal
      decrypted_matrix = encrypted_image / key
      flattened_data = decrypted_matrix.flatten()[:audio_length]

      # Reshape and scale back to the original amplitude range
      audio_data = flattened_data.reshape(original_shape) * max_val

      # Center the waveform around zero
      audio_data -= np.mean(audio_data)

      # Save the decrypted audio
      write(self.decrypted_audio, 44100, audio_data.astype(np.int16))
      self.play_decrypted_button.config(state=tk.NORMAL)
      messagebox.showinfo("Decryption Complete", "Audio decrypted and saved.")


    def play_decrypted(self):
        if self.decrypted_audio:
            self.play_audio(self.decrypted_audio)

    def plot_waveforms(self):
      rate, original_data = read(self.audio_file)
      rate, decrypted_data = read(self.decrypted_audio)

      # Center decrypted data around zero
      decrypted_data = decrypted_data - np.mean(decrypted_data)

      plt.figure(figsize=(12, 8))

      # Original Audio Waveform
      plt.subplot(3, 1, 1)
      plt.title("Original Audio Waveform")
      if len(original_data.shape) > 1:
          for i, channel in enumerate(original_data.T):
              plt.plot(channel, label=f'Channel {i + 1}')
          plt.legend()
      else:
          plt.plot(original_data, color="blue")
      plt.ylim(-np.max(np.abs(original_data)), np.max(np.abs(original_data)))

      # Decrypted Audio Waveform (No Scaling)
      plt.subplot(3, 1, 2)
      plt.title("Decrypted Audio Waveform (Not Scaled)")
      if len(decrypted_data.shape) > 1:
          for i, channel in enumerate(decrypted_data.T):
              plt.plot(channel, label=f'Channel {i + 1}')
          plt.legend()
      else:
          plt.plot(decrypted_data, color="green")
      plt.ylim(-np.max(np.abs(decrypted_data)), np.max(np.abs(decrypted_data)))

      # Spectrogram Comparison
      plt.subplot(3, 1, 3)
      plt.title("Spectrogram Comparison")
      plt.ylabel("Frequency (Hz)")

      # Original Audio Spectrogram
      f_orig, t_orig, Sxx_orig = spectrogram(original_data[:, 0] if original_data.ndim > 1 else original_data, rate)
      plt.pcolormesh(t_orig, f_orig, 10 * np.log10(Sxx_orig), shading='gouraud', cmap="viridis")
      plt.colorbar(label='Original Intensity (dB)')

      plt.tight_layout()
      plt.show()

      # Separate Decrypted Audio Spectrogram
      plt.figure(figsize=(6, 4))
      plt.title("Decrypted Audio Spectrogram")
      f_dec, t_dec, Sxx_dec = spectrogram(decrypted_data[:, 0] if decrypted_data.ndim > 1 else decrypted_data, rate)
      plt.pcolormesh(t_dec, f_dec, 10 * np.log10(Sxx_dec), shading='gouraud', cmap="viridis")
      plt.colorbar(label='Decrypted Intensity (dB)')
      plt.ylabel("Frequency (Hz)")
      plt.xlabel("Time (s)")

      plt.tight_layout()
      plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEncryptDecryptApp(root)
    root.mainloop()
