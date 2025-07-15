from transformers import AutoProcessor, AutoModelForTextToWaveform
import torch
import scipy.io.wavfile as wavfile
import os

prompt = os.getenv("PROMPT", "Peaceful bamboo chimes blended with a gentle mountain stream")
token = os.getenv("HF_TOKEN")

print("🔊 Prompt:", prompt)
print("⚙️ Loading model...")

processor = AutoProcessor.from_pretrained("facebook/musicgen-small", token=token)
model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small", token=token)

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
with torch.no_grad():
    audio_values = model(**inputs).audio_values[0]

sampling_rate = model.config.sampling_rate
wavfile.write("output.wav", sampling_rate, audio_values.numpy())
print("✅ Saved output.wav")
