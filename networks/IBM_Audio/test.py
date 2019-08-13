from networks.IBM_Audio.model import *
import librosa

model_wrpper = ModelWrapper()
data, sr = librosa.load("common/data/training/TIMIT/MZMB0/SA1.WAV")
print(data, sr)
embeddings = model_wrpper.generate_embeddings(data, sr)

print("------------------------>>>>>>>>>>>>>>>>")
print(embeddings)

print("------------------------>>>>>>>>>>>>>>>>")

print(embeddings.shape)