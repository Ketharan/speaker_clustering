from .model import *

model_wrpper = ModelWrapper()
embeddings = model_wrpper._generate_embeddings("common/data/training/TIMIT/MZMB0/SA1.WAV")

print("------------------------>>>>>>>>>>>>>>>>")
print(len(embeddings))

print("------------------------>>>>>>>>>>>>>>>>")
print(embeddings.shape)