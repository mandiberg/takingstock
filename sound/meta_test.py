from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "some example text in the English language"
inputs = tokenizer(text, return_tensors="pt")
sample_rate = model.generation_config.sample_rate
print(sample_rate)

with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.float().numpy())


