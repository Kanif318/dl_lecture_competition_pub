import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "IDEA-Research/grounding-dino-base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
print("processor", processor)
print("model", model)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
print("inputs", inputs)

with torch.no_grad():
    outputs = model(**inputs)
print("outputs",outputs)
print("encoder_last_hidden_state_text", outputs.encoder_last_hidden_state_text.shape)
print("encoder_last_hidden_state_vision", outputs.encoder_last_hidden_state_vision.shape)
print("intermediate_hidden_states", outputs.intermediate_hidden_states.shape)

last_hidden_states = outputs.last_hidden_state
print("last_hidden_state", outputs.last_hidden_state.shape)