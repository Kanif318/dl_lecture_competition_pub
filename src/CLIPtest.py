from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLIPProcessorからtokenizerを取得
tokenizer = processor.tokenizer
text_encoder = model.text_model
vision_encoder = model.vision_model
image_preprocessor = processor.feature_extractor

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_processed = image_preprocessor(image, return_tensors = "pt")

text="a photo of a cat"
tokens = tokenizer(text, return_tensors = "pt", padding = True)
output = text_encoder(**tokens)
#print("text_output", output)
print("text_hidden_shape", output.last_hidden_state.shape)

image_outputs = vision_encoder(**image_processed)
#print("image_output", image_outputs)
print("image_hidden_shape", image_outputs.last_hidden_state.shape)
# outputs = model(**inputs)
# print("outputs_text", outputs.text_embeds)
# print(outputs.text_embeds.shape)
# print("output_image", outputs.image_embeds)
# print(outputs.image_embeds.shape)


# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
