import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt


# Initialize the processor and model from Hugging Face: https://huggingface.co/Salesforce/blip-image-captioning-large
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda") # Running the model on GPU

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# # Load an image
# raw_image = Image.open("./example_img.jpg")

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
 
# # unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt")
# inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU
# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# Visualize the image with its caption
plt.imshow(raw_image)
plt.title(processor.decode(out[0], skip_special_tokens=True))
plt.axis('off')  # Hide the axes
plt.show()

