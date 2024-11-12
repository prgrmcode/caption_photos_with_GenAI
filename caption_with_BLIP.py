from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import matplotlib.pyplot as plt


# Initialize the processor and model from Hugging Face: https://huggingface.co/Salesforce/blip-image-captioning-base
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda") # Running the model on GPU

# Load an image
image = Image.open("./example_img.jpg")

# Prepare the image
inputs = processor(image, return_tensors="pt")
# The return_tensors="pt" argument specifies that the output should be in PyTorch tensor format.
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU

# Generate the captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)

# Visualize the image with its caption
plt.imshow(image)
plt.title(caption)
plt.axis('off')  # Hide the axes
plt.show()