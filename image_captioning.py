import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt


# Load the pretrained model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = "roads.jpeg"
# convert image to an RGB format
image = Image.open(img_path).convert('RGB')

# The pre-processed image is passed through the processor to generate inputs in the required format. The return_tensors argument is set to "pt" to return PyTorch tensors.
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# The two asterisks (**) in Python are used in function calls to unpack dictionaries and pass items in the dictionary as keyword arguments to the function. **inputs is unpacking the inputs dictionary and passing its items as arguments to the model.

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

# Finally, the generated output is a sequence of tokens. To transform these tokens into human-readable text, you use the decode method provided by the processor. The skip_special_tokens argument is set to True to ignore special tokens in the output text.

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print("Generated Caption:", caption)

# Visualize the image with its caption
plt.imshow(image)
plt.title(caption)
plt.axis('off')  # Hide the axes
plt.show()