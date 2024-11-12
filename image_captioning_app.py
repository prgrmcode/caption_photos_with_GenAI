# Step 1: Set up the environment
# Run pip install gradio transformers Pillow matplotlib to install the required libraries.
# Import the required libraries:
import gradio as gr
from PIL import Image
import numpy as np
# from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# Step 2: Load the pretrained model
# processor = Blip2Processor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
).to(device)


# Step 3: Define the image captioning function
def caption_image(input_image: np.ndarray) -> str:
    # Convert the numpy array to a PIL image and convert to RGB
    raw_image = Image.fromarray(input_image).convert("RGB")

    # Process the image
    inputs = processor(images=raw_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model

    
    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the generated tokens to text and store it into 'caption'
    caption = processor.decode(outputs[0], skip_special_tokens=True)    

    return caption

# Step 4: Create the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Simple web app for Generating captions for images using a trained model.",
)

# Step 5: Launch the Web App - Gradio interface
iface.launch()

# Step 6: Run the application script
# python app.py