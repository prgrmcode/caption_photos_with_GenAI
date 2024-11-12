# Step 1: Set up the environment
# Run pip install gradio transformers Pillow matplotlib to install the required libraries.
# Import the required libraries:
import re
import gradio as gr
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import numpy as np
from transformers import AutoProcessor, BlipForConditionalGeneration

# from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import asyncio
import matplotlib.pyplot as plt

# Constants
MAX_IMAGES = 21  # Maximum number of images to process/display

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 2: Load the pretrained model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large", device_map={"": 0}
).to(device)
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
# ).to(device)


"""

# Step 3: Firstly, send an HTTP request to the provided URL and retrieve the webpage's content. This content is 
# then parsed by BeautifulSoup, which creates a parse tree from page's HTML. Look for 'img' tags in this parse tree 
# as they contain the links to the images hosted on the webpage.

# URL of the page to scrape images from
# url = "https://www.bbc.com/news"
url = "https://en.wikipedia.org/wiki/IBM"
# Download the HTML content of the page
response = requests.get(url)
# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")


# Step 4: After extracting these URLs, iterate through each one of them. Send another HTTP request to download 
# the image data associated with each URL.
# For a more efficient approach, asynchronous programming methods or the concurrent.futures library to download multiple images simultaneously.

# Find all img elements. Extract all image tags from the parsed HTML content
image_elements = soup.find_all("img")

# Define the function to clean up captions
def clean_caption(caption: str) -> str:
    # Remove prefixes like "arafed", "araffe", etc.
    cleaned_caption = re.sub(r'\baraf\w*\b', '', caption).strip()
    
    return cleaned_caption

# Open a file to write the captions
with open("captions.txt", "w") as file:
    # Iterate over each image element
    for img_el in image_elements:
        img_url = img_el.get("src")
        
        # Skip if the image URL is empty
        if not img_url:
            continue
        
        # Skip if the image is an SVG or too small - icon
        if 'svg' in img_url or '1x1' in img_url:
            continue
        
        # Correct the URL if it is malformed
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif not img_url.startswith("http://") and not img_url.startswith("https://"):
            continue
        
        try:
            # Download the image data
            response = requests.get(img_url)
            image_data = response.content
            # Convert the image data to a PIL image
            raw_img = Image.open(BytesIO(image_data))
            if raw_img.size[0] * raw_img.size[1] < 400: # Skip very small images
                continue
            # Convert the image to RGB if it is not in RGB format
            raw_img = raw_img.convert("RGB")
            # Process the image
            inputs = processor(images=raw_img, return_tensors="pt").to(device)
            # Move inputs to the same device as the model:
            # inputs = {k: v.to(device) for k, v in inputs.items()}  

            # Generate a caption for the image
            outputs = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            cleaned_caption = clean_caption(caption)
            # Write the caption to the file, prepended by image URL
            file.write(f"{img_url}: {cleaned_caption}\n")
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

print("Captions created and saved to captions.txt")
"""

# Gradio interface logic here:


# Step 3: Define the function to extract image URLs from a website
def extract_image_urls(website_url: str):
    try:
        response = requests.get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")
        img_urls = [img["src"] for img in img_tags if "src" in img.attrs]
        return img_urls
    except Exception as e:
        print(f"Error extracting images from {website_url}: {e}")
        return []


# Step 4: Define the function to clean up captions
def sanitize_caption(caption: str) -> str:
    # Remove prefixes like "arafed", "araffe", etc.
    cleaned_caption = re.sub(r"\baraf\w*\b", "", caption).strip()
    return cleaned_caption


# Step 5: Define the image captioning function
def caption_images_from_url(website_url: str, progress=gr.Progress()):
    img_urls = extract_image_urls(website_url)
    captions = {}
    total_images = min(len(img_urls), MAX_IMAGES)
    progress_counter = 0

    for idx, img_url in enumerate(progress.tqdm(img_urls, desc="Loading...")):
        if len(captions) >= MAX_IMAGES:
            break  # Limit to MAX_IMAGES

        # Skip if the image URL is empty, the image is an SVG or too small - icon
        if not img_url or "svg" in img_url or "1x1" in img_url:
            # progress(progress_counter / total_images)
            continue

        # Correct the URL if it is malformed
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif not img_url.startswith("http://") and not img_url.startswith("https://"):
            # progress(progress_counter / total_images)
            continue

        try:
            # Load the image from the URL
            response = requests.get(img_url, timeout=15)
            image_data = response.content

            # Convert the image data to a PIL image
            raw_img = Image.open(BytesIO(image_data))
            if raw_img.size[0] * raw_img.size[1] < 400:  # Skip very small images
                # progress(progress_counter / total_images)
                continue  # Skip very small images

            # Convert the image to RGB
            raw_img = raw_img.convert("RGB")
            # Process the image
            inputs = processor(images=raw_img, return_tensors="pt").to(device)
            # Generate a caption for the image
            outputs = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # Clean the caption
            cleaned_caption = sanitize_caption(caption)
            captions[img_url] = cleaned_caption

            # progress_counter += 1
            # progress(progress_counter / total_images)

        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            # progress(progress_counter / total_images)
            continue

    print("Operation finished: All images have been processed.")
    return captions


# Function to save captions to file
def save_captions(captions: dict):
    try:
        with open("captions.txt", "w", encoding="utf-8") as file:
            for img_url, caption in captions.items():
                file.write(f"{img_url}: {caption}\n")
        print("Captions created and saved to captions.txt")
        return "<span style='color: green;'>**Captions saved successfully to `captions.txt`.**</span>"
    except Exception as e:
        print(f"Error saving captions: {e}")
        return "<span style='color: red;'>**Failed to save captions.**</span>"


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Captioning with Generative AI")

    with gr.Column():
        website_url = gr.Textbox(
            label="Enter Website URL", placeholder="https://example.com"
        )
        with gr.Row():
            start_button = gr.Button("Generate Captions")
            clear_button = gr.Button("Clear")

    generate_status = gr.Markdown("")  # Status for generating captions

    # Initialize lists to hold component references
    image_urls = []
    caption_inputs = []
    img_displays = []
    rows = []

    # captions_state = gr.State()
    # dynamic_elements = gr.State()

    # Create placeholders for image components and caption inputs
    with gr.Column() as gallery:
        for i in range(MAX_IMAGES):
            with gr.Row(visible=False) as row:
                img_display = gr.Image(label=f"Image {i+1}", height=150, visible=False)
                caption_input = gr.Textbox(
                    label=f"Caption {i+1}", lines=2, visible=False
                )
                # Store references
                rows.append(row)
                img_displays.append(img_display)
                caption_inputs.append(caption_input)

    # Gradio State to store image URLs
    image_urls_state = gr.State([])

    # Function to process website and display images with captions
    def process_and_display(website_url):
        # Clear previous image URLs to avoid duplication
        image_urls_state = []

        # Get captions
        captions = caption_images_from_url(website_url)

        i = 0

        # Prepare lists of updates for components
        img_display_updates = []
        caption_input_updates = []
        row_updates = []

        # Initialize updates to make all components hidden and empty
        for _ in range(MAX_IMAGES):
            img_display_updates.append(gr.update(visible=False, value=None))
            caption_input_updates.append(gr.update(visible=False, value=""))
            row_updates.append(gr.update(visible=False))

        # Update components with actual data
        for img_url, caption in captions.items():
            if i >= MAX_IMAGES:
                break
            img_display_updates[i] = gr.update(visible=True, value=img_url)
            caption_input_updates[i] = gr.update(visible=True, value=caption)
            row_updates[i] = gr.update(visible=True)
            image_urls_state.append(img_url)  # Store image URL in the list
            i += 1

        return (
            "### Images and captions loaded.",
            *img_display_updates,
            *caption_input_updates,
            *row_updates,
            image_urls_state,
        )

    # Function to clear the interface
    def clear_interface():
        img_display_updates = [
            gr.update(visible=False, value=None) for _ in range(MAX_IMAGES)
        ]
        caption_input_updates = [
            gr.update(visible=False, value="") for _ in range(MAX_IMAGES)
        ]
        row_updates = [gr.update(visible=False) for _ in range(MAX_IMAGES)]
        image_urls_state.value = []
        return (
            "",
            *img_display_updates,
            *caption_input_updates,
            *row_updates,
            image_urls_state,
            "",  # Clear generate_status
            "",  # Clear save_status
        )

    # Function to save modified captions
    def save_modified_captions(*args):
        *captions_list, image_urls_state = args
        captions = {}
        for i in range(len(image_urls_state)):
            img_url = image_urls_state[i]
            caption = captions_list[i]
            captions[img_url] = caption
        save_status = save_captions(captions)
        return save_status

    with gr.Column():
        save_button = gr.Button("Save Captions")
        save_status = gr.Markdown("")  # Status for saving captions

    # Button Click Events
    start_button.click(
        fn=process_and_display,
        inputs=[website_url],
        outputs=[generate_status]
        + img_displays
        + caption_inputs
        + rows
        + [image_urls_state],
        show_progress=True,  # Enable the progress bar
    )

    clear_button.click(
        fn=clear_interface,
        outputs=[generate_status]
        + img_displays
        + caption_inputs
        + rows
        + [image_urls_state]
        + [save_status],  # Clear save_status as well
    )

    save_button.click(
        fn=save_modified_captions,
        inputs=caption_inputs + [image_urls_state],
        outputs=[save_status],
        show_progress=True,  # Enable the progress bar
    )

demo.launch(share=True)

"""

# Step 7: Create the Gradio interface
def gradio_interface(website_url: str):
    captions = caption_images_from_url(website_url)
    return captions

def display_images_with_captions(captions):
    elements = []
    for img_url, caption in captions:
        elements.append(gr.Image(value=img_url, label="Image"))
        elements.append(gr.Textbox(value=caption, label="Caption", lines=2))
    return elements

def correct_and_save_captions(captions):
    final_captions = {captions[i]: captions[i+1] for i in range(0, len(captions), 2)}
    save_captions(final_captions)
    return "Captions saved successfully."

# Step 8: Create the Gradio interface for generating captions
iface = gr.Blocks()

with iface:
    website_url = gr.Textbox(label="Website URL")
    generate_button = gr.Button("Generate Captions")
    captions_state = gr.State()
    captions_output = gr.Column()
    
    generate_button.click(gradio_interface, inputs=website_url, outputs=captions_state)
    generate_button.click(display_images_with_captions, inputs=captions_state, outputs=captions_output)


# iface = gr.Interface(
#     fn=gradio_interface,
#     inputs=gr.Textbox(label="Website URL"),
#     outputs=[gr.State(), gr.JSON(label="Generated Captions")],
#     title="Image Captioning with BLIP",
#     description="Web app for automated captioning for images from a website using a trained model.",
# )

# Step 9: Create the Gradio interface for caption correction and saving
iface_correct = gr.Blocks()

with iface_correct:
    captions_input = gr.State()
    save_button = gr.Button("Save Captions")
    save_output = gr.Textbox(label="Save Status")
    
    save_button.click(correct_and_save_captions, inputs=captions_input, outputs=save_output)


# iface_correct = gr.Interface(
#     fn=correct_and_save_captions,
#     inputs=gr.State(),
#     outputs="text",
#     title="Correct and Save Captions",
#     description="Submit corrected captions for images and save them to a file.",
# )


# Step 10: Launch the Web App - Gradio interface
iface.launch()
iface_correct.launch()

"""

# Step 11: Run the application script
# python automated_url_captioner.py
