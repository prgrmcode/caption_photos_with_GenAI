import os
import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import torch
import re

# Constants
MAX_IMAGES = 21  # Maximum number of images to process/display

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)


# Function to extract image URLs from a website
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


# Function to sanitize captions
def sanitize_caption(caption: str) -> str:
    # Remove unintended words or patterns
    cleaned_caption = re.sub(r"\baraf\w*\b", "", caption).strip()
    return cleaned_caption


# Function to caption images from URL
def caption_images_from_url(website_url: str, progress: gr.Progress):
    img_urls = extract_image_urls(website_url)
    captions = {}
    total_images = min(len(img_urls), MAX_IMAGES)

    for idx, img_url in enumerate(img_urls):
        if len(captions) >= MAX_IMAGES:
            break

        if not img_url or "svg" in img_url or "1x1" in img_url:
            continue

        # Correct the URL if it is malformed
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif not img_url.startswith("http://") and not img_url.startswith("https://"):
            img_url = os.path.join(website_url, img_url)

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            sanitized_caption = sanitize_caption(caption)
            captions[img_url] = sanitized_caption
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed.")
    return captions


# Function to caption images from a folder
def caption_images_from_folder(folder_files, progress: gr.Progress):
    captions = {}
    total_images = len(folder_files)

    for idx, file in enumerate(folder_files):
        try:
            image = Image.open(file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            sanitized_caption = sanitize_caption(caption)
            captions[file.name] = sanitized_caption
        except Exception as e:
            print(f"Error processing image {file.name}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed.")
    return captions


# Function to save captions to a file
def save_captions(captions: dict, file_name: str):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            for img_path, caption in captions.items():
                file.write(f"{img_path}: {caption}\n")
        print(f"Captions saved to {file_name}")
    except Exception as e:
        print(f"Error saving captions: {e}")


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Automated Image Captioner")

    website_url_input = gr.Textbox(
        label="Enter Website URL", placeholder="https://example.com"
    )
    url_button = gr.Button("Generate Captions from URL")
    url_output = gr.Textbox(label="Generated Captions", interactive=False)

    folder_upload = gr.File(
        file_count="directory",
        type="filepath",
        label="Select a Folder",
        interactive=True,
    )
    folder_button = gr.Button("Generate Captions from Folder")
    folder_output = gr.Textbox(label="Generated Captions", interactive=False)

    # Button Click Events
    url_button.click(
        fn=caption_images_from_url,
        inputs=[website_url_input],
        outputs=url_output,
        show_progress=True,
    )

    folder_button.click(
        fn=lambda folder_files: caption_images_from_folder(folder_files, gr.Progress()),
        inputs=[folder_upload],
        outputs=folder_output,
        show_progress=True,
    )

demo.launch(share=True)
