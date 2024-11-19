import os
import re
import gradio as gr
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

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
        img_urls = []
        for img in img_tags:
            if "src" in img.attrs:
                img_url = img["src"]
                # Correct relative URLs to absolute URLs
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = requests.compat.urljoin(website_url, img_url)
                elif not img_url.startswith("http://") and not img_url.startswith(
                    "https://"
                ):
                    img_url = requests.compat.urljoin(website_url, img_url)
                img_urls.append(img_url)
        return img_urls
    except Exception as e:
        print(f"Error extracting images from {website_url}: {e}")
        return []


# Function to sanitize captions
def sanitize_caption(caption: str) -> str:
    cleaned_caption = re.sub(r"\baraf\w*\b", "", caption).strip()
    return cleaned_caption


# Function to caption images from URL
def caption_images_from_url(website_url: str, progress=gr.Progress()):
    img_urls = extract_image_urls(website_url)
    captions = []
    total_images = min(len(img_urls), MAX_IMAGES)

    for idx, img_url in enumerate(img_urls[:MAX_IMAGES]):
        if not img_url or "svg" in img_url or "1x1" in img_url:
            continue

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            sanitized_caption = sanitize_caption(caption)
            captions.append((image, sanitized_caption))
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed.")
    return captions


# Function to caption images from a folder
def caption_images_from_folder(folder_path, progress=gr.Progress()):
    captions = []
    allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    files = []

    # Recursively collect image files from the folder
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(allowed_extensions):
                files.append(os.path.join(root, filename))

    total_images = min(len(files), MAX_IMAGES)

    for idx, file_path in enumerate(files[:MAX_IMAGES]):
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            sanitized_caption = sanitize_caption(caption)
            captions.append((image, sanitized_caption))
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed.")
    return captions


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Captioning with Generative AI")

    with gr.Tab("From URL"):
        website_url_input = gr.Textbox(
            label="Enter Website URL", placeholder="https://example.com"
        )
        url_button = gr.Button("Generate Captions from URL")
        url_gallery = gr.Gallery(
            label="Generated Captions from URL", columns=1, height="auto"
        )
        url_status = gr.Markdown()

        def process_url(website_url):
            captions = caption_images_from_url(website_url)
            if captions:
                gallery_data = [[img, cap] for img, cap in captions]
                return gallery_data, "### Images and captions loaded successfully."
            else:
                return [], "### No images found or an error occurred."

        url_button.click(
            fn=process_url,
            inputs=[website_url_input],
            outputs=[url_gallery, url_status],
            show_progress=True,
        )

    with gr.Tab("From Folder"):
        folder_path_input = gr.Textbox(
            label="Enter Folder Path", placeholder="C:/path/to/your/folder"
        )
        folder_button = gr.Button("Generate Captions from Folder")
        folder_gallery = gr.Gallery(
            label="Generated Captions from Folder", columns=1, height="auto"
        )
        folder_status = gr.Markdown()

        def process_folder(folder_path):
            if not os.path.isdir(folder_path):
                return [], "### Invalid folder path."
            captions = caption_images_from_folder(folder_path)
            if captions:
                gallery_data = [[img, cap] for img, cap in captions]
                return gallery_data, "### Images and captions loaded successfully."
            else:
                return [], "### No images found or an error occurred."

        folder_button.click(
            fn=process_folder,
            inputs=[folder_path_input],
            outputs=[folder_gallery, folder_status],
            show_progress=True,
        )

demo.launch()
