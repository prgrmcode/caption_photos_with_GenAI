import os
import re
import gradio as gr
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
import csv

# Constants
MAX_IMAGES = 20  # Maximum number of images to process/display

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)


# Function to sanitize captions
def sanitize_caption(caption: str) -> str:
    cleaned_caption = re.sub(r"\baraf\w*\b", "", caption).strip()
    return cleaned_caption


# Function to extract image URLs from a website
def extract_image_urls(website_url: str):
    try:
        response = requests.get(website_url, timeout=10)
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
                elif not img_url.startswith(("http://", "https://")):
                    img_url = requests.compat.urljoin(website_url, img_url)
                img_urls.append(img_url)
        return img_urls
    except Exception as e:
        print(f"Error extracting images from {website_url}: {e}")
        return []


# Function to generate caption for a single image
def generate_image_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        # Set max_new_tokens to control the length of the generation
        outputs = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        sanitized_caption = sanitize_caption(caption)
        return sanitized_caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Unable to generate caption"


# Function to process images from URL
def process_images_from_url(website_url, progress=gr.Progress()):
    img_urls = extract_image_urls(website_url)
    captions = []
    total_images = min(len(img_urls), MAX_IMAGES)

    for idx, img_url in enumerate(img_urls[:MAX_IMAGES]):
        if not img_url or "svg" in img_url.lower() or "1x1" in img_url.lower():
            continue

        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            caption = generate_image_caption(image)
            captions.append((image, caption))
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed from URL.")
    return captions


# Function to process images from selected folder
def process_images_from_folder(folder_files, progress=gr.Progress()):
    captions = []
    total_images = min(len(folder_files), MAX_IMAGES)

    for idx, file in enumerate(folder_files[:MAX_IMAGES]):
        try:
            # Access the file path to open the image
            image_path = file.name  # Use 'file.path' if necessary
            image = Image.open(image_path).convert("RGB")
            caption = generate_image_caption(image)
            captions.append((image, caption))
        except Exception as e:
            # Handle cases where 'name' attribute might not be available
            file_name = getattr(file, "name", "unknown_image.jpg")
            print(f"Error processing image {file_name}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed from folder.")
    return captions


# Function to save captions to a file
def save_captions(captions, save_path):
    try:
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Image Filename", "Original Caption", "Modified Caption"]
            )  # Header row
            for caption_pair in captions:
                # Handle both filepath and PIL.Image instances
                if isinstance(caption_pair[0], Image.Image):
                    filename = getattr(caption_pair[0], "filename", "unknown_image.jpg")
                else:
                    filename = os.path.basename(caption_pair[0])
                original = caption_pair[1]
                modified = caption_pair[1]
                writer.writerow([filename, original, modified])
        return f"Captions successfully saved to `{save_path}`."
    except Exception as e:
        return f"Error saving captions: {e}"


# Gradio Interface
def setup_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Advanced Image Captioning Tool")

        with gr.Tab("URL Captioning"):
            with gr.Row():
                website_url_input = gr.Textbox(label="Enter Website URL")
                url_button = gr.Button("Generate Captions")

            url_gallery = gr.Gallery(label="Captioned Images")
            url_button.click(
                fn=process_images_from_url,
                inputs=[website_url_input],
                outputs=[url_gallery],
                show_progress=True,
            )

        with gr.Tab("Folder Captioning"):
            with gr.Row():
                folder_upload = gr.File(
                    file_count="multiple", type="filepath", label="Select Images"
                )
                folder_button = gr.Button("Generate Captions")

            folder_gallery = gr.Gallery(label="Captioned Images")
            folder_status = gr.Markdown()  # Define folder_status here

            folder_button.click(
                fn=process_images_from_folder,
                inputs=[folder_upload],
                outputs=[folder_gallery],
                show_progress=True,
            )

            folder_captions = gr.State([])  # To store captions

            def folder_generate_captions(folder_files):
                if not folder_files:
                    return [], "### No files selected. Please upload images."

                captions = process_images_from_folder(folder_files)

                if not captions:
                    return [], "### No images found or an error occurred."

                # Prepare data for display with images and captions as tuples
                display_items = [(item[0], item[1]) for item in captions]

                return display_items, "### Images and captions loaded successfully."

            def update_caption(captions, index, edited_caption):
                if 0 <= index < len(captions):
                    captions[index] = (captions[index][0], edited_caption)
                return captions

            def update_display(captions):
                display_items = []
                for idx, (image, caption) in enumerate(captions):
                    display_items.append(
                        gr.Column(
                            gr.Image(value=image, label=f"Image {idx + 1}"),
                            gr.Textbox(
                                value=caption,
                                label="Edit Caption",
                                interactive=True,
                                elem_id=f"caption-{idx}",
                            ).submit(
                                fn=update_caption,
                                inputs=[folder_captions, idx, gr.Textbox.value],
                                outputs=[folder_captions],
                            ),
                        )
                    )
                return display_items

            folder_button.click(
                fn=folder_generate_captions,
                inputs=[folder_upload],
                outputs=[folder_gallery, folder_status],
                show_progress=True,
            )

            folder_save_path = gr.Textbox(
                label="Save Captions to File",
                placeholder="captions_folder.csv",
                value="captions_folder.csv",
                interactive=True,
            )
            folder_save_btn = gr.Button("Save Captions")

            def folder_save_captions_fn(captions, save_path):
                if not save_path:
                    return "### Please provide a valid filename to save captions."
                status = save_captions(captions, save_path)
                return f"### {status}"

            folder_save_btn.click(
                fn=folder_save_captions_fn,
                inputs=[folder_captions, folder_save_path],
                outputs=[folder_status],
            )

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.launch()
