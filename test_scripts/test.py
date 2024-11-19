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
            for img_url, caption in captions.items():
                file.write(f"{img_url}: {caption}\n")
        print(f"Captions created and saved to {file_name}")
        return f"<span style='color: green;'>**Captions saved successfully to `{file_name}`.**</span>"
    except Exception as e:
        print(f"Error saving captions: {e}")
        return "<span style='color: red;'>**Failed to save captions.**</span>"


# Function to process URL and display captions
def process_and_display(website_url, image_urls_state, progress: gr.Progress):
    try:
        captions = caption_images_from_url(website_url, progress=progress)
        img_display_updates = []
        caption_input_updates = []
        row_updates = []
        image_urls = []

        for i in range(MAX_IMAGES):
            if i < len(captions):
                img_url = list(captions.keys())[i]
                caption = captions[img_url]
                img_display_updates.append(gr.update(visible=True, value=img_url))
                caption_input_updates.append(gr.update(visible=True, value=caption))
                row_updates.append(gr.update(visible=True))
                image_urls.append(img_url)
            else:
                img_display_updates.append(gr.update(visible=False, value=None))
                caption_input_updates.append(gr.update(visible=False, value=""))
                row_updates.append(gr.update(visible=False))

        image_urls_state.value = image_urls

        return (
            "### Images and captions loaded.",
            *img_display_updates,
            *caption_input_updates,
            *row_updates,
            image_urls_state,
        )
    except Exception as e:
        return (
            f"Error: {e}",
            *([gr.update(visible=False, value=None)] * MAX_IMAGES),
            *([gr.update(visible=False, value="")] * MAX_IMAGES),
            *([gr.update(visible=False)] * MAX_IMAGES),
            [],
        )


# Function to process folder and display captions
def process_and_display_folder(folder_files, image_urls_state, progress: gr.Progress):
    if (
        not folder_files or len(folder_files) == 0
    ):  # Check if folder_files is None or empty
        return (
            "Error: No files selected.",
            *([gr.update(visible=False, value=None)] * MAX_IMAGES),
            *([gr.update(visible=False, value="")] * MAX_IMAGES),
            *([gr.update(visible=False)] * MAX_IMAGES),
            [],
        )

    try:
        captions = caption_images_from_folder(folder_files, progress=progress)
        img_display_updates = []
        caption_input_updates = []
        row_updates = []
        image_urls = []

        for i in range(MAX_IMAGES):
            if i < len(captions):
                img_path = list(captions.keys())[i]
                caption = captions[img_path]
                img_display_updates.append(gr.update(visible=True, value=img_path))
                caption_input_updates.append(gr.update(visible=True, value=caption))
                row_updates.append(gr.update(visible=True))
                image_urls.append(img_path)
            else:
                img_display_updates.append(gr.update(visible=False, value=None))
                caption_input_updates.append(gr.update(visible=False, value=""))
                row_updates.append(gr.update(visible=False))

        image_urls_state.value = image_urls

        return (
            "### Images and captions loaded.",
            *img_display_updates,
            *caption_input_updates,
            *row_updates,
            image_urls_state,
        )
    except Exception as e:
        return (
            f"Error: {e}",
            *([gr.update(visible=False, value=None)] * MAX_IMAGES),
            *([gr.update(visible=False, value="")] * MAX_IMAGES),
            *([gr.update(visible=False)] * MAX_IMAGES),
            [],
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
    image_urls_state = []
    save_status = ""
    return (
        "",
        *img_display_updates,
        *caption_input_updates,
        *row_updates,
        image_urls_state,
        save_status,
    )


# Function to save modified captions
def save_modified_captions(*args):
    *captions_list, image_urls, file_name = args
    captions = {}
    for i in range(len(image_urls)):
        captions[image_urls[i]] = captions_list[i]
    save_status = save_captions(captions, file_name)
    return save_status


# Initialize Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Automated Image Captioner")

    with gr.Row():
        website_url = gr.Textbox(
            label="Enter Website URL", placeholder="https://example.com"
        )
        start_button = gr.Button("Generate Captions from URL")
        folder_button = gr.Button("Generate Captions from Folder")
        clear_button = gr.Button("Clear")

    generate_status = gr.Markdown("")  # Status for generating captions
    save_status = gr.Markdown("")  # Status for saving captions

    # Initialize image displays and caption inputs
    img_displays = [
        gr.Image(label=f"Image {i+1}", visible=False) for i in range(MAX_IMAGES)
    ]
    caption_inputs = [
        gr.Textbox(label=f"Caption {i+1}", lines=2, visible=False)
        for i in range(MAX_IMAGES)
    ]
    rows = [gr.Row([img_displays[i], caption_inputs[i]]) for i in range(MAX_IMAGES)]

    # State to store image URLs
    image_urls_state = gr.State([])

    with gr.Column():
        save_button = gr.Button("Save Captions")
        file_name_input = gr.Textbox(label="Enter file name to save captions")

    # File input for folder selection
    folder_upload = gr.File(
        file_count="directory",  # Accept directories
        type="filepath",
        label="Select a Folder",
        interactive=True,
        visible=True,  # Make this visible
    )

    # Button Click Events
    start_button.click(
        fn=process_and_display,
        inputs=[website_url, image_urls_state],  # Remove gr.Progress() from inputs
        outputs=[generate_status]
        + img_displays
        + caption_inputs
        + rows
        + [image_urls_state],
        show_progress=True,
    )

    # Button Click Events
    folder_button.click(
        fn=process_and_display_folder,  # Call function directly
        inputs=[folder_upload],  # Pass selected folder as input
        outputs=[generate_status]
        + img_displays
        + caption_inputs
        + rows
        + [image_urls_state],
        show_progress=True,
    )

    folder_upload.change(
        fn=process_and_display_folder,
        inputs=[folder_upload, image_urls_state],  # Remove gr.Progress() from inputs
        outputs=[generate_status]
        + img_displays
        + caption_inputs
        + rows
        + [image_urls_state],
        show_progress=True,
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
        inputs=caption_inputs + [image_urls_state, file_name_input],
        outputs=[save_status],
        show_progress=True,
    )

demo.launch(share=True)
