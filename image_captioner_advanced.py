# Last version: 2024-11-19
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations

import re
import gradio as gr
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
import csv
from tqdm import tqdm


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


# Function to process images from selected folder
def process_images_from_folder(folder_files, progress=gr.Progress(track_tqdm=True)):
    captions = []
    total_images = min(len(folder_files), MAX_IMAGES)

    # for idx, file in enumerate(folder_files[:MAX_IMAGES]):
    for idx, file in tqdm(enumerate(folder_files[:MAX_IMAGES]), total=total_images):
        try:
            # Access the file path to open the image
            image_path = file.name  # Use 'file.path' if necessary
            image = Image.open(image_path).convert("RGB")
            caption = generate_image_caption(image)
            captions.append(
                (image_path, caption)
            )  # Store the file path with the caption
        except Exception as e:
            # Handle cases where 'name' attribute might not be available
            file_name = getattr(file, "name", "unknown_image.jpg")
            print(f"Error processing image {file_name}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed from folder.")
    return captions


# Function to save captions to file
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


# Function to clear the interface
def clear_interface():
    img_display_updates = [
        gr.update(visible=False, value=None) for _ in range(MAX_IMAGES)
    ]
    caption_input_updates = [
        gr.update(visible=False, value="") for _ in range(MAX_IMAGES)
    ]
    row_updates = [gr.update(visible=False) for _ in range(MAX_IMAGES)]
    return (
        "",
        *img_display_updates,
        *caption_input_updates,
        *row_updates,
        "",
        "",  # Clear generate_status
        "",  # Clear save_status
    )


# Gradio Interface
def setup_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Advanced Image Captioning Tool")

        with gr.Tab("URL Captioning"):
            with gr.Row():
                website_url_input = gr.Textbox(label="Enter Website URL")
                url_button = gr.Button("Generate Captions")
                clear_button = gr.Button("Clear")

            generate_status = gr.Markdown("")  # Status for generating captions
            save_status = gr.Markdown("")  # Status for saving captions

            progress_bar = gr.Progress(
                track_tqdm=True
            )  # Progress bar for captioning process
            url_gallery = gr.Column()
            url_status = gr.Markdown()  # Define url_status here

            # Predefine components for images and captions
            url_image_components = []
            url_caption_components = []
            rows = []

            with url_gallery:
                for idx in range(MAX_IMAGES):
                    with gr.Row(visible=False) as row:
                        image = gr.Image(
                            label=f"Image {idx+1}", height=250, visible=False
                        )
                        caption = gr.Textbox(
                            label=f"Caption {idx+1}", lines=2, visible=False
                        )
                        rows.append(row)
                        url_image_components.append(image)
                        url_caption_components.append(caption)

            url_captions = gr.State([])  # To store captions

            # Function to process images from URL
            def process_images_from_url(
                website_url, progress=gr.Progress(track_tqdm=True)
            ):
                # Clear previous image URLs to avoid duplication
                url_captions.value = []

                img_urls = extract_image_urls(website_url)
                captions = {}
                total_images = min(len(img_urls), MAX_IMAGES)

                # for idx, img_url in enumerate(img_urls[:MAX_IMAGES]):
                for idx, img_url in tqdm(
                    enumerate(img_urls[:MAX_IMAGES]), total=total_images
                ):
                    if (
                        not img_url
                        or "svg" in img_url.lower()
                        or "1x1" in img_url.lower()
                    ):
                        continue

                    try:
                        response = requests.get(img_url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        caption = generate_image_caption(image)
                        captions[img_url] = (
                            caption  # Store the URL and caption in the dictionary
                        )
                    except Exception as e:
                        print(f"Error processing image {img_url}: {e}")
                        continue

                    progress((idx + 1) / total_images)

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
                    url_captions.value.append(img_url)  # Store image URL in the list
                    i += 1

                print("Operation finished: All images have been processed from URL.")
                return (
                    "### Images and captions loaded.",
                    *img_display_updates,
                    *caption_input_updates,
                    *row_updates,
                    url_captions,
                )

            url_button.click(
                fn=process_images_from_url,
                inputs=[website_url_input],
                outputs=[generate_status]
                + url_image_components
                + url_caption_components
                + rows
                + [url_captions],
                show_progress="full",
            )

            url_save_path = gr.Textbox(
                label="Save Captions to File",
                placeholder="captions_url.txt",
                value="captions_url.txt",
                interactive=True,
            )
            url_save_btn = gr.Button("Save Captions")

            def url_save_captions_fn(*args):
                *captions_list, url_captions, file_name = args
                if not file_name:
                    return "### Please provide a valid filename to save captions."
                captions = {}
                for i in range(len(url_captions.value)):
                    image_url = url_captions.value[i]  # Use the URL directly
                    captions[image_url] = captions_list[i]  # Extract the caption value

                status = save_captions(captions, file_name)
                return f"### {status}"

            url_save_btn.click(
                fn=url_save_captions_fn,
                inputs=url_caption_components + [url_captions, url_save_path],
                outputs=[url_status],
            )

            clear_button.click(
                fn=clear_interface,
                outputs=[generate_status]
                + url_image_components
                + url_caption_components
                + rows
                + [url_captions]
                + [save_status],  # Clear save_status as well
            )

        with gr.Tab("Folder Captioning"):
            with gr.Row():
                folder_upload = gr.File(
                    file_count="multiple", type="filepath", label="Select Images"
                )
                folder_button = gr.Button("Generate Captions")
                clear_button_folder = gr.Button("Clear")

            folder_status = gr.Markdown()  # Status for generating captions
            # folder_status = gr.Markdown("")  # Status for generating captions
            folder_gallery = gr.Column()
            progress_bar_folder = gr.Progress(
                track_tqdm=True
            )  # Progress bar for folder captioning process
            save_status = gr.Markdown("")  # Status for saving captions

            # Initialize lists to hold component references
            caption_inputs = []
            img_displays = []
            rows = []

            with folder_gallery:
                for idx in range(MAX_IMAGES):
                    with gr.Row(visible=False) as row:
                        image = gr.Image(
                            label=f"Image {idx + 1}", height=250, interactive=False
                        )
                        caption = gr.Textbox(
                            label=f"Caption {idx+1}", lines=2, interactive=True
                        )
                        rows.append(row)
                        img_displays.append(image)
                        caption_inputs.append(caption)

            # Gradio State to store image URLs and folder path
            image_urls_state = gr.State([])

            folder_captions = gr.State([])  # To store captions

            def folder_generate_captions(
                folder_files, progress=gr.Progress(track_tqdm=True)
            ):
                # Clear previous image URLs to avoid duplication
                image_urls_state.value = []
                if not folder_files:
                    return [], "### No files selected. Please upload images."

                captions = process_images_from_folder(folder_files, progress)

                if not captions:
                    return [], "### No images found or an error occurred."

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
                for img_path, caption in captions:
                    if i >= MAX_IMAGES:
                        break
                    img_display_updates[i] = gr.update(visible=True, value=img_path)
                    caption_input_updates[i] = gr.update(visible=True, value=caption)
                    row_updates[i] = gr.update(visible=True)
                    image_urls_state.value.append(
                        img_path
                    )  # Store image path in the list
                    i += 1

                return (
                    "### Images and captions loaded.",
                    *img_display_updates,
                    *caption_input_updates,
                    *row_updates,
                    image_urls_state,
                )

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
                outputs=[folder_status]
                + img_displays
                + caption_inputs
                + rows
                + [image_urls_state],
                show_progress=True,
            )

            folder_save_path = gr.Textbox(
                label="Save Captions to File",
                placeholder="captions_folder.txt",
                value="captions_folder.txt",
                interactive=True,
            )
            folder_save_btn = gr.Button("Save Captions")

            def folder_save_captions_fn(*args):
                *captions_list, image_urls_state, file_name = args
                if not file_name:
                    return "### Please provide a valid filename to save captions."
                captions = {}
                for i in range(len(image_urls_state.value)):
                    image_path = image_urls_state.value[i]  # Use the file path directly
                    image_name = os.path.basename(image_path)  # Extract the file name
                    captions[image_name] = captions_list[i]
                save_status = save_captions(captions, file_name)
                return save_status

            folder_save_btn.click(
                fn=folder_save_captions_fn,
                inputs=caption_inputs + [image_urls_state, folder_save_path],
                outputs=[save_status],
                show_progress=True,
            )

            clear_button_folder.click(
                fn=clear_interface,
                outputs=[folder_status]
                + img_displays
                + caption_inputs
                + rows
                + [image_urls_state]
                + [save_status],  # Clear save_status as well
            )

    return demo


# Launch the interface
if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.launch()
