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


# Function to generate caption for a single image
def generate_image_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        sanitized_caption = sanitize_caption(caption)
        return sanitized_caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Unable to generate caption"


# Function to process images from URL
def process_images_from_url(website_url):
    try:
        # Extract image URLs
        response = requests.get(website_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")

        # Process images
        captions = []
        for img in img_tags[:MAX_IMAGES]:
            if "src" in img.attrs:
                try:
                    # Construct full URL
                    img_url = img["src"]
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    elif img_url.startswith("/"):
                        img_url = requests.compat.urljoin(website_url, img_url)

                    # Download and process image
                    img_response = requests.get(img_url)
                    image = Image.open(BytesIO(img_response.content)).convert("RGB")

                    # Generate caption
                    caption = generate_image_caption(image)
                    captions.append(
                        {
                            "image": image,
                            "original_caption": caption,
                            "modified_caption": caption,
                        }
                    )
                except Exception as img_error:
                    print(f"Error processing image: {img_error}")

        return captions
    except Exception as e:
        print(f"Error processing URL: {e}")
        return []


# Function to process images from folder
def process_images_from_folder(folder_files):
    captions = []
    for file in folder_files[:MAX_IMAGES]:
        try:
            # Open image
            image_path = file.name
            image = Image.open(image_path).convert("RGB")

            # Generate caption
            caption = generate_image_caption(image)
            captions.append(
                {
                    "image": image,
                    "original_caption": caption,
                    "modified_caption": caption,
                }
            )
        except Exception as e:
            print(f"Error processing image {file.name}: {e}")

    return captions


# Function to save captions
def save_captions(captions, save_path, save_original=True):
    try:
        with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Original Caption", "Modified Caption"])

            for idx, caption_data in enumerate(captions, 1):
                filename = f"image_{idx}.jpg"
                original = caption_data["original_caption"]
                modified = caption_data["modified_caption"]

                writer.writerow([filename, original, modified])

        return f"Captions saved successfully to {save_path}"
    except Exception as e:
        return f"Error saving captions: {str(e)}"


# Create Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        # Shared state for captions and editable captions
        captions_state = gr.State([])
        editable_captions_state = gr.State([])

        # URL Tab
        with gr.Tab("Caption from URL"):
            url_input = gr.Textbox(label="Website URL")
            url_save_path = gr.Textbox(label="Save Path", value="url_captions.csv")
            url_save_original = gr.Checkbox(label="Save Original Captions", value=True)

            url_output_gallery = gr.Gallery(
                label="Generated Captions", columns=3, height=400
            )

            # Caption editing area
            with gr.Column() as url_caption_edit_area:
                url_caption_textboxes = []
                for _ in range(MAX_IMAGES):
                    tb = gr.Textbox(
                        label="Edit Caption", visible=False, interactive=True
                    )
                    url_caption_textboxes.append(tb)

            url_generate_btn = gr.Button("Generate Captions")
            url_save_btn = gr.Button("Save Captions", visible=False)
            url_status = gr.Textbox(label="Status")

            def process_url_images(url):
                # Generate captions
                captions = process_images_from_url(url)
                if not captions:
                    return [], [], "No images found or error occurred."

                # Prepare gallery data
                gallery_data = [
                    (cap["image"], cap["modified_caption"]) for cap in captions
                ]

                # Update caption textboxes
                caption_updates = []
                for idx, cap in enumerate(captions):
                    if idx < len(url_caption_textboxes):
                        caption_updates.append(
                            gr.Textbox(
                                value=cap["modified_caption"],
                                visible=True,
                                interactive=True,
                            )
                        )

                return (
                    gallery_data,  # Gallery
                    captions,  # Captions state
                    "Captions generated successfully.",  # Status
                    url_save_btn,  # Make save button visible
                )

            def update_captions(captions, *edited_captions):
                # Update captions with user edits
                updated_captions = captions.copy()
                for idx, caption in enumerate(edited_captions):
                    if idx < len(updated_captions):
                        updated_captions[idx]["modified_caption"] = caption
                return updated_captions

            url_generate_btn.click(
                process_url_images,
                inputs=url_input,
                outputs=[url_output_gallery, captions_state, url_status, url_save_btn],
            )

            # Dynamically update captions when edited
            for i, tb in enumerate(url_caption_textboxes):
                tb.change(
                    update_captions,
                    inputs=[captions_state] + url_caption_textboxes,
                    outputs=captions_state,
                )

            url_save_btn.click(
                save_captions,
                inputs=[captions_state, url_save_path, url_save_original],
                outputs=url_status,
            )

        # Folder Tab (Similar structure to URL tab)
        with gr.Tab("Caption from Folder"):
            folder_input = gr.File(
                file_count="multiple", type="filepath", label="Select Images"
            )
            folder_save_path = gr.Textbox(
                label="Save Path", value="folder_captions.csv"
            )
            folder_save_original = gr.Checkbox(
                label="Save Original Captions", value=True
            )

            folder_output_gallery = gr.Gallery(
                label="Generated Captions", columns=3, height=400
            )

            # Caption editing area
            with gr.Column() as folder_caption_edit_area:
                folder_caption_textboxes = []
                for _ in range(MAX_IMAGES):
                    tb = gr.Textbox(
                        label="Edit Caption", visible=False, interactive=True
                    )
                    folder_caption_textboxes.append(tb)

            folder_generate_btn = gr.Button("Generate Captions")
            folder_save_btn = gr.Button("Save Captions", visible=False)
            folder_status = gr.Textbox(label="Status")

            def process_folder_images(files):
                # Generate captions
                captions = process_images_from_folder(files)
                if not captions:
                    return [], [], "No images found or error occurred."

                # Prepare gallery data
                gallery_data = [
                    (cap["image"], cap["modified_caption"]) for cap in captions
                ]

                # Update caption textboxes
                caption_updates = []
                for idx, cap in enumerate(captions):
                    if idx < len(folder_caption_textboxes):
                        caption_updates.append(
                            gr.Textbox(
                                value=cap["modified_caption"],
                                visible=True,
                                interactive=True,
                            )
                        )

                return (
                    gallery_data,  # Gallery
                    captions,  # Captions state
                    "Captions generated successfully.",  # Status
                    folder_save_btn,  # Make save button visible
                )

            def update_folder_captions(captions, *edited_captions):
                # Update captions with user edits
                updated_captions = captions.copy()
                for idx, caption in enumerate(edited_captions):
                    if idx < len(updated_captions):
                        updated_captions[idx]["modified_caption"] = caption
                return updated_captions

            folder_generate_btn.click(
                process_folder_images,
                inputs=folder_input,
                outputs=[
                    folder_output_gallery,
                    captions_state,
                    folder_status,
                    folder_save_btn,
                ],
            )

            # Dynamically update captions when edited
            for i, tb in enumerate(folder_caption_textboxes):
                tb.change(
                    update_folder_captions,
                    inputs=[captions_state] + folder_caption_textboxes,
                    outputs=captions_state,
                )

            folder_save_btn.click(
                save_captions,
                inputs=[captions_state, folder_save_path, folder_save_original],
                outputs=folder_status,
            )

    return demo


# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
