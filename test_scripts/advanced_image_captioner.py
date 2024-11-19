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
import base64

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
            captions.append(
                {
                    "image": image,
                    "original_caption": caption,
                    "modified_caption": caption,
                }
            )
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
            captions.append(
                {
                    "image": image,
                    "original_caption": caption,
                    "modified_caption": caption,
                }
            )
        except Exception as e:
            # Handle cases where 'name' attribute might not be available
            file_name = getattr(file, "name", "unknown_image.jpg")
            print(f"Error processing image {file_name}: {e}")
            continue

        progress((idx + 1) / total_images)

    print("Operation finished: All images have been processed from folder.")
    return captions


# Function to save captions to a file
def save_captions(captions, save_path, save_original):
    try:
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Image Filename", "Original Caption", "Modified Caption"]
            )  # Header row
            for caption_pair in captions:
                # Handle both filepath and PIL.Image instances
                if isinstance(caption_pair["image"], Image.Image):
                    filename = getattr(
                        caption_pair["image"], "filename", "unknown_image.jpg"
                    )
                else:
                    filename = os.path.basename(caption_pair["image"])
                original = caption_pair["original_caption"]
                modified = caption_pair["modified_caption"]
                writer.writerow([filename, original, modified])
        return f"Captions successfully saved to `{save_path}`."
    except Exception as e:
        return f"Error saving captions: {e}"


# Function to update captions based on user edits
def update_captions(captions, edited_captions):
    for idx, edited_caption in enumerate(edited_captions):
        if idx < len(captions):
            captions[idx]["modified_caption"] = edited_caption[0]
    return captions


# Create the Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🖼️ Advanced Image Captioning Tool")

        with gr.Tab("Caption from URL"):
            with gr.Row():
                website_url_input = gr.Textbox(
                    label="Enter Website URL",
                    placeholder="https://example.com",
                    interactive=True,
                )
                url_generate_btn = gr.Button("Generate Captions")

            with gr.Column():
                url_gallery = gr.Gallery(
                    label="Generated Captions from URL",
                    show_label=False,
                    columns=2,
                    height="auto",
                )
                url_captions = gr.State([])  # To store captions
                url_edit_captions = gr.Dataframe(
                    headers=["Edited Captions"],
                    datatype=["str"],
                    interactive=True,
                    row_count=(MAX_IMAGES, 1),
                    visible=False,
                )
                url_save_path = gr.Textbox(
                    label="Save Captions to File",
                    placeholder="captions_url.csv",
                    value="captions_url.csv",
                    interactive=True,
                )
                url_save_original_checkbox = gr.Checkbox(
                    label="Save Original Captions", value=True
                )
                url_save_btn = gr.Button("Save Captions")
                url_status = gr.Markdown()

            def url_generate_captions(website_url):
                if not website_url.startswith(("http://", "https://")):
                    return (
                        [],
                        [],
                        "### Please enter a valid URL starting with http:// or https://",
                    )
                captions = process_images_from_url(website_url)
                if not captions:
                    return [], [], "### No images found or an error occurred."
                # Prepare data for `gr.Gallery` with images and captions as tuples
                gallery_images = [
                    (item["image"], item["modified_caption"]) for item in captions
                ]
                return (
                    gallery_images,
                    captions,
                    "### Images and captions loaded successfully.",
                )

            url_generate_btn.click(
                fn=url_generate_captions,
                inputs=[website_url_input],
                outputs=[url_gallery, url_captions, url_status],
                show_progress=True,
            )

            def load_url_editable_captions(captions):
                edited = [[caption["modified_caption"]] for caption in captions]
                return gr.Dataframe.update(visible=True, value=edited)

            with gr.Accordion("Edit Captions", open=False):
                with gr.Row():
                    edit_captions_btn = gr.Button("Load Editable Captions")

                edit_captions_btn.click(
                    fn=load_url_editable_captions,
                    inputs=[url_captions],
                    outputs=[url_edit_captions],
                )

                save_edited_btn = gr.Button("Save Edited Captions")

                save_edited_btn.click(
                    fn=update_captions,
                    inputs=[url_captions, url_edit_captions],
                    outputs=[url_captions],
                )

            def url_save_captions_fn(captions, save_path, save_original):
                if not save_path:
                    return "### Please provide a valid filename to save captions."
                status = save_captions(captions, save_path, save_original)
                return f"### {status}"

            url_save_btn.click(
                fn=url_save_captions_fn,
                inputs=[url_captions, url_save_path, url_save_original_checkbox],
                outputs=[url_status],
            )

        with gr.Tab("Caption from Folder"):
            with gr.Row():
                folder_upload = gr.File(
                    file_count="multiple",
                    type="filepath",
                    label="Select Images",
                    interactive=True,
                    file_types=["image"],
                )
                folder_generate_btn = gr.Button("Generate Captions")

            # Define lists to store the image and caption components
            folder_image_components = []
            folder_caption_components = []

            with gr.Column():
                for i in range(MAX_IMAGES):
                    with gr.Row():
                        image_component = gr.Image(
                            label=f"Image {i+1}", interactive=False
                        )
                        caption_component = gr.Textbox(
                            label=f"Caption {i+1}", interactive=True
                        )
                        folder_image_components.append(image_component)
                        folder_caption_components.append(caption_component)

            folder_save_path = gr.Textbox(
                label="Save Captions to File",
                placeholder="captions_folder.csv",
                value="captions_folder.csv",
                interactive=True,
            )
            folder_save_btn = gr.Button("Save Captions")
            folder_status = gr.Markdown()

            def folder_generate_captions(folder_files):
                if not folder_files:
                    return (
                        [gr.update() for _ in folder_image_components],
                        [gr.update() for _ in folder_caption_components],
                        "### No files selected. Please upload images.",
                    )
                captions = process_images_from_folder(folder_files)
                if not captions:
                    return (
                        [gr.update() for _ in folder_image_components],
                        [gr.update() for _ in folder_caption_components],
                        "### No images found or an error occurred.",
                    )

                image_updates = []
                caption_updates = []
                for i in range(MAX_IMAGES):
                    if i < len(captions):
                        image_updates.append(gr.update(value=captions[i]["image"]))
                        caption_updates.append(
                            gr.update(value=captions[i]["modified_caption"])
                        )
                    else:
                        image_updates.append(gr.update(value=None))
                        caption_updates.append(gr.update(value=""))
                return (
                    image_updates
                    + caption_updates
                    + ["### Images and captions loaded successfully."]
                )

            folder_generate_btn.click(
                fn=folder_generate_captions,
                inputs=[folder_upload],
                outputs=folder_image_components
                + folder_caption_components
                + [folder_status],
                show_progress=True,
            )

            def folder_save_captions_fn(
                image_components, caption_components, save_path
            ):
                captions = []
                for img_comp, cap_comp in zip(image_components, caption_components):
                    if img_comp.value is not None:
                        image = img_comp.value
                        caption = cap_comp.value
                        captions.append(
                            {
                                "image": image,
                                "modified_caption": caption,
                                "original_caption": caption,  # Adjust if you track original captions separately
                            }
                        )
                if not save_path:
                    return "### Please provide a valid filename to save captions."
                status = save_captions(captions, save_path, save_original=True)
                return f"### {status}"

            folder_save_btn.click(
                fn=folder_save_captions_fn,
                inputs=folder_image_components
                + folder_caption_components
                + [folder_save_path],
                outputs=[folder_status],
            )

        gr.Markdown(
            """
        ---
        **Note:**
        - Ensure that the images you upload are in supported formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`).
        - The BLIP model is resource-intensive. For optimal performance, use a machine with sufficient memory and a compatible GPU.
        """
        )

    return demo


# Launch the application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
