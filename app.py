import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

model_dir = "./model/deepfake_vs_real_image_detection"

model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

processor = ViTImageProcessor.from_pretrained(model_dir)

print("Model and Processor loaded successfully!")

def predict(image):
    try:
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

        return {
            "Prediction": predicted_label,
            "Confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        return {"Error": str(e)}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Confidence Score")
    ],
    title="Deepfake vs Real Image Detection",
    description="Upload an image to check if it’s real or deepfake."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=5000)
