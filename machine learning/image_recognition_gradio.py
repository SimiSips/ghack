# Import libararies
import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np


model = load_model("deephospital_model") # load the model

# Labels
labels = ["Pneumonia Negative", "Pneumonia Positive", "Benign", "Malignant"]

def classify_image(inp):
  inp = inp.reshape((-1, 150, 150, 3))
  prediction = model.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(4)}

image = gr.inputs.Image(shape=(150, 150))
label = gr.outputs.Label(num_top_classes=4)

gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch()