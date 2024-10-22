# Basic setup to explore text-to-image generation
from transformers import pipeline

# Load the text-to-image model
model = pipeline('text-to-image', model='CompVis/stable-diffusion-v1-4')

# Try various health-related prompts
prompt = "A poster encouraging people to reduce air pollution"
image = model(prompt)

# Display the image (if using Jupyter)
image[0]['image'].show()

