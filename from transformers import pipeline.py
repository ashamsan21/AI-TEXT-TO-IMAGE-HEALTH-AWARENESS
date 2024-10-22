from transformers import pipeline

def generate_image(prompt: str):
    # Load the pre-trained stable diffusion model from Hugging Face
    model = pipeline('text-to-image', model='CompVis/stable-diffusion-v1-4')

    # Generate image based on the prompt
    images = model(prompt)
    return images

if __name__ == "__main__":
    health_prompt = "Infographic showing the importance of washing hands to prevent illness"
    images = generate_image(health_prompt)

    # Save or display images
    for i, img in enumerate(images):
        img['image'].save(f'output_image_{i}.png')
        print(f"Saved: output_image_{i}.png")
