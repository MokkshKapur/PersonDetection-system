from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import os
import pandas as pd  # Import pandas for DataFrame handling

# Initialize the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Move the model to the correct device
model.to(device)

# Directory containing images
image_directory = "Testing Frames"  # Replace with your actual directory path

# Messages template for the model (the same as your original)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "number", "text": "Provide a detailed description of the person's clothing, their appearance, colors, patterns, and any other visible details"}
        ]
    }
]

# Initialize an empty list to store the data for DataFrame
data = []

# Loop through the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # Resize the image to a smaller resolution (example: 224x224)
        image = image.resize((512, 512))  # Adjust as needed to balance quality and memory usage

        # Prepare the text prompt
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process the inputs
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # Move inputs to the appropriate device
        inputs = inputs.to(device)

        # Generate output
        output_ids = model.generate(**inputs, max_new_tokens=1024)

        # Extract generated text
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Print the output text (or save it to a file, depending on your preference)
        print(f"Output for {filename}:")
        print(output_text)
        print("-" * 50)

        # Append the filename and output text to the data list
        data.append({"image_filename": filename, "description": output_text[0]})

        # Clear the GPU cache after each iteration to reduce memory usage
        torch.cuda.empty_cache()

# Convert the data list to a DataFrame
df = pd.DataFrame(data)
print(df)
# Save the DataFrame to a CSV file
df.to_csv("image_descriptions.csv", index=False)

# Confirm the DataFrame has been saved
print("\nDescriptions saved to 'image_descriptions.csv'")
