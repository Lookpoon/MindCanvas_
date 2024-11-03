import streamlit as st
import openai
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Initialize the BLIP model for image captioning
@st.cache_resource  # Cache the model to avoid reloading on each run
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()

# Display title and description
st.title("🖼️ Image Emotion Prediction")
st.write(
    "Upload an image – the app will predict the emotion it conveys using image captioning and GPT! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask the user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
model_choice = st.selectbox("Choose the GPT model:", options=["gpt-4", "gpt-3.5-turbo"])

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Set OpenAI API key
    openai.api_key = openai_api_key
    
    # Let the user upload an image file
    uploaded_image = st.file_uploader("Upload an image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate a caption for the image using BLIP
        with st.spinner("Generating image description..."):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            image_description = processor.decode(output[0], skip_special_tokens=True)

        st.write(f"Generated Description: *{image_description}*")

        # List of emotions to classify
        emotions = ["awe", "amusement", "contentment", "excitement", "disgust", "fear", "sadness"]

        # Prepare the prompt for the GPT model
        prompt = (
            f"Based on the following description of an image, classify the emotion it conveys "
            f"from these options: {', '.join(emotions)}.\n\n"
            f"Description: {image_description}\n\nEmotion:"
        )

        # Generate an emotion prediction using the OpenAI Chat API
        with st.spinner("Analyzing emotion..."):
            try:
                response = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.5
                )
                predicted_emotion = response.choices[0].message['content'].strip()
                st.success(f"Predicted Emotion: {predicted_emotion}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
