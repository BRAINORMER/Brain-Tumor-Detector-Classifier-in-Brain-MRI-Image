import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Define the classes
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Perform detection and return results above threshold
def detect_and_plot(image, model):
    results = model.predict(image)[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    detected_classes = set()

    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()

        if conf >= 0.5:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f"{classes[int(cls)]} {conf:.2f}",
                     color='white', fontsize=12, backgroundcolor='red')
            detected_classes.add(classes[int(cls)])

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf, detected_classes

# Function to get response from Gemini model for tumor information
def get_gemini_response(tumor_type):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Provide a brief overview of the {tumor_type} brain tumor. Include the following information:
    1. Brief description
    2. Common symptoms
    3. Typical treatment options
    
    Format the response in markdown with appropriate headers."""
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to handle user queries using Gemini API
def get_gemini_response_for_query(user_query):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a helpful and knowledgeable assistant specialized in brain health, brain tumors, and neuroscience. 
Your goal is to assist users with accurate, accessible medical information related to:

- Brain tumors (types, symptoms, treatments, diagnoses)
- Brain anatomy and functions
- Brain-related diseases and disorders
- Neurological health and care

Only reject queries that are **clearly unrelated** to the brain or medical topics. For example, if a user asks about sports scores, technology trends, or travel tips, respond with:

"I'm sorry, I cannot answer questions not related to human brain"

Always format your response with helpful headings, bullet points, and markdown for readability.

**User's question:** {user_query}"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit app setup
st.set_page_config(page_title="Brain Tumor Detector & Classifier in Brain MRI Image", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Brain Tumor Detector & Classifier in Brain MRI Image</h1>", unsafe_allow_html=True)

# Initialize session state for chat visibility and history
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Image upload and processing section
st.subheader("Upload MRI Image:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image using PIL
    image = Image.open(uploaded_image)
    
    # Convert image to RGB format & rezise to 640x640
    image = image.convert("RGB")
    image = image.resize((640, 640))

    st.subheader("Uploaded MRI Image:")
    st.image(image, caption='Uploaded Image (Resized to 640x640)', use_container_width=True)

    # Convert PIL image to a format suitable for YOLO model
    image_np = np.array(image)
    
    # Load the YOLO model
    model_path = "BRAIN_TUMOR_DETECTOR_model.pt"  # Update this path to your model
    model = load_model(model_path)
    
    if model is not None:
        # Perform detection and get the result plot
        result_plot, detected_tumor_types = detect_and_plot(image_np, model)
        
        # Display the result plot in Streamlit
        st.subheader("Detection Results:")
        st.image(result_plot, caption='Detection Results', use_container_width=True)
        
        if detected_tumor_types:
            if "No Tumor" in detected_tumor_types and len(detected_tumor_types) == 1:
                st.subheader("No tumor detected in the image.")

            elif "Pituitary" in detected_tumor_types: 
                st.subheader("Pituitary Gland detected in the image.")

            else:
                if "No Tumor" in detected_tumor_types:
                    detected_tumor_types.remove("No Tumor")
                    
                if "Pituitary" in detected_tumor_types:
                    detected_tumor_types.remove("No Tumor")

                st.subheader("Detected Tumor Types:")
                for tumor in detected_tumor_types:
                    st.markdown(f"- **{tumor}**")

                st.markdown("---")
                st.subheader("Tumor Information:")

                for tumor in detected_tumor_types:
                    st.markdown(f"### {tumor}")
                    tumor_info = get_gemini_response(tumor)
                    st.markdown(tumor_info)
                    st.markdown("---")
        else:
            st.subheader("No confident detection made. Please try with a clearer image.")

# Chat assistant section
if st.session_state.chat_visible:
    st.title("Brain Tumor Chat Assistant")
    st.write("Ask me anything about brain tumors or related topics!")

    # Display chat history with formatting for user and bot messages
    for user_input, bot_response in st.session_state.chat_history:
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")
    
    # User input field for chat
    user_input = st.text_input("Enter your query here:", key="chat_input")

    if user_input:
        bot_response = get_gemini_response_for_query(user_input)
        
        # Append the new query and response to the chat history
        st.session_state.chat_history.append((user_input, bot_response))
        
        # Display the new message immediately
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Bot:** {bot_response}")
        st.markdown("---")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_visible = False

else:
    # Button to start the chat
    if st.button("Start Chat Assistant"):
        st.session_state.chat_visible = True

# Force a rerun of the script to update the UI
st.empty()