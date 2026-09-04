import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import keras
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from huggingface_hub import hf_hub_download
import zipfile
import json

# 1. Custom distance function
@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    sum_square = keras.ops.sum(keras.ops.square(x - y), axis=1, keepdims=True)
    return keras.ops.sqrt(keras.ops.maximum(sum_square, keras.backend.epsilon()))

# 2. Bulletproof Model Loader
@st.cache_resource
def load_siamese_model():
    model_path = hf_hub_download(repo_id="T0KII/signature-similarity", filename="signature_model.keras")
    patched_path = "patched_signature_model.keras"
    
    with zipfile.ZipFile(model_path, 'r') as zfin:
        with zfin.open('config.json') as f:
            config = json.load(f)
            
        def strip_quantization(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)
                for v in obj.values():
                    strip_quantization(v)
            elif isinstance(obj, list):
                for item in obj:
                    strip_quantization(item)
                    
        strip_quantization(config)
        
        with zipfile.ZipFile(patched_path, 'w') as zfout:
            for item in zfin.infolist():
                if item.filename == 'config.json':
                    zfout.writestr(item, json.dumps(config))
                else:
                    zfout.writestr(item, zfin.read(item.filename))
                    
    model = keras.models.load_model(
        patched_path,
        compile=False,
        custom_objects={"euclidean_distance": euclidean_distance}
    )
    return model

# 3. Preprocessing Function with Auto-Crop and Blur
def preprocess_canvas(canvas_data):
    # Convert RGBA numpy array to PIL Image
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA')
    background = Image.new('RGBA', img.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, img)
    
    img_gray = alpha_composite.convert('L')
    
    # Invert the image temporarily to find the bounding box of the dark ink
    inverted_image = ImageOps.invert(img_gray)
    bbox = inverted_image.getbbox()
    
    if bbox:
        # Crop to the exact ink bounds
        img_cropped = img_gray.crop(bbox)
        
        # Calculate dimensions for a perfect square to maintain aspect ratio
        w, h = img_cropped.size
        max_dim = max(w, h)
        
        # Create a new white square and paste the cropped signature in the center
        square_img = Image.new('L', (max_dim, max_dim), 255)
        offset = ((max_dim - w) // 2, (max_dim - h) // 2)
        square_img.paste(img_cropped, offset)
        
        # Add a slight blur and resize the now-perfect square
        img_blurred = square_img.filter(ImageFilter.GaussianBlur(radius=1))
        img_resized = img_blurred.resize((128, 128))
    else:
        # Fallback if canvas is empty
        img_resized = img_gray.resize((128, 128))
    
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    
    return img_array

# 4. App UI & Sidebar Configuration
st.set_page_config(page_title="Signature Verification System", layout="centered")

with st.sidebar:
    st.header("⚙️ Calibration")
    st.write("Adjust the decision boundary for the Siamese Network.")
    threshold = st.slider(
        "Match Threshold (Distance)", 
        min_value=0.1, 
        max_value=1.5, 
        value=0.65, 
        step=0.01,
        help="Higher values are more forgiving. Lower values require near-perfect pixel matches."
    )
    st.info("Because digital canvas strokes differ from scanned physical ink, the required threshold may be higher than during original training.")

st.title("Signature Verification System")
st.write("Draw two signatures below. The AI will output a similarity confidence based on your calibrated threshold.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Signature 1")
    canvas1 = st_canvas(
        stroke_width=6, stroke_color="#000000", background_color="#FFFFFF",
        height=250, width=300, drawing_mode="freedraw", return_image_data=True, key="canvas1",
    )

with col2:
    st.subheader("Signature 2")
    canvas2 = st_canvas(
        stroke_width=6, stroke_color="#000000", background_color="#FFFFFF",
        height=250, width=300, drawing_mode="freedraw", return_image_data=True, key="canvas2",
    )

# 5. Prediction & Metrics Logic
if st.button("Compare Signatures", type="primary", use_container_width=True):
    if canvas1.image_data is not None and canvas2.image_data is not None:
        
        sig1 = preprocess_canvas(canvas1.image_data)
        sig2 = preprocess_canvas(canvas2.image_data)
        
        with st.spinner("Analyzing geometric features..."):
            model = load_siamese_model()
            prediction = model.predict([sig1, sig2])
            distance = float(prediction[0][0])
            
            # Translating distance to an intuitive Accuracy/Similarity percentage
            similarity = max(0.0, (1.0 - (distance / 2.0))) * 100 
            
        st.divider()
        
        # Display polished metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Similarity Score", f"{similarity:.1f}%")
        m2.metric("Euclidean Distance", f"{distance:.4f}")
        m3.metric("Current Threshold", f"{threshold:.2f}")
        
        if distance <= threshold:
            st.success("✅ **Verification Passed:** The model predicts these signatures are from the same writer.")
        else:
            st.error("❌ **Verification Failed:** The model predicts these signatures are forged or from different writers.")