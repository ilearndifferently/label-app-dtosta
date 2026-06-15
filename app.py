import streamlit as st
import time
import re
import cv2
import numpy as np
from rapidfuzz import fuzz

# ==========================================
# 1. IMAGE PREPROCESSING (OpenCV)
# ==========================================

def correct_skew(image):
    """Corrects weird angles by detecting text orientation."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(image_bytes):
    """Cleans image of glare and bad lighting for better OCR extraction."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Fix Lighting (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=15)

    # Fix Shadows/Glare (Adaptive Thresholding)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Deskewing
    inverted = cv2.bitwise_not(thresh)
    deskewed = correct_skew(inverted)
    final_image = cv2.bitwise_not(deskewed)

    return final_image

# ==========================================
# 2. OCR EXTRACTION & VERIFICATION LOGIC
# ==========================================

def extract_text_from_image(uploaded_file):
    """Mocks an OCR API call (e.g., Google Cloud Vision)."""
    # In production, pass the preprocessed image to the OCR engine.
    raw_bytes = uploaded_file.getvalue()
    clean_image_array = preprocess_for_ocr(raw_bytes)
    
    # Simulating fast API processing time
    time.sleep(1.5) 
    
    return {
        "Brand Name": "OLD TOM DISTILLERY",
        "Class/Type": "Kentucky Straight Bourbon Whiskey",
        "Alcohol Content": "45% Alc./Vol. (90 Proof)",
        "Net Contents": "750 mL",
        "Warning": "GOVERNMENT WARNING: (1) According to the Surgeon General..."
    }

def verify_label(extracted_data, application_data):
    """Compares extracted label text against COLA application data."""
    results = {}
    
    # Nuanced / Fuzzy Matching
    brand_similarity = fuzz.ratio(extracted_data["Brand Name"].lower(), application_data["Brand Name"].lower())
    results["Brand Name"] = "Match" if brand_similarity > 85 else "Mismatch"

    # Strict Matching for Government Warning
    warning_text = extracted_data.get("Warning", "")
    if re.search(r"GOVERNMENT WARNING:", warning_text):
        results["Government Warning"] = "Match (All Caps Found)"
    else:
        results["Government Warning"] = "REJECTED - Missing or Incorrect Formatting"
        
    return results

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="TTB Label Verification", layout="wide")

st.title("Alcohol Label Verification Agent")
st.write("Upload label images below to verify them against COLA application data.")

# File Uploader
uploaded_files = st.file_uploader("Upload Label Images (Batch upload supported)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Mock Application Data (Usually pulled from the database)
mock_app_data = {
    "Brand Name": "Old Tom Distillery", 
    "Class/Type": "Kentucky Straight Bourbon Whiskey"
}

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} labels...")
    
    for file in uploaded_files:
        st.subheader(f"Results for: {file.name}")
        with st.spinner('Analyzing image...'):
            
            # Run Extraction and Verification
            extracted_data = extract_text_from_image(file)
            verification_results = verify_label(extracted_data, mock_app_data)
            
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Extracted Text from Label:**")
                st.json(extracted_data)
            with col2:
                st.write("**Verification Status:**")
                if verification_results["Brand Name"] == "Match":
                    st.success(f"Brand Name: {verification_results['Brand Name']} (Nuanced Match)")
                else:
                    st.error(f"Brand Name: {verification_results['Brand Name']}")
                    
                if "REJECTED" in verification_results["Government Warning"]:
                    st.error(f"Warning Statement: {verification_results['Government Warning']}")
                else:
                    st.success(f"Warning Statement: {verification_results['Government Warning']} (Strict Match)")
        st.divider()
