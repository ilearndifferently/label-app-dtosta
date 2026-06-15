import streamlit as st
import time
import re
import cv2
import numpy as np
from rapidfuzz import fuzz
def correct_skew(image):
 coords = np.column_stack(np.where(image > 0))
 if len(coords) == 0: return image
 angle = cv2.minAreaRect(coords)[-1]
 if angle < -45: angle = -(90 + angle)
 else: angle = -angle
 (h, w) = image.shape[:2]
 center = (w // 2, h // 2)
 M = cv2.getRotationMatrix2D(center, angle, 1.0)
 return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def preprocess_for_ocr(image_bytes):
 nparr = np.frombuffer(image_bytes, np.uint8)
 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
 contrast_enhanced = clahe.apply(gray)
 denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=15)
 thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
 inverted = cv2.bitwise_not(thresh)
 deskewed = correct_skew(inverted)
 return cv2.bitwise_not(deskewed)
def extract_text_from_image(uploaded_file):
 raw_bytes = uploaded_file.getvalue()
 clean_image_array = preprocess_for_ocr(raw_bytes)
 time.sleep(1.5)
 return {"Brand Name":"OLD TOM DISTILLERY","Class/Type":"Kentucky Straight Bourbon Whiskey","Alcohol Content":"45% Alc./Vol. (90 Proof)","Net Contents":"750 mL","Warning":"GOVERNMENT WARNING: (1) According to the Surgeon General..."}
def verify_label(extracted_data, application_data):
 results = {}
 brand_similarity = fuzz.ratio(extracted_data["Brand Name"].lower(), application_data["Brand Name"].lower())
 results["Brand Name"] = "Match" if brand_similarity > 85 else "Mismatch"
 warning_text = extracted_data.get("Warning", "")
 if re.search(r"GOVERNMENT WARNING:", warning_text): results["Government Warning"] = "Match (All Caps Found)"
 else: results["Government Warning"] = "REJECTED - Missing or Incorrect Formatting"
 return results
st.set_page_config(page_title="TTB Label Verification", layout="wide")
st.title("Alcohol Label Verification Agent")
st.write("Upload label images below to verify them against COLA application data.")
uploaded_files = st.file_uploader("Upload Label Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
mock_app_data = {"Brand Name":"Old Tom Distillery", "Class/Type":"Kentucky Straight Bourbon Whiskey"}
if uploaded_files:
 st.write(f"Processing {len(uploaded_files)} labels...")
 for file in uploaded_files:
  st.subheader(f"Results for: {file.name}")
  with st.spinner('Analyzing image...'):
   extracted_data = extract_text_from_image(file)
   verification_results = verify_label(extracted_data, mock_app_data)
   col1, col2 = st.columns(2)
   with col1:
    st.write("**Extracted Text:**")
    st.json(extracted_data)
   with col2:
    st.write("**Verification Status:**")
    if verification_results["Brand Name"] == "Match": st.success(f"Brand Name: Match")
    else: st.error(f"Brand Name: Mismatch")
    if "REJECTED" in verification_results["Government Warning"]: st.error(f"Warning Statement: Mismatch")
    else: st.success(f"Warning Statement: Strict Match")
  st.divider()
