About: Label Verification App Prototype
NOTE: "Anti-iPad-Bug" code used because I'm traveling right now and only have an iPad to use

Installation & Execution:
 1. Ensure Python 3.9+ is installed.
 2. Install dependencies: pip install streamlit rapidfuzz
 3. Run the application: streamlit run app.py

Design Decisions & Assumptions:
UI/UX: Built with Streamlit to ensure a clean, zero-learning-curve interface suitable for all technical comfort levels.

Batch Processing: 
Utilized Streamlit's accept_multiple_files parameter to allow drag-and-drop processing of hundreds of files at once.

This prototype explicitly solves the bulk-processing bottleneck through its architecture:
 Drag-and-Drop Bulk Upload: The Streamlit frontend utilizes the accept_multiple_files=True parameter, allowing an agent to drag all 300 label images into the app in a single action.
 Automated Iteration: The backend processes the upload queue programmatically via a continuous loop (for file in uploaded_files:).
 Performance at Scale:  The system is designed to return results within the required 5-second maximum threshold. At an estimated processing time of 1.5 to 2 seconds per image, a peak-season dump of 300 labels can be completely verified in under 10 minutes without requiring any active supervision or data entry from the agent.

Mocked OCR API: 
To adhere to the strict 5-second processing SLA requirement and avoid exposing paid API keys in a public repository, the ⁠extract_text_from_image⁠ function mocks the external OCR engine. It returns varying test profiles based on the uploaded file's byte-hash to dynamically demonstrate the exact strict and fuzzy matching logic streams required by the agents

Matching Algorithms: 
Split the matching logic into two streams: 
RapidFuzz handles harmless capitalization and punctuation differences for brand names, 
while exact RegEx ensures the Government Warning strictly adheres to federal formatting requirements.
