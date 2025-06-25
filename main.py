import os
import cv2
import numpy as np
import json
import pytesseract
import subprocess

INPUT_IMAGE = "hackathon_input_image/input__image.png"
OUTPUT_JSON = "output/result.json"
DEBUG_OUTPUT = "output/raw_llama_response.txt"
MODEL_NAME = "llama3"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    sharpened = cv2.filter2D(denoised, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)
    return enhanced

def extract_text(image_path):
    processed_img = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_img)
    return text

import re

def call_llama(prompt, label=""):
    try:
        proc = subprocess.Popen(
            ["ollama", "run", MODEL_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        stdout, stderr = proc.communicate(prompt, timeout=90)

        if proc.returncode != 0:
            print(f"Ollama error ({label}):\n", stderr)
            return None

        response = stdout.strip()

        # Save full response for debugging
        with open(DEBUG_OUTPUT, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- LLaMA response for {label} ---\n{response}\n")

        # Extract first JSON block from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            print(f"⚠️ Could not extract valid JSON from LLaMA response for {label}")
            return None

        json_block = json_match.group(0)

        return json.loads(json_block)

    except Exception as e:
        print(f"Error calling LLaMA ({label}): {e}")
        return None

def build_prompt_sections(text):
    prompt_1 = f"""
You are an expert medical report parser. Extract ONLY the following sections from the text and return as a JSON object:

- hospital_info: hospital_name, address, phone, website, accreditation, panel (as list), certificate_number, accreditation_date
- patient_info: lab_id, name, age, gender, client_code
- doctor_info: referred_by, consultant, pathologist, reporting_location
- report_info: report_type, collection_date, collection_time, received_date, received_time, report_date, report_time, sample_type

Return only a valid JSON object with these 4 sections.

Text:
{text}
"""

    prompt_2 = f"""
You are an expert medical report parser. Extract only the test results from the following text and return a JSON array with this structure:

"test_results": [
  {{
    "test_name": "...",
    "result_value": "...",
    "unit": "...",
    "reference_range": "...",
    "status": "...",
    "method": "..."
  }}
]

Return ONLY the array in a JSON object under the key "test_results".

Text:
{text}
"""
    return prompt_1, prompt_2

def merge_sections(section1, section2):
    merged = section1 if section1 else {}
    merged["test_results"] = section2.get("test_results", []) if section2 else []
    return merged

def main():
    print("🔍 Reading and processing image...")
    full_text = extract_text(INPUT_IMAGE)
    print("Extracted text length:", len(full_text))

    prompt1, prompt2 = build_prompt_sections(full_text)

    print("Calling LLaMA for header info...")
    section1 = call_llama(prompt1, label="Header Info")

    print("Calling LLaMA for test results...")
    section2 = call_llama(prompt2, label="Test Results")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    if section1 or section2:
        final_json = merge_sections(section1, section2)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

        print(f"Partial/Full JSON saved to {OUTPUT_JSON}")
    else:
        print("Both LLaMA calls failed. See raw response in:", DEBUG_OUTPUT)

if __name__ == "__main__":
    main()
