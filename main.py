import os
import cv2
import numpy as np
import json
import pytesseract
import subprocess
from datetime import datetime


INPUT_IMAGE = "hackathon_input_image/input_image.jpeg"
OUTPUT_JSON = "output/result.json"
MODEL_NAME = "llama3"
DEBUG = True


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
    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
    lines = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            text = data['text'][i].strip()
            if text:
                lines.append(text)
    return ' '.join(lines)

def create_prompt(extracted_text):
    return f"""
You are an expert medical report parser. Extract the following information from the text below and output as a JSON object:

- hospital_info: hospital_name, address, phone, website, accreditation, panel (as list), certificate_number, accreditation_date
- patient_info: lab_id, name, age, gender, client_code
- doctor_info: referred_by, consultant, pathologist, reporting_location
- report_info: report_type, collection_date, collection_time, received_date, received_time, report_date, report_time, sample_type
- test_results: test_name, result_value, unit, reference_range, status, method

Return only the JSON object. Do not add any explanations or notes.

Text:
{extracted_text}
"""


def call_llama(prompt):
    result = subprocess.run(["ollama", "run", MODEL_NAME], input=prompt, capture_output=True, text=True)
    response = result.stdout
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON. Check LLM response:")
        print(response[:500])
        return None


def main():
    print("🔍 Reading and processing image...")
    extracted_text = extract_text(INPUT_IMAGE)
    print("📏 Extracted text length:", len(extracted_text))

    print("🤖 Calling LLaMA with prompt...")
    prompt = create_prompt(extracted_text)
    parsed_json = call_llama(prompt)

    if parsed_json:
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2)
        print(f"✅ Output saved to {OUTPUT_JSON}")
    else:
        print("❌ Failed to produce valid JSON output")

if __name__ == "__main__":
    main()
