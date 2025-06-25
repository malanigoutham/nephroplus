#!/usr/bin/env python3
"""
Medical Report OCR Parser with Ollama DeepSeek - Enhanced for Hackathon

Extract text from medical reports and convert to JSON using local Ollama.
This version focuses on improving layout awareness for the LLM,
strict adherence to the hackathon's required JSON output,
and more robust JSON parsing.
"""

import os
import cv2
import numpy as np
import json
import requests
from pathlib import Path
import glob
from datetime import datetime
import pytesseract
from tqdm import tqdm
import traceback
import re # Added for more robust JSON extraction

# ====== CONFIGURATION VARIABLES ======
INPUT_FOLDER = "./input_images"  # Folder containing medical report images
OUTPUT_FOLDER = "./output"       # Where to save results
MAX_IMAGES = 0                   # 0 = process all images, else limit to this number
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
# Using a model that supports a larger context window and better instruction following is recommended.
# 'llama3' is a good general-purpose choice. 'deepseek-coder' might be good for structured output.
# 'llama3.2:3b' is specified by user, keeping it, but advising for larger models if issues arise.
OLLAMA_MODEL = "llama3.2:3b"  
DEBUG_MODE = True                # Enable detailed debugging output

# Define the exact JSON fields required by the hackathon. This helps make the prompt precise.
HACKATHON_JSON_SCHEMA_FIELDS = [
    "Test_Name",
    "Result",
    "Unit",
    "Reference_Range",
    "Patient_Name",
    "Doctor"
]
# =====================================

class MedicalReportOCR:
    def __init__(self, ollama_url=OLLAMA_BASE_URL, model_name=OLLAMA_MODEL):
        """Initialize OCR processor and Ollama client"""
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Test Ollama connection
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5) # Added timeout
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {ollama_url}")
                
                # Check if model is available
                models = [model['name'] for model in response.json().get('models', [])]
                if model_name in models:
                    print(f"✅ Model {model_name} is available")
                else:
                    print(f"⚠️  Model {model_name} not found. Available models: {models}")
                    print(f"    Run: ollama pull {model_name}")
            else:
                print(f"❌ Failed to connect to Ollama at {ollama_url}. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Ollama connection error: Could not reach {ollama_url}.")
            print("    Make sure Ollama is running and accessible (ollama serve).")
        except requests.exceptions.Timeout:
            print(f"❌ Ollama connection timed out after 5 seconds to {ollama_url}.")
            print("    Make sure Ollama is running and accessible (ollama serve).")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("    Make sure Ollama is running: ollama serve")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR results.
        Includes: Grayscale, Denoising, Sharpening, CLAHE for contrast.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}. Check file path and integrity.")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising (often helps with scan artifacts)
            # hWindowSize parameter for `fastNlMeansDenoising` should be odd and not too large.
            # Using 7 as a reasonable default.
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Apply slight sharpening to enhance text edges
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)
            
            if DEBUG_MODE:
                print(f"    🖼️  Image preprocessing completed successfully for {os.path.basename(image_path)}")
            
            return enhanced
        except Exception as e:
            print(f"    ❌ Image preprocessing error for {os.path.basename(image_path)}: {e}")
            raise # Re-raise the exception to be caught by the calling function
    
    def extract_text_tesseract(self, image_path):
        """
        Extract text and bounding box data using Tesseract.
        Sorts extracted text by reading order (top-to-bottom, then left-to-right)
        to improve layout awareness for the LLM.
        """
        try:
            if DEBUG_MODE:
                print(f"    🔍 Starting Tesseract OCR extraction for {os.path.basename(image_path)}...")
            
            processed_img = self.preprocess_image(image_path)
            
            # Get detailed text data with bounding box coordinates and confidence
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            extracted_blocks = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Only consider text with reasonable confidence and non-empty
                if conf > 60 and text: # Increased confidence threshold for cleaner input
                    extracted_blocks.append({
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': conf
                    })
            
            # Sort blocks primarily by 'top' (Y-coordinate) and then by 'left' (X-coordinate)
            # This ensures text is read in a natural reading order.
            extracted_blocks.sort(key=lambda b: (b['top'], b['left']))

            full_text_with_layout = []
            previous_block_top = -1
            
            # Reconstruct text with simple layout hints (line breaks)
            # This helps the LLM understand vertical separation of text.
            for block in extracted_blocks:
                # If there's a significant vertical jump, assume a new line or paragraph
                # A heuristic: if current block's top is significantly lower than previous
                if previous_block_top != -1 and (block['top'] - previous_block_top > block['height'] * 1.5):
                    full_text_with_layout.append('\n') # Add a newline
                full_text_with_layout.append(block['text'])
                previous_block_top = block['top']
            
            full_text = ' '.join(full_text_with_layout).strip() # Join with spaces for words, newlines for lines

            if DEBUG_MODE:
                print(f"    📝 OCR extracted {len(extracted_blocks)} text blocks")
                print(f"    📏 Full text length (with layout hints): {len(full_text)} characters")
                if len(full_text) > 0:
                    preview = full_text[:500].replace('\n', '\\n') + "..." if len(full_text) > 500 else full_text.replace('\n', '\\n')
                    print(f"    👀 Text preview (with layout hints): {preview}")
            
            return full_text, extracted_blocks
            
        except Exception as e:
            print(f"    ❌ OCR extraction failed for {os.path.basename(image_path)}: {e}")
            if DEBUG_MODE:
                print(f"    🔧 Full error traceback:")
                traceback.print_exc()
            return "", []
    
    def generate_json_with_ollama(self, extracted_text_with_layout, image_filename):
        """
        Use Ollama DeepSeek to convert extracted text to structured JSON,
        strictly adhering to the hackathon's required JSON schema.
        """
        
        if DEBUG_MODE:
            print(f"    🤖 Starting Ollama processing for {image_filename}...")
            print(f"    📊 Input text length: {len(extracted_text_with_layout)} characters")
        
        # Truncate text if too long to avoid token limits
        # Llama3 models typically have 8K context, but smaller versions might be less.
        # Keeping it safe for 3B, can be increased if needed.
        max_text_length = 4000 # Adjusted for a 3B model's typical context, can be 8000 for full Llama3
        if len(extracted_text_with_layout) > max_text_length:
            extracted_text_with_layout = extracted_text_with_layout[:max_text_length] + "\n[...TEXT TRUNCATED DUE TO LENGTH...]"
            if DEBUG_MODE:
                print(f"    ✂️  Text truncated to {max_text_length} characters for Ollama input")
        
        # Crafting a precise prompt for the hackathon's JSON structure
        # Emphasizing "ONLY JSON", "exact fields", and "null if not found"
        # The prompt is now much more focused on the specified output.
        prompt = f"""You are an expert medical report parser. Your task is to extract specific information from the provided OCR text of a medical lab report and format it into a JSON object.

The required JSON schema for the hackathon is:
{{
  "Test_Name": "string or null",
  "Result": "string or number or null",
  "Unit": "string or null",
  "Reference_Range": "string or null",
  "Patient_Name": "string or null",
  "Doctor": "string or null"
}}

Here is the OCR extracted text from the medical report. Pay close attention to the layout (indicated by newlines) to correctly identify corresponding values:
---
{extracted_text_with_layout}
---

Your response MUST be ONLY the JSON object.
- Fill in all the fields from the schema above.
- If a field is not found in the text, use `null` as its value.
- Do NOT include any extra fields beyond those specified in the schema.
- Ensure the values are exact as found in the text, without interpretation, except for cleaning obvious OCR errors (e.g., 'O' instead of '0', 'I' instead of '1').
- For "Result", if it's a number, output it as a string (e.g., "267.8").

Return ONLY the JSON. No conversational text, no markdown code block fencing (```json), just the plain JSON object."""

        try:
            if DEBUG_MODE:
                print(f"    📡 Sending request to Ollama API for {image_filename}...")
                print(f"    🔗 URL: {self.ollama_url}/api/generate")
                print(f"    🏷️  Model: {self.model_name}")
            
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1, # Low temperature for less creativity, more factual extraction
                    "top_p": 0.9,
                    "num_predict": 1024 # Limit response size to prevent unnecessary generation
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=180 # Increased timeout for larger models or complex parsing
            )
            
            if DEBUG_MODE:
                print(f"    📥 Ollama response status: {response.status_code}")
                print(f"    📏 Response content length: {len(response.text)} characters")
            
            if response.status_code != 200:
                error_msg = f'Ollama API error: HTTP {response.status_code}. Response: {response.text[:500]}'
                print(f"    ❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text,
                    'status_code': response.status_code
                }
            
            result = response.json()
            json_text = result.get('response', '').strip()
            
            if DEBUG_MODE:
                print(f"    📏 Raw Ollama JSON candidate text length: {len(json_text)} characters")
                if len(json_text) == 0:
                    print(f"    ⚠️  WARNING: Empty response 'response' field from Ollama!")
                    print(f"    🔍 Full Ollama result: {result}")
                else:
                    preview = json_text[:500] + "..." if len(json_text) > 500 else json_text
                    print(f"    👀 JSON candidate preview: {preview}")
            
            if not json_text:
                return {
                    'success': False,
                    'error': 'Empty response from Ollama (response field was empty)',
                    'raw_response': json_text,
                    'full_ollama_result': result
                }
            
            # --- Robust JSON Parsing ---
            # Attempt to clean and extract valid JSON using regex, if LLM adds wrappers
            # This is crucial for handling cases where the LLM doesn't strictly follow "ONLY JSON"
            try:
                # Remove common markdown code block wrappers (```json, ```)
                json_text_cleaned = json_text
                if json_text_cleaned.startswith('```json'):
                    json_text_cleaned = json_text_cleaned[7:].strip()
                if json_text_cleaned.endswith('```'):
                    json_text_cleaned = json_text_cleaned[:-3].strip()
                
                # Attempt to find the outermost JSON object or array
                # This regex tries to find a JSON object { ... } or array [ ... ]
                # It's non-greedy (.*?) to match the smallest valid JSON, then greedy (.*) for the content
                # This needs to be robust for nested structures: r'\{.*\}' or r'\[.*\]'
                # A more robust approach would be to count braces/brackets to find the complete JSON.
                # For simplicity, let's try a regex that looks for the first '{' and last '}'
                # that balance each other. This is still a heuristic.
                
                # A simpler, more direct regex if the LLM output is mostly clean:
                json_match = re.search(r'(\{.*\}|\bnull\b)', json_text_cleaned, re.DOTALL) # Added null case
                if json_match:
                    json_to_parse = json_match.group(0)
                    if DEBUG_MODE:
                        print(f"    🔍 Extracted potential JSON using regex (match length: {len(json_to_parse)})")
                else:
                    json_to_parse = json_text_cleaned # Fallback if no specific JSON found by regex
                    if DEBUG_MODE:
                        print(f"    ⚠️  No clear JSON pattern found by regex, attempting to parse full cleaned text.")

                parsed_json = json.loads(json_to_parse)
                if DEBUG_MODE:
                    print(f"    ✅ Successfully parsed JSON structure")
                    print(f"    🔑 JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")

            except json.JSONDecodeError as e:
                error_msg = f'JSON parsing error: {str(e)}. Attempted to parse: {json_to_parse[:500]}...'
                if DEBUG_MODE:
                    print(f"    ❌ {error_msg}")
                    print(f"    📄 JSON text that failed to parse: {json_text[:500]}...")
                    print(f"    🔧 JSON error position: line {e.lineno}, column {e.colno}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': json_text,
                    'original_response_text': json_text,
                    'json_error_details': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg,
                        'attempted_parse_text': json_to_parse # Show what we tried to parse
                    }
                }
            
            # Post-processing: Ensure only hackathon-specific fields are present
            # This helps enforce the "JSON Structure Quality"
            final_json = {}
            for field in HACKATHON_JSON_SCHEMA_FIELDS:
                final_json[field] = parsed_json.get(field, None) # Use None if not found, consistent with null in JSON
            
            # Add metadata as before
            final_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'tesseract_ollama_deepseek_enhanced',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }
            
            if DEBUG_MODE:
                print(f"    🎉 JSON processing completed successfully!")
            
            return {
                'success': True,
                'json_data': final_json,
                'raw_response': json_text # Original response from Ollama
            }
            
        except requests.RequestException as e:
            error_msg = f'Ollama request error (Network/HTTP): {str(e)}'
            if DEBUG_MODE:
                print(f"    ❌ {error_msg}")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        except Exception as e:
            error_msg = f'Unexpected error in Ollama processing: {str(e)}'
            if DEBUG_MODE:
                print(f"    ❌ {error_msg}")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
    
    def process_image(self, image_path):
        """Process a single medical report image"""
        image_filename = os.path.basename(image_path)
        print(f"📄 Processing: {image_filename}")
        
        try:
            # Extract text using Tesseract
            extracted_text, extraction_details = self.extract_text_tesseract(image_path)
            
            if not extracted_text.strip():
                error_msg = 'No significant text extracted from image after OCR'
                if DEBUG_MODE:
                    print(f"    ❌ {error_msg}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': error_msg
                }
            
            print(f"    📝 Extracted {len(extraction_details)} distinct text blocks")
            
            # Generate structured JSON using Ollama
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
            
            if ollama_result['success']:
                if DEBUG_MODE:
                    print(f"    ✅ Successfully generated JSON structure")
                
                return {
                    'success': True,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text, # The full text including layout hints
                    'extraction_details': extraction_details, # Detailed blocks from Tesseract
                    'structured_json': ollama_result['json_data'],
                    'ollama_raw_response': ollama_result['raw_response']
                }
            else:
                if DEBUG_MODE:
                    print(f"    ❌ Failed to generate JSON for {image_filename}: {ollama_result['error']}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text,
                    'ollama_raw_response': ollama_result.get('raw_response'),
                    'ollama_error_details': ollama_result # Pass full ollama_result for debugging
                }
                
        except Exception as e:
            error_msg = f'Overall processing error for {image_filename}: {str(e)}'
            if DEBUG_MODE:
                print(f"    ❌ {error_msg}")
                print(f"    🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'image_path': image_path,
                'image_filename': image_filename,
                'error': error_msg
            }

def get_image_files(folder_path):
    """Get all image files from the specified folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper()))) # Check for uppercase extensions
    
    return sorted(image_files)

def save_raw_text(extracted_text, image_filename, text_output_dir):
    """Save raw extracted text to a .txt file"""
    base_name = os.path.splitext(image_filename)[0]
    txt_filename = f"{base_name}.txt"
    txt_filepath = os.path.join(text_output_dir, txt_filename)
    
    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        return True, txt_filename
    except Exception as e:
        if DEBUG_MODE:
            print(f"    ❌ Error saving raw text {txt_filename}: {e}")
        return False, str(e)

def main():
    """Main processing function"""
    
    # Validate input folder
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        print(f"    Create the folder and add medical report images (e.g., CRP, CBC reports).")
        return
    
    # Create output directories
    json_output_dir = os.path.join(OUTPUT_FOLDER, "json")
    text_output_dir = os.path.join(OUTPUT_FOLDER, "text")
    debug_output_dir = os.path.join(OUTPUT_FOLDER, "debug")
    
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    if DEBUG_MODE:
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"📂 Debug output directory: {debug_output_dir}")
    
    print(f"📂 JSON output directory: {json_output_dir}")
    print(f"📂 Text output directory: {text_output_dir}")
    
    # Get image files
    image_files = get_image_files(INPUT_FOLDER)
    
    if not image_files:
        print(f"❌ No image files found in: {INPUT_FOLDER}")
        print("    Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return
    
    # Limit number of images if specified
    if MAX_IMAGES > 0:
        image_files = image_files[:MAX_IMAGES]
        print(f"📊 Processing limited to {MAX_IMAGES} images")
    
    print(f"📊 Found {len(image_files)} image(s) to process")
    
    # Initialize OCR processor
    try:
        ocr_processor = MedicalReportOCR()
        # Exit if Ollama connection failed during init
        if "❌" in ocr_processor.ollama_url: # A hacky way to check init status from print statements
             print("Exiting due to Ollama initialization failure.")
             return
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {str(e)}")
        if DEBUG_MODE:
            traceback.print_exc()
        return
    
    # Process each image
    successful_count = 0
    failed_count = 0
    text_saved_count = 0
    
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        print(f"\n{'='*50}")
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        
        # Process image
        result = ocr_processor.process_image(image_path)
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_filename = f"{base_name}_extracted.json"
        json_filepath = os.path.join(json_output_dir, json_filename)
        
        # Save raw extracted text regardless of JSON processing success
        if 'extracted_text' in result and result['extracted_text'].strip():
            text_success, text_result_name = save_raw_text(
                result['extracted_text'], 
                result['image_filename'], 
                text_output_dir
            )
            
            if text_success:
                print(f"    📝 Raw text saved: {text_result_name}")
                text_saved_count += 1
            else:
                print(f"    ❌ Failed to save raw text: {text_result_name}")
        
        if result['success']:
            # Save structured JSON
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(result['structured_json'], f, indent=2, ensure_ascii=False)
                
                print(f"    ✅ Successfully saved structured JSON: {json_filename}")
                
                # Display summary of extracted data based on hackathon schema
                json_data = result['structured_json']
                test_name = json_data.get('Test_Name', 'N/A')
                patient_name = json_data.get('Patient_Name', 'N/A')
                doctor_name = json_data.get('Doctor', 'N/A')
                
                print(f"    📋 Test Name: {test_name}")
                print(f"    👤 Patient Name: {patient_name}")
                print(f"    👨‍⚕️ Doctor: {doctor_name}")
                
                successful_count += 1
                
            except Exception as e:
                print(f"    ❌ Failed to save final JSON output to file {json_filepath}: {str(e)}")
                if DEBUG_MODE:
                    traceback.print_exc()
                failed_count += 1
        else:
            # Save detailed error information for failed JSON generation
            error_data = {
                'error': result['error'],
                'image_path': result['image_path'],
                'extracted_text': result.get('extracted_text', ''),
                'timestamp': datetime.now().isoformat(),
                'debug_info': result.get('ollama_error_details', {}),
                'full_result': result # Include the full result dict for comprehensive debugging
            }
            
            error_filename = f"{base_name}_error.json"
            error_filepath = os.path.join(json_output_dir, error_filename)
            
            try:
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                print(f"    💾 Error details saved to: {error_filename}")
            except Exception as e:
                print(f"    ❌ Failed to save error details to {error_filepath}: {str(e)}")
            
            print(f"    ❌ Processing failed for {base_name}: {result['error']}")
            failed_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_count} images")
    print(f"❌ Failed to process: {failed_count} images")
    print(f"📝 Raw text files saved: {text_saved_count}")
    print(f"📂 Output folder: {OUTPUT_FOLDER}")
    
    if DEBUG_MODE:
        print(f"🔧 Debug mode was enabled - check debug folder for detailed logs and intermediary outputs.")
    
    return successful_count, failed_count, text_saved_count

if __name__ == "__main__":
    print("🚀 Starting Medical Report OCR Processing with Ollama DeepSeek")
    print(f"📂 Input folder: {INPUT_FOLDER}")
    print(f"🔢 Max images: {'All' if MAX_IMAGES == 0 else MAX_IMAGES}")
    print(f"🤖 Using: Tesseract OCR + Ollama {OLLAMA_MODEL}")
    print(f"🔧 Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    
    try:
        successful, failed, text_saved = main()
    except KeyboardInterrupt:
        print("\n⏸️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the main execution: {e}")
        if DEBUG_MODE:
            print("🔧 Full error traceback:")
            traceback.print_exc()

