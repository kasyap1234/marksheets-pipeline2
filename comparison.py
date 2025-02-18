from processor import *
import re
from fuzzywuzzy import fuzz

def ocr_output(image_path, word_list=[]):
    return ocr_processor(image_path, word_list)



def vlm_output(image_path, prompt):
    inference_output = inference(image_path, prompt)
    return inference_output

def clean_vlm_data_for_comparison(image_path, prompt):
    words = extract_words_to_highlight(vlm_output(image_path, prompt))
    return words_highlighted(words)

def compare_and_output(ocr_data, clean_vlm_data, threshold=80):
    """
    For every word in the cleaned VLM output, finds the matching
    word from the OCR output using fuzzy ratio matching and returns
    a list containing the matched words along with their OCR bounding boxes.
    
    Parameters:
      ocr_data (str): The raw OCR output string with bounding boxes.
                      Example entry: "((x1, y1), (x2, y2)): word"
      clean_vlm_data (list or str): VLM words (or a whitespace-separated string of words)
      threshold (int): The minimum fuzzy ratio required for a match (default 80).





    Returns:
      list: List with dictionaries for each matched VLM word that exists in the OCR output.
            Each dictionary contains:
              - 'vlm_word': the VLM word being searched,
              - 'ocr_word': the matched OCR word,
              - 'bounding_box': the OCR bounding box as a string,
              - 'fuzzy_ratio': the matching fuzzy ratio.
    """
    results = []

    # Ensure clean_vlm_data is a list (split if provided as a string)
    if isinstance(clean_vlm_data, str):
        clean_vlm_list = clean_vlm_data.split()
    else:
        clean_vlm_list = clean_vlm_data

    # Regex pattern to extract OCR bounding box and word.
    # It expects entries like: ((num, num), (num, num)): word
    pattern = r"(\(\([^()]+,[^()]+\),\s*\([^()]+,[^()]+\)\)):\s*(\S+)"
    ocr_entries = re.findall(pattern, ocr_data)

    # Process every word in the cleaned VLM output.
    for vlm_word in clean_vlm_list:
        best_match = None
        best_ratio = 0
        best_bbox = None
        # Compare against each OCR entry.
        for bbox, ocr_word in ocr_entries:
            ratio = fuzz.ratio(vlm_word.lower(), ocr_word.lower())
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_match = ocr_word
                best_bbox = bbox
        if best_match:
            results.append({
                'vlm_word': vlm_word,
                'ocr_word': best_match,
                'bounding_box': best_bbox,
                'fuzzy_ratio': best_ratio
            })
    return results
