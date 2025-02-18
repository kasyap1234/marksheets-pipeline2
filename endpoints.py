
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import shutil
import os
from comparison import ocr_output, clean_vlm_data_for_comparison, compare_and_output

app = FastAPI()

@app.post("/process")
async def process_marksheet(image: UploadFile = File(...), prompt: str = Form(...)):
    """
    Endpoint to process a marksheet image.
    The flow is:
      1. Save the uploaded image temporarily.
      2. Run OCR to extract words along with bounding boxes.
      3. Run the VLM model using the provided prompt, and clean its output.
      4. For every word in the VLM output, perform fuzzy ratio matching against the OCR words.
      5. Return the list of matches with the OCR bounding box and fuzzy ratio.
    
    Request Form-data:
      image: UploadFile - the marksheet image file.
      prompt: str - prompt for the VLM model to extract specific words.
    
    Returns:
      JSON with the list of matched words.
    """
    # Save image temporarily
    temp_file_path = f"temp_{image.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")
    
    try:
        # Call ocr_output to get OCR data.
        # ocr_output returns a list of dictionaries with "geometry" and "value" keys.
        ocr_list = ocr_output(temp_file_path)
        # Convert the list of OCR dictionaries into a formatted string with each entry in the form:
        # "((x1, y1), (x2, y2)): word"
        ocr_data = "\n".join([f"{entry['geometry']}: {entry['value']}" for entry in ocr_list])
        
        # Get cleaned VLM data (a list of words) using the provided prompt.
        cleaned_vlm = clean_vlm_data_for_comparison(temp_file_path, prompt)
        
        # Compare every word from the cleaned VLM output with the OCR output using fuzzy matching.
        result = compare_and_output(ocr_data, cleaned_vlm)
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    
    os.remove(temp_file_path)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)