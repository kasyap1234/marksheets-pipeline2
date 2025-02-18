import bentoml 
from bentoml.models import HuggingFaceModel 
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from PIL.Image import Image 
import json
from typing import Dict, Any 
from PIL import ImageDraw
import numpy as np
import io
model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
@bentoml.service(resources={"gpu" : 1 },traffic={"timeout": 10000})
class OCR: 
    model_ocr_path = HuggingFaceModel("vikp/surya_rec2")
    llm_path = HuggingFaceModel(model_path)

    def __init__(self):
        from surya.recognition import RecognitionPredictor 
        from surya.detection import DetectionPredictor 
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.langs = ["en"]
        
        # Initialize Qwen 2.5 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    @bentoml.api
    def ocrprocess(self, image: Image) -> Dict[str, Any]:
        predictions = self.recognition_predictor([image], [self.langs], self.detection_predictor)
        return {"ocr_output": predictions}

    @bentoml.api
    def extract_info(self, ocr_text: str) -> Dict[str, Any]:
        prompt = f"""Given the OCR output below, extract the student name, father's name, school name, and subject marks along with their bounding boxes. Return the information in the following JSON format:
        {{
            "student_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "father_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "school_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "subject_marks": [
                {{
                    "subject": "subject name",
                    "marks": "marks obtained",
                    "bounding_box": [coordinates]
                }},
                {{
                    "subject": "subject name",
                    "marks": "marks obtained",
                    "bounding_box": [coordinates]
                }}
            ],
            "total_marks": {{
                "value": "total marks",
                "bounding_box": [coordinates]
            }}
        }}

        OCR Output:
        {ocr_text}
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_length=1500,  
            temperature=0.2,
            num_return_sequences=1
        )

    @bentoml.api
    def process_marksheet(self, image: Image) -> Dict[str, Any]:
        """
        Combined endpoint that performs OCR and information extraction in one step
        """
        # Step 1: Perform OCR
        ocr_predictions = self.recognition_predictor([image], [self.langs], self.detection_predictor)
        
        # Step 2: Format OCR output for the prompt
        # ocr_text = json.dumps(ocr_predictions, indent=2)
        ocr_text=ocr_predictions
        # Step 3: Create prompt for information extraction
        prompt = f"""Given the OCR output below, extract the student name, father's name, school name, and subject marks along with their bounding boxes. Return the information in the following JSON format:
        {{
            "student_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "father_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "school_name": {{
                "value": "extracted name",
                "bounding_box": [coordinates]
            }},
            "subject_marks": [
                {{
                    "subject": "subject name",
                    "marks": "marks obtained",
                    "bounding_box": [coordinates]
                }}
            ],
            "total_marks": {{
                "value": "total marks",
                "bounding_box": [coordinates]
            }}
        }}

        OCR Output:
        {ocr_text}
        """
        
        # Step 4: Generate response using Qwen model
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_length=1500,
            temperature=0.2,
            num_return_sequences=1
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Step 5: Parse and return results
        try:
            # extracted_info = json.loads(response)
            return {
                "ocr_output": ocr_predictions,
                "extracted_info": response
            }
        except json.JSONDecodeError:
            return {
                "ocr_output": ocr_predictions,
                "error": "Failed to parse extracted information",
                "raw_response": response
            }

    @bentoml.api()
    def visualize_boxes(self, extracted_info: Dict[str, Any]) -> Image:
        """
        Takes extracted information with bounding boxes and draws them on the original image
        """
        # Create a copy of the image to draw on
        image = self.current_image.copy()
        draw = ImageDraw.Draw(image)
        
        # Color mapping for different types of information
        colors = {
            "student_name": "#FF0000",  # Red
            "father_name": "#00FF00",   # Green
            "school_name": "#0000FF",   # Blue
            "subject_marks": "#FFA500",  # Orange
            "total_marks": "#800080"    # Purple
        }
        
        # Draw boxes for main fields
        for field in ["student_name", "father_name", "school_name"]:
            if field in extracted_info and "bounding_box" in extracted_info[field]:
                bbox = extracted_info[field]["bounding_box"]
                value = extracted_info[field]["value"]
                draw.rectangle(bbox, outline=colors[field], width=2)
                # Add label above the box
                draw.text((bbox[0], bbox[1]-15), f"{field}: {value}", 
                         fill=colors[field])
        
        # Draw boxes for subject marks
        if "subject_marks" in extracted_info:
            for subject in extracted_info["subject_marks"]:
                if "bounding_box" in subject:
                    bbox = subject["bounding_box"]
                    label = f"{subject['subject']}: {subject['marks']}"
                    draw.rectangle(bbox, outline=colors["subject_marks"], width=2)
                    draw.text((bbox[0], bbox[1]-15), label, 
                             fill=colors["subject_marks"])
        
        # Draw box for total marks if present
        if "total_marks" in extracted_info and "bounding_box" in extracted_info["total_marks"]:
            bbox = extracted_info["total_marks"]["bounding_box"]
            value = extracted_info["total_marks"]["value"]
            draw.rectangle(bbox, outline=colors["total_marks"], width=2)
            draw.text((bbox[0], bbox[1]-15), f"Total: {value}", 
                     fill=colors["total_marks"])
        
        return image






