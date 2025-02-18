from doctr.io import DocumentFile 
from doctr.io import ocr_predictor 
import torch
import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from IPython.display import Markdown, display
from openai import OpenAI
import os
import base64
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
def ocr_processor(image_path,word_list=[]): 
    predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',pretrained=True)
    img=DocumentFile.from_images(image_path)
    result=predictor(img)
    output=result.export()
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:
                print("{}: {}".format(obj3["geometry"],obj3["value"]))
                word_list.append({"geometry": obj3["geometry"],"value": obj3["value"]})

    return word_list
 



def parse_json(json_output):
    """
    Parses a JSON string embedded in Markdown fencing (```json ... ```).
    Returns a Python dictionary or list.
    """
    # Remove Markdown fencing if present
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    
    # Parse the JSON string into a Python object
    try:
        return json.loads(json_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

# @title inference function
def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
    checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(checkpoint)
    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
    # image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]
    



#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

prompt='''Spotting  the following information from the provided image:
- Student's Name 
- Father's Name
- Individual Subject Marks
- Total Marks (if present)

Return the extracted information  in   JSON format : 

{
    "student_name": {
      "value": "John Doe",
      
    },
    "father_name": {
      "value": "Jane Doe",
      
    },
    "subject_marks": [
      {
        "subject": "Mathematics",
        "marks": 85,
   
      },
      {
        "subject": "Science",
        "marks": 90,
  
      },
      {
        "subject": "English",
        "marks": 78,
       
      }
    ],
    
  }


'''
response, inputs = inference(image_path, prompt, return_input=True)
display(Markdown(response)) #response of llm 
    

# output =parse_json(response)
# print(output) 
# type(output)   

def extract_words_to_highlight(data_dict):
    """
    Extract words to highlight from the dictionary structure
    
    Parameters:
    data_dict (dict): Dictionary containing the data
    
    Returns:
    list: List of words to highlight
    """
    words_to_highlight = []
    
    # Add student name and father name
    words_to_highlight.append(data_dict["student_name"]["value"])
    words_to_highlight.append(data_dict["father_name"]["value"])
    
    # Add subjects and marks
    for subject_data in data_dict["subject_marks"]:
        words_to_highlight.append(subject_data["subject"])
        words_to_highlight.append(str(subject_data["marks"]))
    
    return words_to_highlight

def words_highlighted(output): 
    words_to_highlight =extract_words_to_highlight(output)
    words_list = []
    for item in words_to_highlight:
        words_list.extend(item.split())
    return words_list


