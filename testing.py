from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from ollama import ChatResponse 
from ollama import chat 

image = Image.open("images/1695832675.webp")
langs = [None] # Replace with your languages or pass None (recommended to use None)
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

predictions = recognition_predictor([image], [langs], detection_predictor)


ocr_output=predictions
prompt =f'''
given {ocr_output} extract student name, father's name, school name from it along with the bounding boxes in JSON format
'''
response: ChatResponse = chat(model='deepseek-r1:7b', messages=[
  {
    'role': 'user',
    'content': prompt,
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
