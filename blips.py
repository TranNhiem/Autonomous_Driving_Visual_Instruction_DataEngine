
'''
@TranRick 2023/07/30
This code features the following:
1. Using BLIP2 to Generate Caption or Respone for an Image 
2. Using InstructBLIP to generate a Abstract Visual Information  

'''

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch
from utils.util import resize_long_edge

BLIP2DICT = {
    'FlanT5 XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5 XL COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'OPT6.7B COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'FlanT5 XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}


InstructBLIPDICT = {
    'FlanT5 XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5 XL COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'OPT6.7B COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'FlanT5 XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}



class ImageCaptioning:
    def __init__(self, device, captioner_base_model='blip'):
        self.device = device
        self.captioner_base_model = captioner_base_model
        self.processor, self.model = self.initialize_model()

    def initialize_model(self,):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        if self.captioner_base_model == 'blip2':
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=self.data_type
            )
        # for gpu with small memory
        elif self.captioner_base_model == 'blip':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=self.data_type)
        else:
            raise ValueError('arch not supported')
        model.to(self.device)
        return processor, model

    def image_caption(self, image_src):
        image = Image.open(image_src)
        image = resize_long_edge(image, 384)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep1, BLIP2 caption:')
        print(generated_text)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return generated_text
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."