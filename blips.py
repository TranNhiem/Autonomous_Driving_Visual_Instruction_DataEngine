
'''
@TranRick 2023/06/30
This code features the following:
1. Using BLIP2 to Generate Caption or Respone for an Image 
2. Using InstructBLIP to generate a Abstract Visual Information  

'''
import os
from PIL import Image
import requests
from transformers import (Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, 
                          BlipForConditionalGeneration, InstructBlipProcessor, 
                          InstructBlipForConditionalGeneration, BitsAndBytesConfig)
import torch
from utils import resize_long_edge
from torchvision import transforms
from pathlib import Path
## Initialize the Pretrained Models From Huggingface Transformers
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
    'Ins_FlanT5 XXL': 'Salesforce/instructblip-flan-t5-xxl',
    'Ins_FlanT5 XL': 'Salesforce/instructblip-flan-t5-xl',
    'vicuna-13B': 'Salesforce/instructblip-vicuna-13b',
    'vicuna-7B': 'Salesforce/instructblip-vicuna-7b',
}


class Viusal_Understanding():
    def __init__(self, device, base_model='instructblip', blip_model="OPT2.7B COCO",  load_bit=8, cache_dir='/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2',):
        self.device = device
        self.visual_understand = base_model
        self.blip_model = blip_model
        print(cache_dir)
        self.load_bit = load_bit
        self.cache_dir=Path(cache_dir) #'/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2'
        #self.cache_dir = cache_dir #
        self.processor, self.model = self.initialize_model()
       
    def initialize_model(self,):
        
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16

        quant_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        quant_config_8bit=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            )
          
        if self.load_bit == 4:
            print("Using 4bit loading model")
            quant_config = quant_config_4bit
        elif self.load_bit == 8:
            print("Using 8bit loading model")
            quant_config = quant_config_8bit
        else: 
            print("Using fp16 loading model")
            quant_config = None

        if self.visual_understand == 'blip2':
            print("Using BLIP2")
            processor = Blip2Processor.from_pretrained(BLIP2DICT[self.blip_model],cache_dir=self.cache_dir, trust_remote_code=True,use_auth_token=True )#
          
            model = Blip2ForConditionalGeneration.from_pretrained(
                BLIP2DICT[self.blip_model],  
                cache_dir=self.cache_dir,
                torch_dtype=self.data_type if self.load_bit !=8 or self.load_bit !=4 else None, 
                quantization_config=quant_config, 
                trust_remote_code=True
            )
        elif self.visual_understand == 'instructblip':
            print("Using instructBLIP")
            processor = InstructBlipProcessor.from_pretrained(InstructBLIPDICT[self.blip_model], cache_dir=self.cache_dir, trust_remote_code=True, use_auth_token=True)
            model = InstructBlipForConditionalGeneration.from_pretrained(
                InstructBLIPDICT[self.blip_model], 
                cache_dir=self.cache_dir,
                torch_dtype=self.data_type, 
                quantization_config=quant_config, 
                trust_remote_code=True

            )
        # for gpu with small memory
        elif self.visual_understand == 'blip':
            print("Using BLIP")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=self.data_type, )
        else:
            raise ValueError(f'This {self.visual_understand} based Model is not supported')
        
        if self.load_bit == 16:
            model.to(self.device)

        return processor, model

    def abstract_visual_output(self, raw_image, instruction_input, llm_decoding_strategy="nucleus", max_length=512):
        inputs = self.processor(raw_image, instruction_input, return_tensors="pt").to(self.device, torch.float16)
        
        if llm_decoding_strategy == "beam_search":
            out = self.model.generate(**inputs, num_beams=5,
                                      max_length=max_length,
                                      top_p=0.9,
                                      repetition_penalty=1.5,
                                      length_penalty=1.0,
                                      temperature=1,)
        elif llm_decoding_strategy == "nucleus":
            out = self.model.generate(**inputs, do_sample=True, max_length=max_length, top_p=0.95, top_k=0)
        
        elif llm_decoding_strategy =="contrastive_search": 
            out = self.model.generate(**inputs, max_length=max_length, penalty_alpha=0.6, top_k=6, repetition_penalty=1.5,)

        answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return answer
    

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        caption = self.abstract_visual_output(raw_image, 'a photo of')
        caption = caption.replace('\n', ' ')#.strip()  # trim caption
        return caption
    
def main(device, image_src, base_model, blip_model, cache_dir, load_bit, instruction_input):
    image = Image.open(image_src)
    image = resize_long_edge(image, 224)
    visual_understanding_model=Viusal_Understanding(device=device, base_model=base_model, blip_model=blip_model, load_bit=load_bit,  cache_dir=cache_dir,)
    respones = visual_understanding_model.abstract_visual_output(image, instruction_input)
    print(respones)
   
   
if __name__ == '__main__':

    image_path="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/leftImg8bit/train/bochum/bochum_000000_000600_leftImg8bit.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "blip2" # ['instructblip' , 'blip2' ]
    blip_model="OPT2.7B COCO" # ["vicuna-7B", "OPT2.7B COCO", "OPT6.7B COCO",] 
    cache_dir='/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2/'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    load_bit=4
    instruction_input = "A photo of"#"Describe this image in detail"
    main(device, image_path, base_model=base_model,blip_model=blip_model, load_bit=load_bit, cache_dir=cache_dir, instruction_input=instruction_input )
    