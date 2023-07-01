'''
@TranRick 2023/06/30
This code use to Generate the Visual Instruction dataset
Section 1 : Setting the Parameters and Configure BLIP Model 
Section 2 : Construct Instruction Input from GPT and Instruction Responese from BLIP --> Suggestions & Solution from GPT 

'''
import argparse
import json
import os
import sys

import torch
import yaml
from blips import Viusal_Understanding
from llm_gpt import (Generate_instruction_Input_output, call_chatgpt,
                     call_gpt3, get_instructions, prepare_chatgpt_message,
                     prepare_gpt_prompt, set_openai_key)
from torchvision import transforms
from utils import CityscapesSegmentation, print_info

##***************************************************************************************************
## ------------  Section 1 Setting the Parameters and Configure BLIP Model -----------------
##***************************************************************************************************
  
device = "cuda" if torch.cuda.is_available() else "cpu"
set_openai_key() ## Remember to "export OPENAI_API_KEY=yourkey" in the terminal

## ------------ Setting the BLIP -----------------
def get_blip_model(base_model="blip2", blip_model="OPT2.7B COCO" , cache_dir="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2/", load_bit=4):
    
    BLIP2_models_available = ['FlanT5 XXL','FlanT5 XL COCO','OPT6.7B COCO','OPT2.7B COCO', 'FlanT5 XL','OPT6.7B', 'OPT2.7B',]
    InstructBLIP_available= ['Ins_FlanT5 XXL','Ins_FlanT5 XL','vicuna-13B', 'vicuna-7B',]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if base_model == "blip2":
        if blip_model in BLIP2_models_available:
            blip_model = Viusal_Understanding(device, base_model=base_model,blip_model=blip_model, cache_dir=cache_dir, load_bit=load_bit)
    elif base_model == "instructblip":
        if blip_model in InstructBLIP_available:
            blip_model = Viusal_Understanding(device, base_model=base_model,blip_model=blip_model, cache_dir=cache_dir, load_bit=load_bit)
    else:
        raise ValueError("Currently only support blip2 and instructblip model")    
        
    return blip_model

## Some Default Parameters FOR BLIP llm Model
BLIP_llm_decoding_strategy="nucleus"
BLIP_max_length_token_output=100 


##***************************************************************************************************
##  Section 2  onstruct Instruction Input from GPT and Instruction Responese from BLIP --> Suggestions & Solution from GPT -----------------
##***************************************************************************************************
  



## ------------ Instruction input and Responese -----------------


# Valid of Select GPT Model
VALID_CHATGPT_MODELS = ['gpt-4', "gpt-35-turbo"]
VALID_GPT3_MODELS = ['text-davinci-003', 'text-davinci-002', 'davinci']


## 1.1 Setting Prompt to GPTs Model to Create Instruction input for Blip2 or InstructBLIP

input_INSTRUCTION = \
"I have an image. " \
"Ask me questions about the content of this image related to Driving Domain, the information providing is related to street views information. " \
"Carefully asking me informative questions to maximize your information about this image content. " \
"Each time ask one question only without giving an answer. " \
"Avoid asking many yes/no questions." \
"I'll put my answer beginning with \"Answer:\"." \


sub_INSTRUCTION = \
"Next Question. Avoid asking yes/no questions. \n" \
"Question: "


## 1.2 Setting Prompt for GPTs Model Provide the Advice and Suggestion 

solution_INSTRUCTION = \
'Now summarize the information you get from abstract visual information. ' \
'Based on the summarize information, you are a helpful assistant please provide some suggestions, advices and other assistance to Driver. ' \
'Don\'t add information. Don\'t miss information. \n' \
'Summary: '


ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t know honestly. Don\'t imagine any contents that are not in the image.'
SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following blip2 huggingface demo
FIRST_instruction = 'Describe this image in detail.'

instruction_dict=get_instructions(input_INSTRUCTION, sub_INSTRUCTION, solution_INSTRUCTION, ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION, FIRST_instruction)

## 1.3 Setting Prompt for GPTs Model Provide the Advice and Suggestion
def summarize_and_suggestion(questions, answers, model, max_gpt_token=100):
    
    if model in VALID_GPT3_MODELS:
        summary_prompt = prepare_gpt_prompt(
                    input_INSTRUCTION, 
                    questions, answers, 
                    solution_INSTRUCTION)
        summary, n_tokens = call_gpt3(summary_prompt, model=model, max_tokens=max_gpt_token)

    elif model in VALID_CHATGPT_MODELS:
        summary_prompt = prepare_chatgpt_message(
                    input_INSTRUCTION, 
                    questions, answers, 
                    solution_INSTRUCTION
                )
        summary, n_tokens = call_chatgpt(summary_prompt, model=model, max_tokens=max_gpt_token)
  
    else:
        raise ValueError('{} is not a valid question model'.format(model))
        
    summary = summary.replace('\n', ' ').strip()
    return summary, summary_prompt, n_tokens


def visual_instruction_input_response(blip, image, GPT_model, n_rounds=10, max_gpt_token=100, n_blip2_context=-1, print_mode='chat', BLIP_llm_decoding_strategy="nucleus", BLIP_max_length_token_output=100):
    

    
    results = {}
    instruction_input_output = Generate_instruction_Input_output(
                    image, blip, GPT_model, 
                    FIRST_instruction, input_INSTRUCTION,
                    sub_INSTRUCTION, 
                    VALID_CHATGPT_MODELS, VALID_GPT3_MODELS,
                    ANSWER_INSTRUCTION, SUB_ANSWER_INSTRUCTION,
                    max_gpt_token, n_blip2_context,
                    )

    questions, answers, n_token_chat = instruction_input_output.chatting(n_rounds, print_mode=print_mode, 
                                                                         BLIP_llm_decoding_strategy=BLIP_llm_decoding_strategy, 
                                                                         BLIP_max_length_token_output=BLIP_max_length_token_output)

    summary, summary_prompt, n_token_sum = summarize_and_suggestion(questions, answers, model=GPT_model)

    ## Append all Question and Answer (chat history) and Final Suggestion and Summary
    
    results['Visual_instruction'] = {'instruction_input_response_suggestion': summary, 'each_input_response': summary_prompt, 'n_token': n_token_chat + n_token_sum}
    ## Get all the Answers from BLIP2
    #results['BLIP2_response'] = {'blip_response': answers}#[0]
    ## Default BLIP2 caption
    caption = blip.caption(image)
    results['BLIP_short_captioning'] = {'image_caption': caption}
    
    return results


def generate(blip, image, GPT_model, n_rounds=10, n_blip2_context=-1, print_mode='chat', BLIP_max_length_token_output=100, BLIP_llm_decoding_strategy='nucleus'):
    """
    Caption images with a set of blip2 models

    Args:
        blip (dict): A dict of blip2 models. Key is the blip2 model name
        image: Single image input PIL image 
        img_ids (list): a list of image ids in the dataset used to caption
        gpt_model (str): the model name used to ask quetion. Valid values are 'gpt3', 'chatgpt', and their concrete model names 
                    including 'text-davinci-003', 'davinci,' and 'gpt-3.5-turbo'.
        n_rounds (int): the number generate input instruction and response
        n_blip2_context (int): how many previous QA rounds can blip2 see. negative value means blip2 can see all 
        print_mode (str): print mode. 'chat' for printing everying. 'bar' for printing everthing but the chat process. 'no' for no printing
    """
    if GPT_model == 'gpt3':
        GPT_model = 'text-davinci-003'
    elif GPT_model == 'chatgpt':
        GPT_model = 'gpt-35-turbo'
    
    # for blip2_tag, blip2 in blip2s.items():
    visual_instruction_data = visual_instruction_input_response(blip=blip, 
                                        image=image, 
                                        GPT_model=GPT_model,
                                        n_rounds=n_rounds, 
                                        n_blip2_context=n_blip2_context, 
                                        print_mode=print_mode, 
                                        BLIP_llm_decoding_strategy=BLIP_llm_decoding_strategy, 
                                        BLIP_max_length_token_output=BLIP_max_length_token_output, 
                                        )

    return visual_instruction_data


def main(args): 
    if args.datasets == 'cityscape':
        ## ------------ Loading the Dataset -----------------
        # Which subset of Cityscapes dataset to use 
        split = 'train' # ["train_extra", "train", "val", "test"]
        version= "gtCoarse"   # ['gtFine', 'gtCoarse'] gtFine is 5k image, gtCoarse is 20k image
        img_size= 480 # Resize Corresponding Original Image Shape if transform is None

        ## If we want to Resize to Square 
        transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #mageNet normalization
            ## convert back to PIL image
            transforms.ToPILImage()
            ])
        ## Dataset is iteratble object
        dataset = CityscapesSegmentation(root_dir=args.data_dir, split=split, img_size=img_size,  transforms=transform, version=version)
        

        ## ------------ Loading the BLIP2 Model -----------------
        ## Load the BLIP2 Model
        blip = get_blip_model(base_model=args.blip_type_model, blip_model=args.blip_LLM , cache_dir=args.cache_dir, load_bit=args.blip_load_bit)


        ## ------------ Generate Visual Instruction Dataset -----------------
        # images_names=[]
        instruction_input_output=[]
        for i, (image_and_lable, img_names) in enumerate(dataset):
            # do something with the image and label, for example print their shapes
            image= image_and_lable[0]
            label= image_and_lable[1]
            print(f"Image shape: {image.size}, Segment Label shape: {label.size}")
            # label.save(f'./cityscape_test_imgs/test_segment_{i}.png')
            image.save(f'./cityscape_test_imgs/test_image_{i}.png')
            name= img_names[i]
            name_start = str(name).find("/cityscape_synthetic/")
            image_path = name[0][name_start-1:]
            #images_names.append(image_path) 

            ## Generate Visual Instruction Dataset
            visual_instruction_data= generate(blip, image, args.gpt_model, 
                                               n_rounds=10,
                                               n_blip2_context=args.n_blip2_context,
                                               print_mode=args.chat_mode, 
                                               BLIP_max_length_token_output=args.bli2_max_lenght_token_gen, 
                                               BLIP_llm_decoding_strategy=args.blip2_llm_decoding_strategy
                                               )
            
            image_input_output={'image_name': image_path, 'visual_instruction_data': visual_instruction_data}
            instruction_input_output.append(image_input_output)
            
            
            if i==1: 
                break

        
        save_name= os.path.join(args.save_path, 'visual_instruction_data.json')
        with open(save_name, 'w') as f:
            json.dump(instruction_input_output, f)

    else: 
         raise NotImplementedError('Dataset {} is currently not supported.'.format(args.datasets))
      

if __name__ == '__main__':
    
    def parse():
        parser = argparse.ArgumentParser(description='Generating captions in test datasets.')
        parser.add_argument('--data_dir', type=str, default="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/", 
                            help='root path to the datasets')
        parser.add_argument('--datasets', nargs='+', choices=['cityscape', 'kitty', 'waymo', 'others'], default='cityscape',
                        help='Names of the datasets to use in the experiment.  Default is Cityscape.')
        
        parser.add_argument('--save_path', type=str, default='./cityscape_test_imgs/', 
                            help='root path for saving results')
 
        parser.add_argument('--blip_type_model', type=str, default='blip2', choices=['blip2', 'instructblip'], help='choosing type of BLIP  Model')
        parser.add_argument('--blip_LLM', type=str, default='OPT2.7B COCO',choices= ['FlanT5 XXL','FlanT5 XL COCO','OPT6.7B COCO','OPT2.7B COCO', 'FlanT5 XL','OPT6.7B', 'OPT2.7B','Ins_FlanT5 XXL','Ins_FlanT5 XL','vicuna-13B', 'vicuna-7B',], 
                            help='choosing existing LLM from BLIP Available Model')
        parser.add_argument('--blip_load_bit', type=int, default=4, choices=[4, 8, 16],help='choosing bit to load BLIP LLM model')
        parser.add_argument('--cache_dir', type=str, default="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLIP2/", 
                            help='Saving and Loading the Pretrained in certain Directory instead save in Huggingface cache default')
        
        parser.add_argument('--n_blip2_context', type=int, default=-1, 
                            help='Number of QA rounds visible to BLIP-2. Default is 1, which means BLIP-2 only remember one previous question. -1 means BLIP-2 can see all the QA rounds')
        parser.add_argument('--blip2_llm_decoding_strategy', type=str, default='nucleus',choices=['beam_search', 'nucleus', 'contrastive_search'], 
                            help='Decoding strategy for BLIP-2 LLM. Default is nucleus sampling.')
        parser.add_argument('--bli2_max_lenght_token_gen', type=int, default=100, 
                            help='Max length of tokens generated by BLIP-2 LLM. Default is 100.')
      
        parser.add_argument('--n_rounds', type=int, default=10, 
                            help='Number of QA rounds between GPT and BLIP. Default is 10, which costs about 3k tokens in GPT API.')
        parser.add_argument('--chat_mode', type=str, default="chat", choices=['chat', 'no'],
                            help='chat mode or no chat mode if chat mode, it will print out the chat history')
        parser.add_argument('--gpt_model', type=str, default='chatgpt', choices=['gpt-4','gpt-35-turbo',  'gpt3', 'text-davinci-003' ],
                            help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
        
        args = parser.parse_args()
        return args

    args = parse()
    main(args)


   
