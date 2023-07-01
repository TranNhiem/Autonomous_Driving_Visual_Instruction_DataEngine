
'''
@TranNhiem 2023/06/30 
include the following functions: 
1. Processing Image Resize, Crop, Padding Etc.
2. Dataloader for Cityscape Dataset Return (Image, Segmentation Label) and Image's Name 

'''


import os 
import cv2
import argparse
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import json 


##------------------ Image Processing ------------------## 

def resize_long_edge(image, target_size=384):
    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    # Determine the new dimensions
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image



##------------------CityScape Dataloader ------------------## 

class CityscapesSegmentation(Dataset):
    '''
    Data Structure of CityScape Dataset
        cityscapes
    ├── gtFine # Segmentation Label
    │   ├── test (19 City Folders)
    │   ├── train (19 City Folders)
    │   └── val (19 City Folders)
    └── leftImg8bit
        ├── test (19 City Folders)
        ├── train (19 City Folders)
        └── val (19 City Folders)

    '''

    def __init__(self, root_dir, split,img_size=256, transforms=None, version='gtFine'):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.img_size=img_size 

        if version=='gtFine':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
            self.label_dir = os.path.join(root_dir, 'gtFine', split)
        else:
        
            self.image_dir = os.path.join(root_dir, 'leftImg8bit', split)
            self.label_dir = os.path.join(root_dir, 'gtCoarse', split)

        self.images = []
        for city_dir in os.listdir(self.image_dir):
            city_image_dir = os.path.join(self.image_dir, city_dir)
            for image_name in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, image_name)
                if version=='gtFine':
                    label_name = '{}_gtFine_color.png'.format(image_name.split('_leftImg8bit')[0])
                else:
                    label_name = '{}_gtCoarse_color.png'.format(image_name.split('_leftImg8bit')[0])
                label_path = os.path.join(self.label_dir, city_dir, label_name)
                assert os.path.isfile(image_path), f"{image_path} does not exist"
                assert os.path.isfile(label_path), f"{label_path} does not exist"
                self.images.append((image_path, label_path))

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, idx):
    #     breakpoint()
    #     if isinstance(idx, int)
    #         image_path, label_path = self.images[idx]
        
            
    #     elif isinstance(idx, slice):
    #         # Handle slice object
    #         start = idx.start if idx.start is not None else 0
    #         stop = idx.stop if idx.stop is not None else len(self.images)
    #         step = idx.step if idx.step is not None else 1
    #         image_paths, label_paths = zip(*self.images[start:stop:step])
    #         return [Image.open(p).convert('RGB') for p in image_paths], [Image.open(p) for p in label_paths]
    #     else:
    #         raise TypeError('Invalid argument type')
        
    #     image = Image.open(image_path).convert('RGB')
    #     label = Image.open(label_path)
    #     if self.transforms is not None:
    #         image = self.transforms(image)
    #         ## Resize lable image to match the size of the image
    #         #label = transforms.Resize(image.shape[1:], Image.NEAREST)(label)
    #         img_size=image.size
    #         label = transforms.Resize((img_size[1],img_size[0]), Image.NEAREST)(label)
    #         #breakpoint()
    #     return image, label

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            indices = range(start, stop, step)
            images = [self.images[i] for i in indices]

            return [self._load_image(i) for i in images]
        else:
            idx=idx#+ 15000 ## Controll Increasment value 
            return self._load_image(self.images[idx]), self.images
            #return self.images[idx]

    def _load_image(self, image_paths):
        image_path, label_path = image_paths
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        image= resize_long_edge(image, target_size=self.img_size)
        label= resize_long_edge(label, target_size=self.img_size)

        if self.transforms is not None:
            image = self.transforms(image)
            img_size = image.size
            label = transforms.Resize((img_size[1], img_size[0]), Image.NEAREST)(label)
    
        return image, label
    
    def read_all_imgs_json(self, json_names_path, condition_img=None, num_imgs=10): 
        with open(json_names_path, 'r') as f: 
            for line in f:
                #Load the JSON Data from the file 
                data= json.loads(line)
            # Extract image names and image IDs from the JSON data
            image_names = []
            image_ids = []

            for i, item in enumerate(data):
                
                image_id = item['image_id']
                image_name = item['image_name']
                image_name= os.path.join(self.root_dir,image_name)
                
                
                if condition_img is not None and item['image_name'] == condition_img:
                    #breakpoint()
                    # Found the image name, extract the image ID
                    image_id = item['image_id']
                    image_name_= self.root_dir+condition_img

                    #print(f"Image ID for image name '{image_name_}': {image_id}")

                    return image_id, image_name_

                image_ids.append(image_id)
                image_names.append(image_name)
                # if i== num_imgs: 
                #     break
            return image_ids, image_names
        

def main(args):

    dataset = CityscapesSegmentation(root_dir=data_dir, split='train_extra', transforms=transform, version="gtCoarse")
    return dataset

if __name__ == '__main__':


    data_dir="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/cityscape_synthetic/"
    save_dir="./cityscape_test_imgs/"
    split = 'train'
    img_size= 256

    ## If want to Resize and Transform Image via ImageNet normalization and Resize Square 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ## convert back to PIL image
        transforms.ToPILImage()
        ])
    
    dataset = CityscapesSegmentation(root_dir=data_dir, split='train_extra', img_size=img_size,  transforms=transform, version="gtCoarse")
    
    images_names=[]
    for i, (image_and_lable, img_names) in enumerate(dataset):
        # do something with the image and label, for example print their shapes
        image= image_and_lable[0]
        label= image_and_lable[1]
        print(f"Image shape: {image.size}, Label shape: {label.size}")
        label.save(f'./cityscape_test_imgs/test_segment_{i}.png')
        image.save(f'./cityscape_test_imgs/test_image_{i}.png')
        #print("this is image name: ", img_names[i])
        name= img_names[i]
        name_start = str(name).find("/cityscape_synthetic/")
        image_path = name[0][name_start-1:]
        print(image_path)
        images_names.append(image_path) 
        if i==4: 
            break
    with open('./cityscape_test_imgs/images_names.json', 'w') as f:
        json.dump(images_names, f)
      
    
    


