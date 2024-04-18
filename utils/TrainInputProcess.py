from transformers import AutoTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
import os
import collections
from PIL import Image,ImageFile
import argparse
from tqdm import tqdm
import torch.nn as nn
import clip

from clip import load as clip_load
from clip import tokenize as clip_tokenize

from transformers import AutoTokenizer,GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BertModel, RobertaModel, BlipForConditionalGeneration,BlipProcessor, CLIPProcessor, AutoProcessor
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel
from transformers import (WEIGHTS_NAME, AutoConfig)

from transformers import CLIPTextModel, CLIPVisionModel

# from utils import set_random_seed, model_select, parse_arg, 

from utils.MyDataSet import MyDataSet2, llmDataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from torchvision import transforms

transform = transforms.Compose([
    # args.crop_size, by default it is set to be 224
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

                            
class TrainInputProcess:
    def __init__(self,
                args,
                text_model_name="roberta",
                image_model_name="vit",
                dataset_type=None,
                data_text_dir=None,
                data_image_dir=None,
                ):
        self.args = args
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.dataset_type = dataset_type
        self.data_text_dir = data_text_dir
        self.data_image_dir = data_image_dir

        self.dataset_types = ['train','dev','test']
        self.text_type = '.txt'
        self.data_dict = dict()
        self.input = dict()
        
        # if self.text_model_name == 'bert':
        #     self.tokenizer = AutoTokenizer.from_pretrained("../../_weight/deberta")
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # elif self.text_model_name == 'roberta':
        #     self.tokenizer = AutoTokenizer.from_pretrained("../_weight/roberta-base", add_prefix_space=True)
        # elif self.text_model_name == 'flant5':
        #     self.tokenizer = AutoTokenizer.from_pretrained("./data/models/flant5", add_prefix_space=True)
        # elif self.text_model_name == "deberta":
        #     self.tokenizer = AutoTokenizer.from_pretrained("../_weight/deberta", add_prefix_space=True)
        # elif self.text_model_name == "gpt2":
        #     self.tokenizer = AutoTokenizer.from_pretrained("./data/models/gpt2", add_prefix_space=True)
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # elif self.text_model_name == "bart":
        #     self.tokenizer = AutoTokenizer.from_pretrained("./data/models/bart", add_prefix_space=True)
        # elif self.text_model_name == "clip":
        #     self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        

        # text_config = AutoConfig.from_pretrained("../_weight/deberta")
        # image_config = AutoConfig.from_pretrained("../_weight/vit-base-patch16-224-in21k")
        # self.roberta = RobertaModel(text_config, add_pooling_layer=False)
        # self.roberta.cuda()
        # self.vit = ViTModel(image_config)
        # self.vit.cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.name_path_dict[self.text_model_name],add_prefix_space=True)  # text tokenizer
        self.processor = AutoProcessor.from_pretrained(self.args.name_path_dict[self.image_model_name])   #image processor
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        # self.token_model, self.preprocess = clip_load(("ViT-B/32"), device=self.device, jit=False)
        
        self.text_model = AutoModel.from_pretrained(args.name_path_dict[text_model_name]).to(self.device)
        self.image_model = AutoModel.from_pretrained(args.name_path_dict[image_model_name]).to(self.device)
        
        if image_model_name == "clip" or "laion":
            self.image_model = CLIPVisionModel.from_pretrained(args.name_path_dict[image_model_name]).to(self.device)
        # self.ct_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.cv_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.ct_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        # self.cv_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)



    # process fine-tune text
    # process_label: False-- 5 class; True-- 7 class.
    def get_text_dataset(self, process_label=False):
        for dataset_type in self.dataset_types:
            data_file_name = dataset_type + self.text_type
            text_path = os.path.join(self.data_text_dir, data_file_name)
            sentence_l, image_l, sentiment_l, aspect_l = [], [], [], []
            text_data = []
            with open(text_path,'r',encoding="utf-8") as f:
                while True:
                    text = f.readline().rstrip('\n').split()
                    if text == []:
                        break
                    aspect = f.readline().rstrip('\n').split()
                    sentiment = f.readline().rstrip('\n')
                    image_path = f.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect) - 1
                    text = text[:start_pos] + aspect + text[start_pos+1:]
                    text_data.append((" ".join(aspect), " ".join(text), sentiment, image_path))
                     
            for item in text_data:
                aspect_l.append([item[0]])
                sentence_l.append([item[1]])
                sentiment_l.append(int(item[2])+1)  # -1,0,1 -> 0, 1, 2 
                image_l.append(item[3])
            self.data_dict[dataset_type] = (aspect_l, sentence_l, sentiment_l, image_l)
    
    
    
    
    
    def generate_dualc_input(self):
        for dataset_type in self.dataset_types:
            aspect_l, sentence_l, sentiment_l, image_l = self.data_dict[dataset_type]
            if self.text_model_name == "clip":
                new_sentence_l = [" ".join(sentence) for sentence in sentence_l]
                tokenized_inputs = self.tokenizer(new_sentence_l, padding=True, return_tensors="pt").to(self.device)
            else:
                tokenized_inputs = self.tokenizer(sentence_l, truncation=True, is_split_into_words=True, padding='max_length', max_length=60, return_tensors='pt').to(self.device)
                tokenized_aspect = self.tokenizer(aspect_l, truncation=True, is_split_into_words=True, padding='max_length', max_length=60, return_tensors='pt').to(self.device)
                tokenized_inputs["aspect_ids"] = tokenized_aspect["input_ids"]
                tokenized_inputs["aspect_masks"] = tokenized_aspect["attention_mask"]

            # tokenized_inputs["aspect"] = aspect_l
            tokenized_inputs["sentiment"] = torch.tensor(sentiment_l)
            
            pixel_values, image_features = [], []
            with torch.no_grad():
                # 基本图像和sentence进行tokenize处理
                for image_path in tqdm(image_l, desc="image"):
                    image_file_path = os.path.join(self.data_image_dir, image_path)
                    image = Image.open(image_file_path).convert('RGB')

                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    pixel_values.append(inputs["pixel_values"])
                    outputs = self.image_model(**inputs)
                    image_feature = torch.squeeze(outputs.last_hidden_state, 1)
                    image_features.append(image_feature) 
            
            tokenized_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)
            tokenized_inputs["image_feature"] = torch.cat(image_features, dim=0)
            
            
            
            self.input[dataset_type] = tokenized_inputs



    def encode_input(self):
        with torch.no_grad():
            for dataset_type in self.dataset_types:
                print(f"model encode on {dataset_type} datset")
                curr_dataset = MyDataSet2(inputs = self.input[dataset_type])
                curr_dataloader = DataLoader(curr_dataset, batch_size = 1)

                image_feature, text_feature = [], []
                for batch in tqdm(curr_dataloader, desc="model encode"):
                    input_ids = batch["input_ids"].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    pixel_values = batch["pixel_values"].cuda()

                    text_outputs = self.roberta(input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True, return_dict=True)
                    image_outputs = self.vit(pixel_values)

                    text_feature.append(text_outputs["last_hidden_state"].cpu())
                    image_feature.append(image_outputs["last_hidden_state"].cpu())

                text_feature = torch.cat(text_feature, dim=0)
                image_feature = torch.cat(image_feature, dim=0)


                self.input[dataset_type]["text_feature"] = text_feature
                self.input[dataset_type]["image_feature"] = image_feature





def prepare_data(args, file_path, text_model_name="clip", image_model_name="clip", dataset_type=2015):
    
    tip = TrainInputProcess(
                            args,
                            args.text_model_name,
                            args.image_model_name,
                            args.dataset_type,
                            data_text_dir=args.data_text_dir,
                            data_image_dir=args.data_image_dir,
                            )
    
    # finetune
    tip.get_text_dataset()
    tip.generate_dualc_input()
    
    torch.save(tip.input, file_path)
        
    
        


if __name__ == '__main__':
    prepare_data()