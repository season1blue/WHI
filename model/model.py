# 忽略not init权重的warning提示
from transformers import logging
logging.set_verbosity_error()

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel
from transformers import T5EncoderModel, BloomModel, DistilBertModel, DebertaModel, GPT2Model, GPTNeoModel, AutoTokenizer, BloomForTokenClassification
from transformers import BartModel, T5Model, AutoModel

from transformers import CLIPTextModel, CLIPVisionModel

from model.modeling_dtca import MultiHeadAttention
from model.modeling_dtca import ScaledDotProductAttention
from model.modeling_dtca import optimal_transport_dist

from model.gan import UnimoEncoder, get_extended_attention_mask
from model.gan import CLIPVisionEmbeddings, BertEmbeddings, get_head_mask

from model.modeling_output import BaseModelOutputWithPooling
from utils.utils import cal_loss

from clip import load as clip_load
import faulthandler

from transformers import BertLayer
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import json
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码
import math

class SENModel(nn.Module):

    def __init__(self,
                 args,
                 text_config,
                 vision_config,
                 alpha,
                 beta,
                 text_model_name="roberta",
                 image_model_name='vit',
                 add_pooling_layer=True):
        super().__init__()

        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_config = text_config  # text config
        self.vision_config = vision_config  # vision config

        # text_config.hidden_size = 512
        # vision_config.hidden_size = 768
        
        self.image_text_cross = nn.MultiheadAttention(
            num_heads=8,
            embed_dim= vision_config.hidden_size,
            kdim=text_config.hidden_size,
            vdim=text_config.hidden_size,
            batch_first=True
            )
        self.dropout = nn.Dropout(0.1)
        self.loss_fct = CrossEntropyLoss()
        self.text_linear = nn.Linear(text_config.hidden_size, args.text_num_labels)
        self.classifier0 = nn.Linear(text_config.hidden_size, args.text_num_labels)
        self.image_linear = nn.Linear(vision_config.hidden_size, args.text_num_labels)
        self.CRF = CRF(args.text_num_labels, batch_first=True)

        # text model
        self.text_config = text_config
        self.encoder = UnimoEncoder(vision_config=self.vision_config, text_config=self.text_config)
        self.args = args
        
        self.text_model = AutoModel.from_pretrained(args.name_path_dict[text_model_name])
        self.image_model = AutoModel.from_pretrained(args.name_path_dict[image_model_name])

    def forward(self,
                input_ids=None,
                attention_mask=None,
                aspect_ids=None,
                aspect_mask=None,
                text_feature=None,
                image_feature=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=True,
                output_hidden_states=True,
                image_labels=None,
                sentiment=None,
                cross_labels=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.text_config.use_return_dict
        text_outputs = self.text_model(input_ids, attention_mask=attention_mask)
        # image_outputs = self.image_model(pixel_values)
        aspect_outputs = self.text_model(aspect_ids, attention_mask=aspect_mask)
        
        text_feature = text_outputs["last_hidden_state"]  #16, 60, 768
        aspect_feature = aspect_outputs["last_hidden_state"]
        

        if self.args.add_gan:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            encoder_outputs = self.encoder(
                vision_embeds=image_feature,
                text_embeds=aspect_feature,
                attention_mask=extended_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            aspect_feature = encoder_outputs.last_text_state
            image_feature = encoder_outputs.last_vision_state

        fusion = text_feature[:, 0] + aspect_feature[:, 0]
        
        sequence_output1 = self.dropout(fusion)
        fusion_logits = self.text_linear(sequence_output1)
        
        text_loss = self.loss_fct(fusion_logits.view(-1, self.args.text_num_labels), sentiment.view(-1))
        if self.args.only_text_loss :
            loss = text_loss 
        else:
            # image_text_cross_attention, _ = self.image_text_cross(aspect_feature, image_feature, image_feature)  # image 16, 50, 768 text 16, 60, 768
            # image_fusion = text_feature + aspect_feature + image_text_cross_attention  
            # image_logits = self.image_linear(image_fusion[:, 0])
            # image_loss = self.loss_fct(image_logits.view(-1, self.args.text_num_labels), sentiment.view(-1))


            # * token-patch matching # word patch align loss
            batch_size, image_len, _ = image_feature.shape
            text_pad = (attention_mask == 1).clone().detach()
            image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
            ot_dist = optimal_transport_dist(text_feature, image_feature, text_pad, image_pad)
            word_region_align_loss = ot_dist.mean()

            loss = text_loss +  self.beta * word_region_align_loss   #27


        if self.args.add_gan_loss:
            loss += cal_loss(output=encoder_outputs)

        return {"loss": loss, "logits": fusion_logits, "cross_logits": None, }
        # text_token_logits         4, 60, 5
