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
from transformers.models.bart.modeling_bart import BartClassificationHead

from aspect.module import MultiHeadAttention, optimal_transport_dist



import faulthandler
faulthandler.enable()

class ASPModel(nn.Module):

    def __init__(self,
                 args,
                 text_config,
                 vision_config,
                 text_num_labels,
                 alpha,
                 beta,
                 text_model_name="deberta",
                 image_model_name='vit',
                 ):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(args.name_path_dict[text_model_name])
        self.image_model = AutoModel.from_pretrained(args.name_path_dict[image_model_name])

        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_config = text_config  # text config
        self.vision_config = vision_config  # vision config
        self.text_num_labels = text_num_labels

        text_config.hidden_size = 768

        self.image_text_cross = MultiHeadAttention(
            8, text_config.hidden_size, text_config.hidden_size, text_config.hidden_size)
        self.dropout = nn.Dropout(text_config.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.classifier0 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.CRF = CRF(self.text_num_labels, batch_first=True)



    def forward(self, input_ids, pixel_values,
                attention_mask=None, labels=None, cross_labels=None, pairs=None
            ):


        # image_outputs = self.vit(pixel_values)

        # if self.text_model_name == "roberta":
        #     text_outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # elif self.text_model_name == "deberta":
        #     text_outputs = self.deberta(input_ids, attention_mask=attention_mask)
        # elif self.text_model_name == "bert":
        #     text_outputs = self.bert(input_ids, attention_mask=attention_mask)
        # elif self.text_model_name == "bart":
        #     text_outputs = self.bart(input_ids, attention_mask=attention_mask)
            
        text_outputs = self.text_model(
            input_ids, attention_mask=attention_mask)
        image_outputs = self.image_model(pixel_values)

        text_last_hidden_states = text_outputs["last_hidden_state"]
        image_last_hidden_states = image_outputs["last_hidden_state"]  # 32, 197, 768

        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))

        #  * vision-aware text # cross_crf_loss
        # image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states, image_last_hidden_states)
        # cross_logits = self.classifier0(image_text_cross_attention)
        # mask = (labels != -100)
        # mask[:, 0] = 1
        # cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10

        # # * token-patch matching # word patch align loss
        # batch_size, image_len, _ = image_last_hidden_states.shape
        # text_pad = (attention_mask == 1).clone().detach()
        # image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
        # ot_dist = optimal_transport_dist(text_last_hidden_states, image_last_hidden_states, text_pad, image_pad)
        # word_region_align_loss = ot_dist.mean()

        # loss = self.alpha * text_loss + cross_crf_loss  + self.beta * word_region_align_loss   #27
        loss = text_loss


        return {"loss": loss, "logits": text_token_logits, "cross_logits": text_token_logits, }
        # text_token_logits         4, 60, 5



