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


class GCELoss(nn.Module):

    def __init__(self, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.ignore_index = ignore_index
             
    def forward(self, logits, targets, weights, q=0.0):
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        weights = weights[valid_idx]
        # vanilla cross entropy when q = 0
        if q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            if logits.size(-1) == 1:
                pred = torch.sigmoid(logits)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(logits, dim=-1)
            pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
            loss = (1-pred**q) / q
        loss = (loss.view(-1)*weights).sum() / weights.sum()
        return loss
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0.1):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, embed_dim)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

    def forward(self, k, q, memory_len):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, embed_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = torch.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.tanh(torch.bmm(qw, kt))
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        attentions = torch.squeeze(score, dim=1)
        # print(attentions[:2])
        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).to(self.device)
        for i, l in enumerate(memory_len):
            if l < k_len:
                mask[i, l:] = 0
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        # print(masked[:2])
        # print(masked.shape)
        _sums = masked.sum(-1)  # sums per row
        attentions = torch.div(masked, _sums.view(_sums.size(0), 1))
        # print(attentions[:2])

        score = torch.unsqueeze(attentions, dim=1)

        output = torch.bmm(score, kx)  # (n_head*?, k_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, k_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, k_len, embed_dim)
        return output

class BertPooler(nn.Module):
    def __init__(self, config, pos=0):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pos = pos

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, self.pos]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(Attention):
    '''q is a parameter'''

    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product', q_len=1, dropout=0.1):
        super(SelfAttention, self).__init__(embed_dim, hidden_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.FloatTensor(q_len, embed_dim))

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(SelfAttention, self).forward(k, q)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class MultimodalEncoder(nn.Module):
    def __init__(self, config, deep=1):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(deep)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
    
    
class MultimodalEncoder(nn.Module):
    def __init__(self, config, deep=1):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(deep)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class GANModel(nn.Module):

    def __init__(self,
                 args,
                 text_config,
                 vision_config,
                 text_num_labels,
                 alpha,
                 beta,
                 text_model_name="roberta",
                 image_model_name='vit',
                 add_pooling_layer=True):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_config = text_config  # text config
        self.vision_config = vision_config  # vision config
        self.text_num_labels = text_num_labels

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
        self.classifier1 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.classifier0 = nn.Linear(text_config.hidden_size, self.text_num_labels)
        self.CRF = CRF(self.text_num_labels, batch_first=True)

        # text model
        self.text_config = text_config
        self.encoder = UnimoEncoder(vision_config=self.vision_config, text_config=self.text_config)
        self.args = args
        
        self.text_model = AutoModel.from_pretrained(args.name_path_dict[text_model_name])
        self.image_model = AutoModel.from_pretrained(args.name_path_dict[image_model_name])

        self.vismap2text = nn.Linear(vision_config.hidden_size, text_config.hidden_size)
        self.comb_attention = MultimodalEncoder(text_config, deep=1)
        self.text_pooler = BertPooler(text_config)
        self.img_pooler = BertPooler(text_config)

        self.gen_linear = nn.Linear(text_config.hidden_size, text_config.hidden_size)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                text_feature=None,
                text_logits_feature=None,
                text_hidden_feature=None,
                image_feature=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=True,
                output_hidden_states=True,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.text_config.use_return_dict

        text_outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        # image_outputs = self.image_model(pixel_values)
        # print(image_feature.size())
        
        text_feature = text_outputs["last_hidden_state"]  #16, 60, 768
        
        

        if self.args.add_gan:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            encoder_outputs = self.encoder(
                vision_embeds=image_feature,
                text_embeds=text_feature,
                attention_mask=extended_attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            text_feature = encoder_outputs.last_text_state
            image_feature = encoder_outputs.last_vision_state



        sequence_output1 = self.dropout(text_feature)
        text_token_logits = self.classifier1(sequence_output1)
        
        # * text only # text_loss
        # loss_fct = GCELoss()            
        # weights = torch.ones(labels.shape).to(labels.device)
        # text_loss = loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1), weights)
        
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))
        if self.args.only_text_loss :
            #  * vision-aware text # cross_crf_loss
            loss = text_loss 
        else:
            image_text_cross_attention, _ = self.image_text_cross(text_feature, image_feature, image_feature)
            cross_logits = self.classifier0(image_text_cross_attention)
            mask = (labels != -100)
            mask[:, 0] = 1
            cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10

            # * token-patch matching # word patch align loss
            batch_size, image_len, _ = image_feature.shape
            text_pad = (attention_mask == 1).clone().detach()
            image_pad = torch.zeros(batch_size, image_len, dtype=torch.bool, device=attention_mask.device)
            ot_dist = optimal_transport_dist(text_feature, image_feature, text_pad, image_pad)
            word_region_align_loss = ot_dist.mean()

            loss = self.alpha * text_loss + cross_crf_loss  + self.beta * word_region_align_loss   #27


        if self.args.add_gan_loss:
            loss += cal_loss(output=encoder_outputs)

        return {"loss": loss, "logits": text_token_logits, "cross_logits": None, }
        # text_token_logits         4, 60, 5
