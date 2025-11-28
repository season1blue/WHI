from typing import Any, Optional, Tuple
import math

import torch
from torch import nn, Tensor, device

from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
# from transformers.configuration_utils import PretrainedConfig
from model.modeling_output import BaseModelOutput, BaseModelOutputWithPooling
# from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn.functional as F

LAYER_NUM = 12
# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
    """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
    head_mask = [None] * num_hidden_layers

    return head_mask



class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(3, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(3).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds

        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*16, 768 3个子图
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds) # bsz, 48, 768
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)

        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*4, 768 3个子图
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds) # bsz, 3, 768
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = LAYER_NUM
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()  #  torch.Size([16, 50, 768])
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        qks = (key_states, value_states) if output_qks else None


        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, qks


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = LAYER_NUM  # 3
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        mixed_query_layer = self.query(hidden_states)


        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        if past_key_values is not None:
            key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_values[0], value_layer], dim=2)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and past_key_values is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            bsz, nheads, length, dsize = past_key_values[0].size()
            visual_attention_mask = torch.ones((bsz, 1, 1, length)).to(attention_mask.device)  # bsz, 3, len, 64
            attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)
            attention_scores = attention_scores + attention_mask
        elif attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # bsz, 128, 768

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        self_outputs, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            current_layer,
            past_key_values,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.att = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True) 

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        choice = False
        if choice:
            residual = hidden_states

            hidden_states = self.layer_norm1(hidden_states)

            bsz, tgt_len, embed_dim = hidden_states.size()
            proj_shape = (bsz, -1, self.embed_dim)
            past_key_values[0] = past_key_values[0].view(*proj_shape)
            past_key_values[1] = past_key_values[1].view(*proj_shape)
            hidden_states, _ = self.att(hidden_states, past_key_values[0], past_key_values[1])  # 8, 197, 768

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = hidden_states

            return outputs
        else:
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states, attn_weights, qks = self.self_attn(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                output_qks=output_qks,
                current_layer=current_layer,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attn_weights,)

            if output_qks:
                outputs += (qks, )

            return outputs

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.att = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True) 

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # proj_shape = (hidden_states.size(0), -1, self.embed_dim)
        # past_key_values[0] = past_key_values[0].view(*proj_shape)
        # past_key_values[1] = past_key_values[1].view(*proj_shape)

        # hidden_states, _ = self.att(hidden_states, past_key_values[0], past_key_values[1])  # 8, 60, 768

        self_attention_outputs, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            current_layer=current_layer,
            past_key_values=past_key_values,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        if output_qks:
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.num_hidden_layers = LAYER_NUM
        self.vision_num_hidden_layers = LAYER_NUM
        self.text_num_hidden_layers = LAYER_NUM

        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])


        ### Multi-level Cross-modal Generation (MCG)

        self.text_generator = nn.ModuleList([Generator(vision_config, 60)])
        self.vision_generator = nn.ModuleList([Generator(text_config, 50)])

        ###Stage-refined Context Sampler (SCS)
        self.patch_selector = nn.ModuleList([ContextSampler(vision_config.hidden_size) for _ in range(vision_config.num_hidden_layers)])
        self.token_selector = nn.ModuleList([ContextSampler(text_config.hidden_size) for _ in range(text_config.num_hidden_layers)])


    def forward(
        self,
        vision_embeds=None,
        text_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.vision_num_hidden_layers == self.text_num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        all_generated_vision_hidden_states = ()
        all_generated_text_hidden_states = ()

        all_cycle_vision_hidden_states = ()
        all_cycle_text_hidden_states = ()

        all_patch_policy = ()
        all_token_policy = ()

        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds

        ## initial mask selector
        bz, token_len, hidden_size = text_hidden_states.size()
        bz, patch_len, hidden_size = vision_hidden_states.size()
        prev_token_decision = torch.ones(bz, token_len-1, 1, dtype=attention_mask.dtype, device=attention_mask.device)# exclude CLS token
        prev_patch_decision = torch.ones(bz, patch_len-1, 1, dtype=attention_mask.dtype, device=attention_mask.device)


        for idx in range(self.num_hidden_layers):
            # vision
            # TODO: 9-3 layers past text as pkv to vision
            output_qks = True
            if idx == 0:
                bsz, token_length, dsize = text_embeds.size()
                visual_past_key_values = None
            else:
                generated_text_hidden_states = generated_text_hidden_states.contiguous()
                visual_past_key_values = [generated_text_hidden_states.view(bsz, LAYER_NUM, token_length, dsize//LAYER_NUM), generated_text_hidden_states.view(bsz, LAYER_NUM, token_length, dsize//LAYER_NUM)]

            if idx != 0 :
                vision_layer_module = self.vision_layers[idx]
                vision_layer_output = vision_layer_module(
                        vision_hidden_states,
                        output_attentions=output_attentions,
                        past_key_values=visual_past_key_values,
                        current_layer=idx,
                        output_qks=output_qks,
                )
                vision_hidden_states = vision_layer_output[0]

            # text
            # TODO: 9-3 layers past vison qks to text
            if idx == 0:
                bsz, patch_length, dsize = vision_embeds.size()
                text_past_key_values = None
            else:
                generated_vision_hidden_states = generated_vision_hidden_states.contiguous()
                text_past_key_values = [generated_vision_hidden_states.view(bsz, LAYER_NUM, patch_length, dsize//LAYER_NUM) ,generated_vision_hidden_states.view(bsz, LAYER_NUM, patch_length, dsize//LAYER_NUM)]

            if idx != 0 :
                layer_head_mask = head_mask[idx] if head_mask is not None else None
                text_layer_module = self.text_layer[idx]
                text_layer_output = text_layer_module(
                        text_hidden_states,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        visual_hidden_state=None,
                        past_key_values=text_past_key_values,
                        output_attentions=output_attentions,
                        output_qks=output_qks,
                        current_layer=idx,
                )
                text_hidden_states = text_layer_output[0]

            #texthiddenstates 16 60 768   
            
            # Stage-refined Context Sampler (SCS)
            # 1 denote keep token
            spatial_text_hidden_states = text_hidden_states[:, 1:]  # 16 59 768
            pred_token_score = self.token_selector[idx](spatial_text_hidden_states, prev_token_decision).reshape(bz,-1,2)
            token_hard_keep_decision = F.gumbel_softmax(pred_token_score, hard=True)[:, :, 0:1]
            if idx != 0:
                token_hard_keep_decision = token_hard_keep_decision
                token_hard_keep_decision = (token_hard_keep_decision != 0.).float()
            cls_policy = torch.ones(bz, 1, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            token_policy = torch.cat([cls_policy, token_hard_keep_decision], dim=1)
            prev_token_decision = token_hard_keep_decision

            spatial_vision_hidden_states = vision_hidden_states[:, 1:]
            pred_patch_score = self.patch_selector[idx](spatial_vision_hidden_states, prev_patch_decision).reshape(bz, -1,2)
            patch_hard_keep_decision = F.gumbel_softmax(pred_patch_score, hard=True)[:,:, 0:1]
            if idx != 0:
                patch_hard_keep_decision = patch_hard_keep_decision
                patch_hard_keep_decision = (patch_hard_keep_decision != 0.).float()
            patch_policy = torch.cat([cls_policy, patch_hard_keep_decision], dim=1)
            prev_patch_decision = patch_hard_keep_decision

            ## merge with attention mask
            token_policy = token_policy.unsqueeze(1).transpose(-1,-2)
            token_policy = (token_policy - 1)*1e4 + attention_mask

            patch_policy = patch_policy.unsqueeze(1).transpose(-1,-2)
            patch_policy = (patch_policy-1)*1e4


            ## reconstruction for MCG ###
            text_generator = self.text_generator[0]
            generated_text_hidden_states = text_generator(vision_hidden_states)
            
            vision_generator = self.vision_generator[0]
            generated_vision_hidden_states = vision_generator(text_hidden_states, token_policy)


            ## cycle consistency for MCG
            cycle_text_hidden_state =  text_generator(generated_vision_hidden_states)
            cycle_vision_hidden_state = vision_generator(generated_text_hidden_states)
            # cycle_text_hidden_state = generated_text_hidden_states
            # cycle_vision_hidden_state = generated_vision_hidden_states



            # if output_attentions:
            #     all_vision_attentions = all_vision_attentions + (vision_layer_output[1], )
            #     all_text_attentions = all_text_attentions + (text_layer_output[1], )
            if patch_policy != None and token_policy != None:
                patch_policy = patch_policy.squeeze(1).transpose(-1,-2)
                patch_policy = (patch_policy==0.).float()
                token_policy = token_policy.squeeze(1).transpose(-1,-2)
                token_policy = (token_policy==0.).float()

                all_patch_policy += (patch_policy, )
                all_token_policy += (token_policy, )

            #     else:
            all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
            all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )

            all_generated_text_hidden_states += (generated_text_hidden_states, )
            all_generated_vision_hidden_states += (generated_vision_hidden_states, )

            all_cycle_text_hidden_states += (cycle_text_hidden_state, )
            all_cycle_vision_hidden_states += (cycle_vision_hidden_state, )




        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        return BaseModelOutput(
            last_text_state=text_hidden_states, 
            last_vision_state=vision_hidden_states,
            hidden_states=all_text_hidden_states, attentions=all_text_attentions,
            all_generated_vision_hidden_states=all_generated_vision_hidden_states,
            all_generated_text_hidden_states=all_generated_text_hidden_states,
            vision_states= all_vision_hidden_states,
            all_cycle_text_hidden_states = all_cycle_text_hidden_states,
            all_cycle_vision_hidden_states = all_cycle_vision_hidden_states,
            all_token_policy= all_token_policy,
            all_patch_policy= all_patch_policy,
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(UnimoModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = UnimoEncoder(vision_config, text_config)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,

        pixel_values=None,
        aux_values=None,
        rcnn_values=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=None
    ):
        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*3

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # all encoder
        # *使用GAN组成的ENcoder
        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_text_state=sequence_output,
            last_vision_state=encoder_outputs.last_vision_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            all_generated_vision_hidden_states= encoder_outputs.all_generated_vision_hidden_states,
            all_generated_text_hidden_states=encoder_outputs.all_generated_text_hidden_states,
            vision_states= encoder_outputs.vision_states,
            all_cycle_vision_hidden_states = encoder_outputs.all_cycle_vision_hidden_states,
            all_cycle_text_hidden_states= encoder_outputs.all_cycle_text_hidden_states,
            all_patch_policy= encoder_outputs.all_patch_policy,
            all_token_policy=encoder_outputs.all_token_policy,
        )

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings



class Generator(nn.Module):
    def __init__(self, config, query_num):
        super().__init__()
        self.query_num = query_num

        self.modality_query = torch.nn.Parameter(torch.randn(self.query_num, config.hidden_size), requires_grad=True)
        generator_layer = TransformerDecoderLayer(config.hidden_size,config.num_attention_heads)
        self.generator_layer = TransformerDecoder(generator_layer, 1)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.modality_query.data.uniform_(-initrange, initrange)

    def forward(self,
        hidden_states,
        attention_mask=None,
    ):
        bsz,max_len, _ = hidden_states.size()
        if attention_mask is not None:
            attention_mask = (attention_mask.squeeze(1).squeeze(1)==0).sum(1)
            src_mask = (
            (1.0 - get_mask(attention_mask, max_len))
                .type(torch.bool)
                .to(hidden_states.device)
            )

        else:
            src_mask = None
        modality_query = self.modality_query.unsqueeze(0).repeat(bsz, 1, 1)
        encoder_output = hidden_states.transpose(0,1)
        modality_query = modality_query.transpose(0,1)
        output = self.generator_layer(memory=encoder_output, tgt = modality_query, memory_key_padding_mask = src_mask)
        return output.transpose(0,1)




def get_mask(nums, max_num):
    batch_size = nums.size(0)
    assert len(
        nums.size()) == 1, r"nums should be a tensor with [batchsize x 1]"
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    assert (
        len(non_pad_mask.size()) == 2
    ), "mask should have shape of (N, S) where N is the batch size, \
    and S is the sequence length. But got the sie of {}"                                                        .format(non_pad_mask.size())
    return non_pad_mask


class ContextSampler(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)