import os
import numpy as np
import torch
import random
import argparse
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (WEIGHTS_NAME, AutoConfig)
from transformers import BertForTokenClassification, RobertaForTokenClassification, AlbertForTokenClassification, ViTForImageClassification, SwinForImageClassification, DeiTModel, ConvNextForImageClassification
from transformers import T5ForConditionalGeneration, BloomForTokenClassification, DistilBertForTokenClassification, DebertaForTokenClassification, GPTNeoForTokenClassification, GPT2ForTokenClassification, BloomModel
from transformers import AutoTokenizer, BartModel, T5Model

import os
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel
from transformers import CLIPTextConfig, CLIPVisionConfig

def parse_arg():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--aspect_epochs', type=int, default=100)
    
    parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display a string')
    parser.add_argument('--task_name', type=str, default='dualc', nargs='?', help='display a string')
    parser.add_argument('--batch_size', type=int, default=4, nargs='?', help='display an integer')
    parser.add_argument('--output_result_file', type=str, default="./result.txt", nargs='?', help='display a string')
    parser.add_argument('--output_dir', type=str, default="./results", nargs='?', help='display a string')
    parser.add_argument('--log_dir', type=str, default="./data/log.log")
    parser.add_argument('--lr', type=float, default=2e-5, nargs='?', help='display a float')
    parser.add_argument('--epochs', type=int, default=100, nargs='?', help='display an integer')
    parser.add_argument('--alpha', type=float, default=0.6, nargs='?', help='display a float')
    parser.add_argument('--beta', type=float, default=0.6, nargs='?', help='display a float')
    parser.add_argument('--text_model_name', type=str, default="roberta", nargs='?')
    parser.add_argument('--image_model_name', type=str, default="vit", nargs='?')
    parser.add_argument('--random_seed', type=int, default=42, nargs='?')

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")  # origin 50
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every X updates steps.")  # origin 500
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    
    parser.add_argument('--device_id', type=str, default="cuda:0")

    parser.add_argument("--enable_log", action="store_true")
    parser.add_argument("--add_gan", action="store_true")
    parser.add_argument("--add_gan_loss", action="store_true", help="有提升就是会慢，平时可以收起来")
    parser.add_argument("--add_cycle", action="store_true")
    parser.add_argument("--add_gpt", action="store_true")
    parser.add_argument("--add_llm", action="store_true")
    parser.add_argument("--only_text_loss", action="store_true")

    args = parser.parse_args()
    args.name_path_dict = {
        "bert":     "../_weight/deberta",
        "roberta":  "../_weight/roberta-base",
        "deberta":  "../_weight/deberta",
        "clip":     "openai/clip-vit-base-patch32",
        "robertat": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "vit":      "../_weight/vit-base-patch16-224-in21k",
        "xlm":      'xlm-roberta-base',
        "debertal": "microsoft/deberta-v3-large",
        "laion":    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    }
    
    args.data_text_dir = '../data/twitter' + args.dataset_type
    args.data_image_dir = '../data/ImgData/twitter' + args.dataset_type
    args.cache_dir = 'cache'
    
    args.refresh_aspect = False
    args.refresh_data = False
    
    return args



def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

def model_select(args):
    if args.text_model_name == "clip":
        text_config = CLIPTextConfig()
    else:
        model_path1 = args.name_path_dict[args.text_model_name]
        text_config = AutoConfig.from_pretrained(model_path1)
    
    if args.image_model_name == "clip":
        image_config = CLIPVisionConfig()
    else:
        model_path1 = args.name_path_dict[args.image_model_name]
        image_config = AutoConfig.from_pretrained(model_path1)
        if args.image_model_name == "laion":
            image_config = image_config.vision_config
            # image_config.num_heads = image_config.num_attention_heads
    return text_config, image_config

    # text pretrained model selected
    if args.text_model_name == 'bert':
        model_path1 = './data/models/bert-base-uncased'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'roberta':  # HERE
        model_path1 = "../_weight/roberta-base"
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(
            model_path1).state_dict()
    elif args.text_model_name == 'albert':
        model_path1 = "./data/models/albert-base-v2"
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(
            model_path1).state_dict()
    elif args.text_model_name == 'electra':
        model_path1 = './models/electra-base-discriminator'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'flant5':
        model_path1 = './data/models/flant5'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = T5Model.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'bloom':
        model_path1 = './data/models/bloom'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = BloomModel.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'distilbert':
        model_path1 = './data/models/distilbert'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = DistilBertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'deberta':
        model_path1 = '../_weight/deberta'
        text_config = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = DebertaForTokenClassification.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'bart':
        model_path1 = './data/models/bart'
        tokenizer = AutoTokenizer.from_pretrained(model_path1)
        text_config = AutoConfig.from_pretrained(model_path1, vocab_size=len(tokenizer))
        text_pretrained_dict = BartModel.from_pretrained(model_path1).state_dict()
    elif args.text_model_name == 'clip':
        text_config = CLIPTextConfig()
        model_path1 = 'openai/clip-vit-base-patch32'
        text_pretrained_dict = CLIPTextModel.from_pretrained(model_path1).state_dict()

    # image pretrained model selected
    if args.image_model_name == 'vit':  # HERE
        model_path2 = "../_weight/vit-base-patch16-224-in21k"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
    if args.image_model_name == 'clip':  # HERE
        image_config = CLIPVisionConfig()
        model_path2 = 'openai/clip-vit-base-patch32'
        image_pretrained_dict = CLIPVisionModel.from_pretrained(model_path2).state_dict()
    elif args.image_model_name == 'swin':
        model_path2 = "./models/swin-tiny-patch4-window7-224"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(
            model_path2).state_dict()
    elif args.image_model_name == 'deit':
        model_path2 = "./models/deit-base-patch16-224"
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
    elif args.image_model_name == 'convnext':
        model_path2 = './models/convnext-tiny-224'
        image_config = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ConvNextForImageClassification.from_pretrained(
            model_path2).state_dict()
        
        
        
    return text_config, image_config, text_pretrained_dict, image_pretrained_dict


from transformers import CLIPConfig, CLIPModel
from transformers import CLIPTextConfig, CLIPVisionConfig


def _calculate_distillation_loss(features, teacher_features, T = 6, teacher_is_score=True):
    if teacher_is_score:
        teacher_prob=F.softmax(teacher_features/T, dim=-1)
    else:
        teacher_prob=teacher_features

    KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1),
                                         teacher_prob,
                                         reduction='none') * T
    return KD_loss.sum((1,2))

import ot
def ws_dis(aa, bb):
    loss = 0
    for a,b in zip(aa, bb):
        M = ot.dist(a, b, metric='euclidean').detach().cpu().numpy() # 距离计算方式, 'euclidean' / 'cosine'
        alpha = ot.unif(len(a))
        beta = ot.unif(len(b))
        
        pW = ot.emd2(alpha, beta, M)
        loss += pW

    return loss


def cal_loss(output):
    # print(len(output.all_generated_vision_hidden_states)) # 12
    # print(output.all_generated_vision_hidden_states[0].size())  8, 197, 768
    # print(output.vision_states[0].size()) # 8, 197, 768
    # print(output.all_patch_policy) 
    img_tag = 1
    cycle = False
    loss = 0
    if output.all_generated_vision_hidden_states and output.all_generated_text_hidden_states and output.vision_states and output.hidden_states:
        # print("-----")
        # print(output.all_generated_vision_hidden_states)
        # print(output.all_generated_text_hidden_states)
        vae_loss_t2v = [
            ws_dis(v, k.detach())  * m / m.sum((-1, -2))
            for v, k, m in zip(output.all_generated_vision_hidden_states,
                               output.vision_states, output.all_patch_policy)
        ]
        vae_loss_v2t = [
            ws_dis(v, k.detach()) * m / m.sum((-1, -2))
            for v, k, m in zip(output.all_generated_text_hidden_states,
                               output.hidden_states, output.all_token_policy)  #!
        ]

        vae_loss = ((sum(vae_loss_t2v) * img_tag).mean() +
                    (sum(vae_loss_v2t) * img_tag).mean()) / len(
                        output.all_generated_vision_hidden_states)
        loss += vae_loss * 0.001
        
        # if 0 < loss.item() < 100000  :
        #     pass
        # else:
        #     print(vae_loss_t2v.item(), vae_loss_v2t.item())
        #     print("vision", output.all_generated_vision_hidden_states)
        #     print("text", output.all_generated_text_hidden_states)
        #     exit()

        # if output.all_cycle_vision_hidden_states and output.all_cycle_text_hidden_states and cycle:
        #     cycle_loss_t = [
        #         _calculate_distillation_loss(v, k) for v, k in zip(
        #             output.all_cycle_text_hidden_states, output.hidden_states)
        #     ]
        #     cycle_loss_v = [
        #         _calculate_distillation_loss(v, k)
        #         for v, k in zip(output.all_cycle_vision_hidden_states,
        #                         output.vision_states)
        #     ]
        #     cycle_loss = (sum(cycle_loss_t) * img_tag).mean() + (
        #         sum(cycle_loss_v) * img_tag).mean() / len(
        #             output.all_generated_vision_hidden_states)
        #     loss += cycle_loss * 0.001
    return loss