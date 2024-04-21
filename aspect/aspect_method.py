import torch
import os
from tqdm import tqdm
import collections
from PIL import Image,ImageFile
from transformers import AutoTokenizer, AutoProcessor
from aspect.aspect_model import ASPModel
from aspect.module import MultiHeadAttention, optimal_transport_dist

from transformers import (WEIGHTS_NAME, AutoConfig)
from transformers import BertForTokenClassification, RobertaForTokenClassification, AlbertForTokenClassification, ViTForImageClassification, SwinForImageClassification, DeiTModel, ConvNextForImageClassification
from transformers import T5ForConditionalGeneration, BloomForTokenClassification, DistilBertForTokenClassification, DebertaForTokenClassification, GPTNeoForTokenClassification, GPT2ForTokenClassification, BloomModel
from transformers import AutoTokenizer, BartModel, T5Model
from torch.utils.data import Dataset

import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from utils.utils import model_select
from copy import deepcopy


def logits2span_bk(p_pred_labels, text_inputs, p_pairs):
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = text_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = []
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.append([start_pos, end_pos])
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp == 1:  # B
                    if flag:
                        pred_pair.append([start_pos, end_pos])
                    start_pos = j
                    end_pos = j
                    sentiment = pp - 2
                    flag = True
                elif pp == 2:  # I
                    if flag:
                        end_pos = j
                else: #pp=0 O
                    if flag:
                        pred_pair.append([start_pos, end_pos])
                    flag = False
        pred_pair_list.append(pred_pair.copy())
    return pred_pair_list

def logits2span(p_pred_labels, text_inputs, p_pairs):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = text_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp == 1:  # B
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 2
                    flag = True
                elif pp == 2:  # I
                    if flag:
                        end_pos = word_ids[j]
                else:  #pp=0 O
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                    flag = False
        true_pair = set(pairs[0] for pairs in p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)
    return pred_pair_list

def cal_f1(p_pred_labels, text_inputs, p_pairs, is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list, aspect_list = [], []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = text_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        aspect = []
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                    aspect.append((start_pos, end_pos))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp == 1:  # B
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                        aspect.append((start_pos, end_pos))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 2
                    flag = True
                elif pp == 2:  # I
                    if flag:
                        end_pos = word_ids[j]
                else:  #pp=0 O
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos)))
                        aspect.append((start_pos, end_pos))
                    flag = False
        true_pair = set(pairs[0] for pairs in p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        pred_pair_list.append(pred_pair.copy())
        aspect_list.append(aspect.copy())
        correct_num += len(true_pair & pred_pair)
    precision = 0
    recall = 0
    f1 = 0

    precision = correct_num / predict_num if predict_num != 0 else 0
    recall = correct_num / gold_num if gold_num != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0
    if is_result:
        return precision * 100, recall * 100, f1 * 100, aspect_list
    else:
        return precision * 100, recall * 100, f1 * 100

class aspect_dataset(Dataset):
    def __init__(self, file, args) -> None:
        self.file = file
        self.args = args
        self.file_type = file.split('/')[-1].split('.')[0]

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.name_path_dict[self.args.text_model_name],add_prefix_space=True)  # text tokenizer
        self.processor = AutoProcessor.from_pretrained(self.args.name_path_dict[self.args.image_model_name])

        self.raw_data, self.pairs, self.sentence = self.process_data(self.args.refresh_aspect)

    def __len__(self):
        return len(self.raw_data["input_ids"])

    def __getitem__(self, index):
        d = dict()
        for key in self.raw_data.keys():
            d[key] = self.raw_data[key][index]
        return d



    def _get_data(self, file):
        sentence_d = collections.defaultdict(list)
        sentence_l, image_l, label_l, pair_l, senti_l, allabel_l = [], [], [] ,[] ,[], []
        with open(file, 'r', encoding="utf-8") as f:
            # Done split RT @ and etc
            while True:
                text = f.readline().rstrip('\n').split()
                # text = self.preprocess(text) # clean dataset, TODO re test
                if text == []:
                    break
                aspect = f.readline().rstrip('\n').split()
                sentiment = f.readline().rstrip('\n')
                image_path = f.readline().rstrip('\n')
                start_pos = text.index("$T$")
                end_pos = start_pos + len(aspect) - 1
                text = text[:start_pos] + aspect + text[start_pos+1:]
                sentence_d[" ".join(text)].append((start_pos, end_pos, sentiment, image_path))
            for key, value in sentence_d.items():

                text = key.split()
                sentence_l.append(text)
                n_key = len(text)
                s_label = [0] * n_key  # all is O
                s_senti = [-1] * n_key
                s_allabel = [0] * n_key
                s_pair = [] 
                image_l.append(value[0][3])
                for vv in value:
                    s_label[vv[0]] = 1  # B
                    for i in range(vv[0] + 1, vv[1] + 1):
                        s_label[i] = 2  # I

                    v_sentiment = int(vv[2]) + 1
                    for i in range(vv[0], vv[1]+1):
                        s_senti[i] = v_sentiment
                    # sentiment -1, 0, 1 -> 0, 1, 2

                    # 0代表实体外，1代表in，234代表实体开始及其情绪。 -1,0,1 ->2,3,4
                    s_allabel[vv[0]] = int(vv[2]) + 3 # B
                    for i in range(vv[0] + 1, vv[1] + 1):
                        s_allabel[i] = 1  # I

                    s_pair.append((str(vv[0]) + "-" + str(vv[1]), " ".join(text[vv[0]: vv[1]+1]), v_sentiment))  # [('4-5', 'Chuck Bass', 5), ('8-9', '# MCM', 4)]
                # print(text)  #['RT', '@', 'ltsChuckBass', ':', 'Chuck', 'Bass', 'is', 'everything', '#', 'MCM']
                # print(s_label)  # [0, 0, 0, 0, 1, 2, 0, 0, 1, 2]
                # print(s_senti)  # [-1, -1, -1, -1, 3, 3, -1, -1, 2, 2]
                # print(s_allabel)  # [0, 0, 0, 0, 4, 1, 0, 0, 3, 1]
                label_l.append(s_label)
                pair_l.append(s_pair)
                senti_l.append(s_senti)
                allabel_l.append(s_allabel)
        return sentence_l, image_l, label_l, pair_l, senti_l, allabel_l
    
    def tokenize_data(self, sentence_l, image_l, label_l, pair_l, senti_l, allabel_l):
        images = []
        
        for image_path in tqdm(image_l, desc="aspect image"):
            img_path = os.path.join(self.args.data_image_dir, image_path)
            image = Image.open(img_path)
            image = image.convert('RGB')
            images.append(image)

        
        pixel_values = self.processor(images=images,return_tensors="pt")["pixel_values"]
        new_sentence_l = []
        for sentence in sentence_l:
            new_sentence_l.append(" ".join(sentence))

        tokenized_inputs = self.tokenizer(sentence_l, truncation=True, is_split_into_words=True, padding='max_length', max_length=60, return_tensors='pt')

        text_labels, cross_labels, senti_labels, all_labels = [], [], [], []
        for i, label in enumerate(label_l):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            label_ids = []
            cross_label_ids = []
            senti_ids, allabel_ids = [], []
            label_n = len(label)
            pre_word_idx = None
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None or word_idx >= label_n:
                    label_ids.append(-100)
                    senti_ids.append(-100)
                    allabel_ids.append(-100)
                    cross_label_ids.append(0)
                else:
                    if pre_word_idx != word_idx:
                        label_ids.append(label[word_idx])
                        senti_ids.append(senti_l[i][word_idx])
                        cross_label_ids.append(label[word_idx])
                        allabel_ids.append(allabel_l[i][word_idx])
                    else:
                        label_ids.append(-100)
                        senti_ids.append(-100)
                        allabel_ids.append(-100)
                        cross_label_ids.append(0)
                pre_word_idx = word_idx
            cross_labels.append(cross_label_ids)
            text_labels.append(label_ids)
            senti_labels.append(senti_ids)
            all_labels.append(allabel_ids)
        tokenized_inputs["labels"] = torch.tensor(text_labels)
        tokenized_inputs["pairs"] = pair_l
        tokenized_inputs["cross_labels"] = torch.tensor(cross_labels)
        tokenized_inputs["senti_labels"] = torch.tensor(senti_labels)
        tokenized_inputs["all_labels"] = torch.tensor(all_labels)
        tokenized_inputs["pixel_values"] = pixel_values

        return tokenized_inputs

    def process_data(self, refresh_data=True):
        file_name = self.file_type + ".pt"
        inputs_dir = os.path.join(self.args.cache_dir, self.args.dataset_type, file_name)
        sentence_l, image_l, label_l, pair_l, senti_l, allabel_l = self._get_data(self.file)
        
        print(refresh_data)
        exit()
        if os.path.exists(inputs_dir) and not refresh_data:
            print("Loading data from save file")
            data = torch.load(inputs_dir)
            t = data.word_ids
        else:
            print("reprocessing the data")
            tokenized_inputs = self.tokenize_data(sentence_l, image_l, label_l, pair_l, senti_l, allabel_l)
            data = tokenized_inputs
            torch.save(data, inputs_dir)
            t = data.word_ids
        
        pairs = deepcopy(data["pairs"])
        for k in ["token_type_ids", "pairs", "senti_labels", "all_labels"]:
            if k in data.keys():
                del data[k]

        return data, pairs, sentence_l

   



class aspect_method():
    def __init__(self, args) -> None:
        self.args = args
        # Prepare data
        self.train_data = aspect_dataset(os.path.join(self.args.data_text_dir, "train.txt"), self.args)
        self.dev_data = aspect_dataset(os.path.join(self.args.data_text_dir, "dev.txt"), self.args)
        # train_pairs = train_data.pairs
        # dev_pairs = dev_data.pairs
        


    def evaluate(self, args, model, eval_dataloader, text_inputs, pairs):
        eval_loss, nb_eval_steps = 0.0, 0
        model.to(args.device)
        model.eval()

        aspect_pred_list, span_list = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader, desc="aspect processing")):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(args.device)

                aspect_outputs = model(**batch)
                aspect_logits = np.argmax(aspect_outputs["logits"].detach().cpu(), -1) 
                aspect_span = logits2span(p_pred_labels=aspect_logits, text_inputs=text_inputs, p_pairs=pairs)
                aspect_loss, aspect_logits = aspect_outputs["loss"], aspect_outputs["logits"]


                eval_loss += aspect_loss  

                aspect_pred_labels = np.argmax(aspect_logits.cpu(), -1)
                aspect_pred_list.append(aspect_pred_labels)

                span_list += aspect_span

                nb_eval_steps += 1
        aspect_pred_sum = np.vstack(aspect_pred_list)
        aspect_precision, aspect_recall, aspect_f1, predict_span = cal_f1(aspect_pred_sum, text_inputs, pairs, is_result=True)

        eval_loss = eval_loss.item() / nb_eval_steps
        
        results = {"aspect_f1": aspect_f1, "aspect_precision" : aspect_precision, "aspect_recall": aspect_recall,  "loss": float(eval_loss)}


        return results, predict_span


    def train(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.args.batch_size)
        dev_dataloader = DataLoader(self.dev_data, batch_size=self.args.batch_size)

        # Load Model
        text_config, image_config = model_select(self.args)
        model = ASPModel(self.args, text_config, image_config, text_num_labels=3, alpha=self.args.alpha, beta=self.args.beta)  # 3=BIO
        model.to(self.args.device)
        model_dict = model.state_dict()

        # load pretrained model weights
        # for k, v in image_pretrained_dict.items():
        #     if model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        #         model_dict[k] = v
        # for k, v in text_pretrained_dict.items():
        #     if model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        #         model_dict[k] = v
        model.load_state_dict(model_dict)
        model.zero_grad()

        t_total = len(train_dataloader) * self.args.aspect_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        optimizer_grouped_parameters = [{"params": params_decay, "weight_decay": self.args.weight_decay}, {"params": params_nodecay, "weight_decay": 0.0}, ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon, no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)


        # Parameter Init
        global_step = 0
        best_result = {"aspect_precision" : 0, "aspect_recall": 0, "aspect_f1": 0, "loss": 0}

        best_model = None
        # Train
        for epoch in range(int(self.args.aspect_epochs)):
            for step, batch in tqdm(enumerate(train_dataloader), desc="Train", ncols=50, total=len(train_dataloader)):
                model.train()
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.args.device)  # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'pairs', 'cross_labels', 'senti_labels', 'all_labels', 'pixel_values'])

                # with torch.autograd.detect_anomaly():
                aspect_outputs = model(**batch)
                aspect_logits = np.argmax(aspect_outputs["logits"].detach().cpu(), -1) 
                # aspect_span = self.logits2span(p_pred_labels=aspect_logits, text_inputs=train_data, p_pairs=train_data["pairs"])

                loss = aspect_outputs["loss"]
                
                if not torch.any(torch.isnan(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            # Evaluate on every epoch end 
            results, _ = self.evaluate(self.args, model, dev_dataloader, self.dev_data.raw_data, self.dev_data.pairs)
            if results["aspect_f1"] >= best_result["aspect_f1"]:
                best_result = results
                best_model = model
                # save model 
                print(results)
                torch.save(model, os.path.join(self.args.cache_dir, "predict.pt"))


        # 完成predict，返回dev预测结果
        # aspect_result = self.predict()



        return best_model

    def predict(self, type="train"):
        data = self.train_data if type == "train" else self.dev_data
        dataloader = DataLoader(data, batch_size=self.args.batch_size)
        checkpoint_path = os.path.join(self.args.cache_dir, "predict.pt")


        if os.path.exists(checkpoint_path):
            print(checkpoint_path, "exists, loading")
            text_config, image_config = model_select(self.args)
            apsect_predict_model = ASPModel(self.args, text_config, image_config, text_num_labels=3, alpha=self.args.alpha, beta=self.args.beta)
            
            apsect_predict_model = torch.load(checkpoint_path)
            _, aspect = self.evaluate(self.args, apsect_predict_model, dataloader, data.raw_data, data.pairs)
            
        else:
            print(checkpoint_path, "is not exists")
            apsect_predict_model = self.train()
            _, aspect = self.evaluate(self.args, apsect_predict_model, dataloader, data.raw_data, data.pairs)

        # print(len(aspect))
        # print(len(data.sentence))
        
        # text_aspect = deepcopy(aspect)
        # for i, a in enumerate(aspect):
        #     print(i)
        #     print(a)
        #     print(data.sentence[i])
        #     text_aspect[i] = [data.sentence[i][
        #         max(0, t[0]) : min(len(data.sentence[i]), t[1]+1)
        #         ] 
        #         for t in a]
        #     print(text_aspect[i])
        #     exit()
        


        return aspect