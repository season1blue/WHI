from transformers import AutoConfig,TrainingArguments,Trainer,EvalPrediction,BertForTokenClassification,ViTForImageClassification
from typing import Callable, Dict


def cal_f1(p_pred_labels, text_inputs, p_pairs, is_result=False):
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
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp > 1:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 2
                    flag = True
                elif pp == 1:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)
    precision = 0
    recall = 0
    f1 = 0

    precision = correct_num / predict_num if predict_num != 0 else 0
    recall = correct_num / gold_num if gold_num != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0

    if is_result:
        return precision * 100, recall * 100, f1 * 100, pred_pair_list
    else:
        return precision * 100, recall * 100, f1 * 100

from difflib import SequenceMatcher
def count_similar_tuples(a, b):
    # 初始化相似元组计数器
    similar_count = 0
    
    # 遍历a中的每个元组
    for a_tuple in a:
        a_text, a_num = a_tuple
        # 遍历b中的每个元组
        for b_tuple in b:
            b_text, b_num = b_tuple
            # 如果文本相似且数字匹配
            if SequenceMatcher(None, a_text, b_text).ratio() > 0.8 and a_num == b_num:
                similar_count += 1
                break  # 找到相似的就跳出内层循环，避免重复计数
    
    return similar_count

def cal_pair_f1(pred_result, pred_aspect, true_pairs):
    gold_num, predict_num, correct_num = 0, 0, 0
    pred_pair_list = []
    
    
    pred_pairs = [ (aspect[0], int(sentiment)) for aspect, sentiment in zip(pred_aspect, pred_result)]
    
    
    flatten_true_pairs = []
    for pair in true_pairs:
        for p in pair:
            flatten_true_pairs.append(p)
            
    # print(len(pred_pairs))  # ('Meeks', 0)
    # print(len(flatten_true_pairs))  #('8-8', 'Meeks', 1)

    flatten_true_pairs = [(f[1], f[2]) for f in flatten_true_pairs]
    
    gold_num = len(flatten_true_pairs)
    predict_num = len(pred_pairs)
    correct_num = count_similar_tuples(pred_pairs, flatten_true_pairs)

    precision = correct_num / predict_num if predict_num != 0 else 0
    recall = correct_num / gold_num if gold_num != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision != 0 or recall != 0 else 0

    return precision, recall, f1



def cal_single_f1(p_pred_labels,p_inputs,p_pairs,is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = p_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue

            if word_ids[j] != word_ids[j - 1]:
                if 0<pp<4:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 1
                    flag = True
                elif pp == sentiment + 4:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(pred_pair)
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)

    precision = 0
    recall = 0
    f1 = 0
    if predict_num != 0:
        precision = correct_num / predict_num
    if gold_num != 0:
        recall = correct_num / gold_num
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    if is_result:
        return precision * 100, recall * 100, f1 * 100, pred_pair_list
    else:
        return precision * 100, recall * 100, f1 * 100


