import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from time import time, asctime, localtime
import logging
# 忽略not init权重的warning提示
from transformers import logging as log_ignore
log_ignore.set_verbosity_error()

from model.model import GANModel
from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1
from utils.utils import set_random_seed, model_select, parse_arg
from utils.evaluate import evaluate

from utils.TrainInputProcess import prepare_data
# docker run -it --init --gpus=all  --name sa -v /media/seasonubt/data/Research/EA/:/workspace cuda118 /bin/bash

# parameters
args = parse_arg()
# set random seed
set_random_seed(args.random_seed)
args.device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")


# 1、创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# 2、创建一个handler，用于写入日志文件
fh = logging.FileHandler(args.log_dir)
fh.setLevel(logging.DEBUG)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 3、定义handler的输出格式（formatter）
formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt="%m/%d %H:%M:%S")
# 4、给handler添加formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 5、给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


# file_name = 'input_' + args.text_model_name + '_clip.pt'
# data_input_file = os.path.join("data/", args.dataset_type, file_name)

file_name = 'input_' + args.text_model_name + "_" + args.image_model_name + '.pt'
file_path = os.path.join("data/", args.dataset_type, file_name)
print(file_path)
refresh = False
if not os.path.exists(file_path) or refresh:
    print("Preparing Data")
    prepare_data(args, file_path=file_path)
    
data_inputs = torch.load(file_path)



test_pairs = data_inputs["test"]["pairs"]
data_inputs["train"].pop("pairs")
data_inputs["dev"].pop("pairs")
data_inputs["test"].pop("pairs")

train_dataset = MyDataSet2(inputs=data_inputs["train"])
dev_dataset = MyDataSet2(inputs=data_inputs["dev"])
test_dataset = MyDataSet2(inputs=data_inputs["test"])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)



text_config, image_config = model_select(args)
vb_model = GANModel(args, text_config, image_config, text_num_labels=5, text_model_name=args.text_model_name,
                     image_model_name=args.image_model_name, alpha=args.alpha, beta=args.beta)

vb_model.to(args.device)
vb_model_dict = vb_model.state_dict()

# load pretrained model weights
# for k, v in image_pretrained_dict.items():
#     if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
#         vb_model_dict[k] = v
# for k, v in text_pretrained_dict.items():
#     if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
#         vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)





t_total = len(train_dataloader) * args.epochs

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
params_decay = [p for n, p in vb_model.named_parameters() if not any(nd in n for nd in no_decay)]
params_nodecay = [p for n, p in vb_model.named_parameters() if any(nd in n for nd in no_decay)]

optimizer_grouped_parameters = [{"params": params_decay, "weight_decay": args.weight_decay}, {"params": params_nodecay, "weight_decay": 0.0}, ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, no_deprecation_warning=True)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

global_step, epochs_trained = 0, 0
best_result = {"precision" : 0, "recall": 0, "f1": 0, "loss": 0}
tr_loss, logging_loss = 0.0, 0.0
vb_model.zero_grad()

epoch_start_time = time()
step_start_time = None

if not args.enable_log :
    print("Log is forbidden !!! ")
    logging.disable(logging.ERROR)
logger.info("======================== New Round =============================")
logger.info(f"{args.dataset_type}, add_gan:{args.add_gan}, add_gan_loss: {args.add_gan_loss}, add_gpt: {args.add_gpt}, text_model {args.text_model_name}")
logger.info(args)

for epoch in range(epochs_trained, int(args.epochs)):

    for step, batch in tqdm(enumerate(train_dataloader), desc="Train", ncols=50, total=len(train_dataloader)):
        vb_model.train()
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(args.device)  #['input_ids', 'attention_mask', 'labels', 'cross_labels', 'pixel_values'

        # with torch.autograd.detect_anomaly():
        outputs = vb_model(**batch)
        loss = outputs["loss"]
        
        if not torch.any(torch.isnan(loss)):
            
            loss.backward()
            # tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(vb_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                vb_model.zero_grad()
                global_step += 1

            # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            #     # # Log metrics
            #     # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            #     # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
            #     if step_start_time is None:
            #         step_start_time = time()
            #         print()
            #         logger.info(
            #             f"loss_{global_step}: {(tr_loss - logging_loss) / args.logging_steps}, epoch {epoch + 1}: {step + 1}/{num_steps}")
            #     else:
            #         log_tim = (time() - step_start_time)
            #         print()
            #         logger.info(
            #             f"epoch {epoch + 1}, loss: {(tr_loss - logging_loss) / args.logging_steps}")
            #         step_start_time = time()
            #     logging_loss = tr_loss

            # save model if args.save_steps>0
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Log metrics
                results, _ = evaluate(args, vb_model, test_dataloader, data_inputs["test"], test_pairs)
                if results["f1"] >= best_result["f1"]:
                    best_result = results
                    best_result["epoch"] = epoch
                print()
                logger.info("#RES: f1:{0:.3f}, precision:{1:.3f}, recall:{2:.3f}, loss:{3:.3f} at {4}".format(results["f1"], results["precision"], results["recall"], results["loss"], epoch))
                logger.info("Best: f1:{0:.3f}, precision:{1:.3f}, recall:{2:.3f}, loss:{3:.3f} at {4}".format(best_result["f1"], best_result["precision"], best_result["recall"], best_result["loss"], best_result["epoch"]))
                logger.info("---------")

                # for key, value in results.items():
                #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                # tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

    #     if 0 < args.max_steps < global_step:
    #         break

    # if 0 < args.max_steps < global_step:
    #     break




# if __name__ == "__main__":
#     main()