import torch
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from transformers import LlamaTokenizer

# python_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# print("PYTHON_PATH", python_path)
# sys.path.append(python_path)
# import llm.print as print

from transformers import LlamaModel, LlamaConfig

def loadLLM(model_path = "./data/models/llama7bhf"):
    configuration = LlamaConfig(vocab_size=768)
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.float16)
    set_seed(42)

    model_args = {
        "model_name_or_path" : model_path,
        "gradient_checkpointing" : True,
    }

    config = AutoConfig.from_pretrained(model_args["model_name_or_path"])
    config.gradient_checkpointing = model_args["gradient_checkpointing"]
    config.output_hidden_states=True
    # if training_args.resume_from_checkpoint is not None:
    #     print(f'Load checkpoint from {training_args.resume_from_checkpoint}.')

    model = AutoModelForCausalLM.from_pretrained(
        model_args["model_name_or_path"],
        local_files_only=True,
        config=config,
    )
        
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args["model_name_or_path"],
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    return model, tokenizer


if __name__ == "__main__":
    
    model, tokenizer = loadLLM()
    model.to("cuda:0")
    text = "hello world"
    token = tokenizer(text, truncation=True, max_length=128, padding="longest")
    token2 = tokenizer(text, truncation=True, max_length=128, padding="max_length")
    print(token)
    print(token2)
    print(token.keys())
    token = torch.tensor(token).cuda().unsqueeze(0)
    feature = model(token)
    # print(feature)
    # print(feature["logits"].size())
    # print(feature.keys())
