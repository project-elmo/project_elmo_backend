from transformers import (
    GPT2LMHeadModel,
    BertLMHeadModel,
    RobertaForCausalLM,
    AlbertModel,
    ElectraForCausalLM,
    LlamaModel,
    AutoModel,
)
import sys


def load_model(repo_id, path):
    if "gpt2" in repo_id:
        GPT2LMHeadModel.from_pretrained(repo_id, cache_dir=path)
    elif "bert" in repo_id:
        BertLMHeadModel.from_pretrained(repo_id, cache_dir=path)
    elif "roberta" in repo_id:
        RobertaForCausalLM.from_pretrained(repo_id, cache_dir=path)
    elif "albert" in repo_id:
        AlbertModel.from_pretrained(repo_id, cache_dir=path)
    elif "electra" in repo_id:
        ElectraForCausalLM.from_pretrained(repo_id, cache_dir=path)
    # elif "llama" in repo_id:
    #     LlamaModel.from_pretrained(repo_id, cache_dir=path)
    else:
        AutoModel.from_pretrained(repo_id, cache_dir=path)


if __name__ == "__main__":
    repo_id = sys.argv[1]
    path = sys.argv[2]
    load_model(repo_id, path)
