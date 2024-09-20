import os
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM 
from prompts import *

def eval_ppl(qa_prompt, prefix_prompt):
    if args.target_model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        system_prompt = eval_system_mistral
    else:
        system_prompt = eval_system
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qa_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prefix_prompt},
    ]
    prefix_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    target_ids = input_ids.clone()
    target_ids[:, :prefix_ids.shape[-1]-1] = -100
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids.to(model.device), labels=target_ids.to(model.device))
        ppl = torch.exp(output.loss)
    return ppl

def main(args):

    file_name = f"{args.model_name.split('/')[-1]}_k{args.topk}_a{args.alpha}_{args.dataset}"
    log_file = open(os.path.join(args.output_dir, f"{args.target_model_name.split('/')[-1]}_eval_ppl_{args.dataset}_{args.alpha}.txt"), "w")
    with open(os.path.join(args.input_dir, file_name+".json"), "r") as fin:
        lines = fin.readlines()

    if args.dataset == "nq":
        eval_prompt_temp = q_c_nq_eval_prompt_temp
    elif args.dataset == "tqa":
        eval_prompt_temp = q_c_tqa_eval_prompt_temp
    elif args.dataset == "hotpotqa":
        eval_prompt_temp = q_c_hotpotqa_eval_prompt_temp
    elif args.dataset == "musique":
        eval_prompt_temp = q_c_musique_eval_prompt_temp
    elif args.dataset == "wiki":
        eval_prompt_temp = q_c_wiki_eval_prompt_temp

    eval_ppl_prefix_temp = eval_prompt_temp.split('{context}')[0]
    eval_ppl_temp =  eval_ppl_prefix_temp + "{context}"
    
    ppl_list = []
    for line in tqdm(lines):
        dic = json.loads(line)
        question = dic["question"]
        context = dic["gen_ctx"]
        qa_prompt = eval_ppl_temp.format(context=context, question=question)
        prefix_prompt = eval_ppl_prefix_temp.format(question=question)
        ppl = eval_ppl(qa_prompt, prefix_prompt)
        ppl_list.append(ppl.item())
    ppl_list.sort()
    buffer_count = int(len(lines)/4)
    ppl_list = ppl_list[buffer_count:-buffer_count]
    avg_ppl = sum(ppl_list) / len(ppl_list)
    log_file.write(f"{avg_ppl}") 
    log_file.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help='model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') # meta-llama/Meta-Llama-3-8B-Instruct  mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--target_model_name', '-tm', type=str, help='target_model_name', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--alpha', '-alpha', type=float, default=0.5)
    parser.add_argument('--input_dir', type=str, default="outputs/")
    parser.add_argument('--output_dir', type=str, default="evaluations/ppl/")
    parser.add_argument('--dataset', type=str, default="nq")
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.target_model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    args.input_dir = os.path.join(args.input_dir, args.target_model_name.split('/')[-1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)