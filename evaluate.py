import os
import argparse
import json
import torch
import gc
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM 
from prompts import *
from utils import *
    
def generate_answer(qa_prompt, tokenizer, model):
    if args.target_model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        system_prompt = eval_system_mistral
    else:
        system_prompt = eval_system
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qa_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict = True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0
    )
    generated_tokens = outputs[:, inputs.input_ids.shape[-1]:]
    pred = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    pred = pred.strip()
    return pred

def main(args):

    in_file_name = f"{args.model_name.split('/')[-1]}_k{args.topk}_a{args.alpha}_{args.dataset_name}"
    out_file_name = f"{args.model_name.split('/')[-1]}_k{args.topk}_a{args.alpha}_{args.dataset_name}"
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.target_model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    with open(os.path.join(args.input_dir, in_file_name+".json"), "r") as fin:
        lines = fin.readlines()
    eval_out = open(os.path.join(args.output_dir, out_file_name+".json"), "w")
    log_file = open(os.path.join(args.output_dir, out_file_name+".txt"), "w")

    if args.target_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        if args.dataset_name == "nq":
            eval_prompt_temp = q_c_nq_eval_prompt_temp
        elif args.dataset_name == "tqa":
            eval_prompt_temp = q_c_tqa_eval_prompt_temp
        elif args.dataset_name == "hotpotqa":
            eval_prompt_temp = q_c_hotpotqa_eval_prompt_temp
        elif args.dataset_name == "musique":
            eval_prompt_temp = q_c_musique_eval_prompt_temp
        elif args.dataset_name == "wiki":
            eval_prompt_temp = q_c_wiki_eval_prompt_temp
    else:
        if args.dataset_name == "nq":
            eval_prompt_temp = c_q_nq_eval_prompt_temp
        elif args.dataset_name == "tqa":
            eval_prompt_temp = c_q_tqa_eval_prompt_temp
        elif args.dataset_name == "hotpotqa":
            eval_prompt_temp = c_q_hotpotqa_eval_prompt_temp
        elif args.dataset_name == "musique":
            eval_prompt_temp = c_q_musique_eval_prompt_temp
        elif args.dataset_name == "wiki":
            eval_prompt_temp = c_q_wiki_eval_prompt_temp
        
    total_em = 0
    total_f1 = 0
    total_pre = 0
    total_recall = 0
    total_acc = 0
    print("Total Samples: ", len(lines))
    
    for line in tqdm(lines):
        dic = json.loads(line)
        question = dic["question"]       
        context = dic["gen_ctx"]
        ground_truths = eval(dic["answer"]) if isinstance(dic["answer"], str) else dic["answer"]
        qa_prompt = eval_prompt_temp.format(context=context, question=question)
        pred = generate_answer(qa_prompt, tokenizer, model) 
        pred = parse_prediction(pred)
        dic.update({"prediction": pred, "answers": list(ground_truths)})
        em = em_score(pred, ground_truths)
        f1, pre, recall, acc = f1_score(pred, ground_truths)
        dic.update({"em": str(em), "f1": str(f1), "precision": str(pre), "recall": str(recall), "accuracy": str(acc)})
        eval_out.write(json.dumps(dic, ensure_ascii=False)+"\n")
        
        total_em += em
        total_f1 += f1
        total_pre += pre
        total_recall += recall
        total_acc += acc
        
    total_em /= len(lines)
    total_f1 /= len(lines)
    total_pre /= len(lines)
    total_recall /= len(lines)
    total_acc /= len(lines)
    print("EM: ", total_em)
    print("F1: ", total_f1)
    print("Precision: ", total_pre)
    print("Recall: ", total_recall)
    print("Accuracy: ", total_acc)
    log_file.write(f"EM: {total_em}\n")
    log_file.write(f"F1: {total_f1}\n")
    log_file.write(f"Precision: {total_pre}\n")
    log_file.write(f"Recall: {total_recall}\n")
    log_file.write(f"Accuracy: {total_acc}\n")
 
    eval_out.close()
    log_file.close()
    del model
    # torch.cuda.empty_cache()
    gc.collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help='model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') # meta-llama/Meta-Llama-3-8B-Instruct 
    parser.add_argument('--target_model_name', '-tm', type=str, help='target_model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') # mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--alpha', '-alpha', type=float, default=0.5)
    parser.add_argument('--input_dir', type=str, default="outputs/")
    parser.add_argument('--output_dir', type=str, default="evaluations/")
    parser.add_argument('--dataset_name', '--dataset', type=str, default="nq")
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    
    args.input_dir = os.path.join(args.input_dir, args.target_model_name.split('/')[-1])
    args.output_dir = os.path.join(args.output_dir, args.target_model_name.split('/')[-1])
    if not os.path.exists(os.path.join(args.output_dir)):
        os.makedirs(args.output_dir)
    
    main(args)