import os
os.environ["WORLD_SIZE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import *
            
def main(args):
    output_file_name = f"{args.model_name.split('/')[-1]}_k{args.topk}_a{args.alpha}_{args.dataset_name}"+".json"
    output_file = open(os.path.join(args.output_dir, output_file_name), "w")
    
    print("dataset:", args.dataset_name)
    print("alpha: ", args.alpha)
    print("top-k: ", args.topk)
    
    # load dataset
    print('Loading Dataset...')
    with open(os.path.join(args.data_dir, args.dataset_name+".jsonl"), 'r') as f:
        dataset = f.readlines()
    print(len(dataset), "Samples")
        
    # start inference
    print('Start Inference...')
    model.eval()
    target_model.eval()
    
    # generate batch
    evd_comp_messages = []
    ctx_gen_messages = []
    questions = []
    answers = []
    for idx, data in enumerate(dataset):
        data = json.loads(data)
        question = data['question']
        answer = data['answers']
        ret_docs = "\n".join([dic['text'] for dic in data['ctxs'][:args.topk]])
        evd_comp_prompt = evd_comp_prompt_temp.format(question=question, ret_docs=ret_docs)
        ctx_gen_prompt = ctx_gen_prompt_temp.format(question=question)
        evd_comp_message = [
            {"role": "system", "content": evd_comp_system},
            {"role": "user", "content": evd_comp_prompt},
        ]
        ctx_gen_message = [
            {"role": "system", "content": ctx_gen_system},
            {"role": "user", "content": ctx_gen_prompt},
        ]
        evd_comp_messages.append(evd_comp_message)
        ctx_gen_messages.append(ctx_gen_message)
        questions.append(question)
        answers.append(answer)
    
    nbatch = (len(evd_comp_messages)-1) // args.batch_size + 1
    for k in tqdm(range(nbatch)):
        start_idx = k*args.batch_size
        end_idx = min((k+1)*args.batch_size, len(evd_comp_messages))
        batch_size = end_idx - start_idx
        evd_comp_messages_batch = evd_comp_messages[start_idx: end_idx]
        ctx_gen_messages_batch = ctx_gen_messages[start_idx: end_idx]
        question_batch = questions[start_idx: end_idx]
        answer_batch = answers[start_idx: end_idx]
        evd_comp_outputs = tokenizer.apply_chat_template(
            evd_comp_messages_batch,
            add_generation_prompt=False,
            padding="longest",
            return_dict=True,
            return_tensors="pt"
        )
        ctx_gen_outputs = tokenizer.apply_chat_template(
            ctx_gen_messages_batch,
            add_generation_prompt=False,
            padding="longest",
            return_dict=True,
            return_tensors="pt"
        )
        if args.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            gen_tokens = torch.tensor([128006, 78191, 128007, 271]) # <|start_header_id|>assistant<|end_header_id|>\n\n
            gen_tokens = gen_tokens.repeat(batch_size, 1)
            gen_att = torch.tensor([1, 1, 1, 1])
            gen_att = gen_att.repeat(batch_size, 1)
            evd_comp_ids = torch.cat([evd_comp_outputs.input_ids, gen_tokens], dim=-1).to(model.device)
            ctx_gen_ids = torch.cat([ctx_gen_outputs.input_ids, gen_tokens], dim=-1).to(target_model.device)
            attention_mask = torch.cat([evd_comp_outputs.attention_mask, gen_att], dim=-1).to(model.device)
            attention_mask_target = torch.cat([ctx_gen_outputs.attention_mask, gen_att], dim=-1).to(target_model.device)
        else:
            evd_comp_ids = evd_comp_outputs.input_ids
            ctx_gen_ids = ctx_gen_outputs.input_ids
            attention_mask = evd_comp_outputs.attention_mask
            attention_mask_target = ctx_gen_outputs.attention_mask
        
        # ensemble decoding
        gen_ids = None
        past_key_values = None
        past_key_values_target = None
        for step in range(args.decoding_len):
            with torch.no_grad():
                outputs = model(
                    input_ids=evd_comp_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    attention_mask=attention_mask
                )
                lm_logits = outputs.logits
                past_key_values = outputs.past_key_values
                logit_for_next_step = lm_logits[:, -1:] 
                next_id = torch.argmax(logit_for_next_step, axis=-1) 
                
                if args.alpha != 0.0:
                    target_outputs = target_model(input_ids=ctx_gen_ids, 
                                                past_key_values=past_key_values_target, 
                                                use_cache=True,
                                                attention_mask=attention_mask_target)
                    lm_logits = target_outputs.logits
                    past_key_values_target = target_outputs.past_key_values
                    logit_for_next_step_target = lm_logits[:, -1:] 
                    ensembled_logits = logit_for_next_step * (1. - args.alpha) + logit_for_next_step_target * args.alpha
                    next_id = torch.argmax(ensembled_logits, axis=-1)
             
                if gen_ids == None:
                    gen_ids = next_id
                else:
                    gen_ids = torch.cat([gen_ids, next_id], dim=-1)
                
                complete = True 
                for i in range(len(gen_ids)):
                    if tokenizer.eos_token_id not in gen_ids[i]:
                        complete = False
                        break
                if complete:
                    break
                
                evd_comp_ids = ctx_gen_ids = next_id
                next_id_mask = next_id != tokenizer.eos_token_id
                attention_mask = torch.cat([attention_mask, next_id_mask], dim=-1)
                attention_mask_target = torch.cat([attention_mask_target, next_id_mask], dim=-1)
        
        # decode              
        for i in range(len(gen_ids)):
            if tokenizer.eos_token_id in gen_ids[i]:
                end = gen_ids[i].tolist().index(tokenizer.eos_token_id)
                for j in range(end,len(gen_ids[i])):
                    gen_ids[i][j] = tokenizer.eos_token_id
        gen_ctx_batch = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        # save 
        for i in range(batch_size):
            json_output = {"question": question_batch[i], "answer": answer_batch[i], "gen_ctx": gen_ctx_batch[i]} 
            output_file.write(json.dumps(json_output, ensure_ascii=False)+"\n")
        
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help='model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') 
    parser.add_argument('--target_model_name', '-tm', type=str, help='target_model_name', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--alpha', '-alpha', type=float, default=0.5)
    parser.add_argument('--decoding_len', '-len', type=int, help='decoding length', default=128)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--data_dir', type=str, default="data/")
    parser.add_argument('--output_dir', type=str, default="outputs/")
    parser.add_argument('--dataset_name', '--dataset', type=str, default="nq")
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.output_dir, args.target_model_name.split('/')[-1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
    print("Summarization Model:", args.model_name)
    if args.model_name == args.target_model_name:
        print("Target Model:", args.model_name)
        target_model = model
    else:
        print("Target Model:", args.target_model_name)
        target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
 
    main(args)
    