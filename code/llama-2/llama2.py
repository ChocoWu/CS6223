import json 
import os
import random
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from prompt import *
from utils import*
from transformers import logging
logging.set_verbosity_error()


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []

    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sent = sent.split('</s>')[0]
        sent = sent.split('\n\n\n')[0]
        sent = sent.split("\n\n")[0]
        sents.append(sent)
    return sents

def generate_result(model, tokenizer, input_txt):
    zh_sent = input_txt['zh_sent']
    en_modi_sent = input_txt['en_modi_sent']

    # print(f"Input text: {input_txt}\n")

    # prompt =  SIMPLE_PROMPT_INSTRUCTION + DEMONSTRATION + f"\nNow, please give your evaluation for the following translations:\nSource: {zh_sent}\nTranslation: {en_modi_sent}" + " [/INST]"
    prompt =  COMPLEX_PROMPT_INSTRUCTION + DEMONSTRATION + f"\nNow, please give your evaluation for the following translations:\nSource: {zh_sent}\nTranslation: {en_modi_sent}" + " [/INST]"
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # print(f"input_ids: {input_ids}")
    raw_text_len = len(input_ids)
    # print(f"raw_text_len {raw_text_len}")
    input_ids = input_ids.to(model.device)
    outputs = model.generate(input_ids,
                            num_return_sequences=1,
                            max_length=800, 
                            use_cache=True,
                            temperature=0.1,
                            do_sample=True,
                            eos_token_id=[tokenizer.eos_token_id]
                    )
    
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    logger.write(output_text)
    output_text = output_text.split('[/INST]')[-1].strip()
    logger.write("\n***********************\n")
    return output_text


def repair(model, tokenizer, input_txt, with_error=False):
    zh_sent = input_txt['zh_sent']
    en_modi_sent = input_txt['en_modi_sent']

    if with_error:
        prompt =  REPAIR_PROMPT_INSTUCTION_start + f"Source: {zh_sent}\nTranslation: {en_modi_sent}\n" + REPAIR_PROMPT_INSTUCTION_med + "\n".join(input_txt["detected_errors"]) + REPAIR_PROMPT_INSTUCTION_end
    else:
        prompt =  REPAIR_PROMPT_INSTUCTION_start + f"Source: {zh_sent}\nTranslation: {en_modi_sent}\n" + REPAIR_PROMPT_INSTUCTION_end_wo

    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    raw_text_len = len(input_ids)
    input_ids = input_ids.to(model.device)
    outputs = model.generate(input_ids,
                            num_return_sequences=1,
                            max_length=800, 
                            use_cache=True,
                            temperature=0.1,
                            do_sample=True,
                            eos_token_id=[tokenizer.eos_token_id]
                    )
    
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    logger.write(output_text)
    output_text = output_text.split('[/INST]')[-1].strip()
    logger.write("\n***********************\n")
    return output_text


def sample_data_for_repair(data_path_1, data_path_2, save_path):
    data = []
    with open(data_path_1, 'r', encoding='utf-8') as f:
        _temp = json.load(f)
        data.extend(_temp)
    with open(data_path_2, 'r', encoding='utf-8') as f:
        _temp = json.load(f)
        data.extend(_temp)
    res = []
    types_list = ["grammer_swap", "grammer_agreement", "grammer_punc", "semantic_omission", "semantic_substitution", "semantic_addition", "semantic_ambiguity"]
    for i in types_list:
        _temp_res = []
        for instance in data:
            if instance["modi_type"] == i and instance["is_killed"] == 1:
                _temp_res.append(instance)
        if i == "grammer_punc":
            res.extend(_temp_res)
        else:
            res.extend(random.sample(_temp_res, 4))
    print("the sample length is ", len(res))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        
    

def postprocess(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for instance in data:
        detected_errors = instance['detected_errors']
        errors = []
        _temp = detected_errors.split('\n')
        print(_temp)
        for i in _temp:
            print(i)
            if i and i[0].isdigit():
                errors.append(i)
        instance['detected_errors'] = errors
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def is_mutant_killed(data_path, save_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f) 
    for instance in tqdm(data, total=len(data)):
        errors_list = instance["detected_errors"]
        ori_word = instance["ori_word"]
        modi_word = instance["modi_word"]
        flag = 0
        for error in errors_list:
            if (ori_word in error) and (modi_word in error):
                instance["is_killed"] = 1
                flag = 1
                break
        if flag == 0:
            instance["is_killed"] = 0
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def count_result(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = {}
    for instance in data:
        if instance["modi_type"] in results.keys():
            if instance["is_killed"] == 1:
                if "kill" in results[instance["modi_type"]].keys():
                    results[instance["modi_type"]]["kill"] += 1
                else:
                    results[instance["modi_type"]]["kill"] = 1
            else:
                if "not_kill" in results[instance["modi_type"]].keys():
                    results[instance["modi_type"]]["not_kill"] += 1
                else:
                    results[instance["modi_type"]]["not_kill"] = 1
        else:
            if instance["is_killed"] == 1:
                results[instance["modi_type"]] = {"kill": 1}
            else:
                results[instance["modi_type"]] = {"not_kill": 1}
        
    print(results)


if __name__ == "__main__":

    # # load model
    # # model_name = "./pretrained_ckpt/vicuna/7b-v1.5/"
    # model_name = "./pretrained_ckpt/llama-2/13b-chat/"

    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

    # repair the data based on the detected error
    # with open("repair_data.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # logger = Logger("./repair_log.txt")
    # for sample in tqdm(data, total=len(data)):
    #     outputs = repair(model, tokenizer, sample, with_error=True)
    #     sample["rectified_sent_w_error"] = outputs
    #     outputs = repair(model, tokenizer, sample, with_error=False)
    #     sample["rectified_sent_wo_error"] = outputs
    # with open("rectified_data.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)

    # postprocess the repaired the data
    with open("rectified_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for instance in data:
        rectified_sent_w_error = instance["rectified_sent_w_error"]
        instance["postprocess_rectified_sent_w_error"] = rectified_sent_w_error.split("\n")[3]
        rectified_sent_wo_error = instance["rectified_sent_wo_error"]
        instance["postprocess_rectified_sent_wo_error"] = rectified_sent_wo_error.split("\n")[3]
    with open("postprocess_rectified_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # data_path = "./complex/filter"
    # save_path = "./complex"
    # file_list = ["samples_gold", "samples_llm"]
    # for file_name in file_list:
    
    #     # load datasets
    #     with open(os.path.join(data_path, file_name+".json"), "r", encoding="utf-8") as f:
    #         data = json.load(f)
        
    #     logger = Logger(os.path.join(save_path, f"{file_name}_log.txt"))

    #     for sample in tqdm(data, total=len(data)):

    #         outputs = generate_result(model, tokenizer, sample)
    #         sample["detected_errors"] = outputs

    #     with open(os.path.join(save_path, f"detected_{file_name}.json"), "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=4, ensure_ascii=False)
        
    #     # postprocess data
    #     postprocess(os.path.join(save_path, f"detected_{file_name}.json"), os.path.join(save_path, f"postprocess_detected_{file_name}.json"))
        
    #     # is the mutant killed?
    #     is_mutant_killed(os.path.join(save_path, f"postprocess_detected_{file_name}.json"), os.path.join(save_path, f"mutant_detected_{file_name}.json"))
    
    #     # count the results
    #     count_result(os.path.join(data_path, f"mutant_detected_{file_name}.json"))

    # sample the data needed to be repaired
    #   sample_data_for_repair("./simple/filter/mutant_detected_samples_gold.json", "./simple/filter/mutant_detected_samples_llm.json", "./repair_data.json") 

    
