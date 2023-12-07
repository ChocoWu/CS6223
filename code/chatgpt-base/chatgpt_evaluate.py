import random
from datetime import datetime
import backoff
import openai
import argparse
from functools import lru_cache
from transformers import logging

logging.set_verbosity_error()
from utils import *


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                                     openai.error.APIError, openai.error.APIConnectionError,
                                     openai.error.Timeout, openai.error.ServiceUnavailableError))
def completions_with_backoff(**kwargs):
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y-%m-%d")
    system_content = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-04\nCurrent date:{currentTime}'
    print(system_content)
    return openai.ChatCompletion.create(
        model=kwargs['engine'],
        temperature=kwargs['temperature'],
        max_tokens=kwargs['max_tokens'],
        presence_penalty=kwargs['presence_penalty'],
        frequency_penalty=kwargs['frequency_penalty'],
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": kwargs['prompt']},
        ]
    )


@lru_cache(maxsize=1000)
def get_gpt_output(prompt, **kwargs):
    # gpt_logger.write(prompt)
    response = completions_with_backoff(prompt=prompt, engine=kwargs['engine'],
                                        temperature=kwargs['temperature'], max_tokens=kwargs['max_tokens'],
                                        presence_penalty=kwargs['presence_penalty'],
                                        frequency_penalty=kwargs['frequency_penalty'])

    response_str = response['choices'][0]['message']['content']
    gpt_logger.write(response_str)
    # gpt_logger.write('#' * 55)
    gpt_logger.write('*******************')
    return response_str


def parse_args():
    parser = argparse.ArgumentParser()

    # User options
    parser.add_argument('--exp', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--shot_number', type=int, default=8, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('--resume', type=str, default='')

    # GPT settings
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo', choices=['text-davinci-002', 'gpt-3.5-turbo'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=768,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    parser.add_argument('--ckpt_root', type=str, default='./checkpoint/')

    args = parser.parse_args()

    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("_%Y_%m_%d_%H_%M_%S")
    # args.exp = args.exp + currentTime

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.exp)
    create_dir(args.ckpt_path)
    _logger = Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    openai.api_key = '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'
    # openai.api_key = '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' # xinyu
    # GPT parameters
    gpt_args = dict(
        engine=args.engine,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    # TRAINING
    logger = Logger(os.path.join(args.ckpt_path, 'log.txt'))
    gpt_logger = Logger(os.path.join(args.ckpt_path, f'gpt_log.txt'))

    # load data
    with open("./samples_gold.json", "r") as f:
        samples_gold = json.load(f)
    
    with open("./samples_llm.json", "r") as f:
        samples_llm = json.load(f)

    # data = load_csv('../data/sample_200_for_gold.csv')

    def get_prompt(source, translation):
        return f'''
Instruction: As a Chinese-to-English translation evaluator, your task is to assess the quality of translated English sentences. Examine the provided source and translations, focusing on aspects like grammar swapping, grammar agreement, punctuation, semantic substitution, semantic omission, semantic addition, and semantic ambiguity. Identify and enumerate any major errors, which include actual translation or grammatical/semantic errors. Be concise in listing the errors without providing modified sentences.  
Source: 设立新的奖项计划是为了表彰那些礼貌得体的政治人士。
Translation: The Wednesday establishment of a new award programs is to recognize those polite and well-mannered figures political. 
Output: 
(1) words: programs (error type: grammar agreement)
(2) words: figures --- political (error type: grammar swapping)
(3) words: Wednesday (error type: semantic addition)

Source: {source}
Translation: {translation}
Output:
'''
    for i, data in enumerate([samples_gold, samples_llm]):

        evaluation_res = []
        
        for idx, instance in enumerate(data):
            source = instance["zh_sent"]
            translation = instance["en_modi_sent"]
            
            prompt = get_prompt(source, translation)

            print(prompt)

            res = get_gpt_output(prompt, **gpt_args)

            evaluation_res.append({"zh_sent": source,
                                   "en_modi_sent": translation,
                                   "detected_error": res})
                
        if evaluation_res:
            if i==0: filename="gold"
            else: filename="llm"
            save_json(evaluation_res, f"eval_{filename}_complex.json")


