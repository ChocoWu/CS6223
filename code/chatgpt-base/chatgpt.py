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
    args.exp = args.exp + currentTime

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
    data = load_csv('../data/sample_200_for_llm.csv')

    for idx, instance in enumerate(data):
        en_data = instance['en']
        zh_data = instance['zh']
        prompt = 'Translate the following sentence: '
        prompt += zh_data
        print(prompt)
        res = get_gpt_output(prompt, **gpt_args)
        if res:
            logger.write(str(idx) + '\t' + zh_data + '\t' + en_data + '\t' + res)

