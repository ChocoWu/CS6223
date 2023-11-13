import json
import numpy as np
import pandas as pd
import os


def sample_data(data_path, save_path, sample_num):
    data = pd.read_csv(data_path, header=0)
    s_data = data.sample(n=sample_num)
    s_data.to_csv(save_path, header=['zh', 'en'], index=False)


def split_data(data_path, save_path, num):
    data = pd.read_csv(data_path, header=0)
    data_1 = data.iloc[:num, :]
    data_2 = data.iloc[num:, :]
    data_1.to_csv(os.path.join(save_path, 'sample_200_for_llm.csv'), header=['zh', 'en'], index=False)
    data_2.to_csv(os.path.join(save_path, 'sample_200_for_gold.csv'), header=['zh', 'en'], index=False)


if __name__ == '__main__':
    data_path = '../data/damo_mt_testsets_zh2en_news_wmt20.csv'
    save_path = '../data/sample_400.csv'
    # sample_data(data_path, save_path, sample_num=400)
    split_data(save_path, '../data/', 200)