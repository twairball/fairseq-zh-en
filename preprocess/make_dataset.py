from __future__ import print_function
import numpy as np
import pandas as pd
import random
import argparse

"""
Random split dataset
"""
def split_dataset(df, val_pct=0.1, test_pct=0.1, shuffle=True):
    random.shuffle(df) if shuffle else df
    
    size_n = len(df)
    train_n = int(size_n * (1 - test_pct - val_pct))
    val_n = int(size_n * val_pct)
    test_n = int(size_n * test_pct)
    test_start_ind = train_n + val_n
    
    train_data = df[0:train_n]
    val_data = df[train_n:test_start_ind]
    test_data = df[test_start_ind:]

    return train_data, val_data, test_data

"""
Pipeline to clean, split, and output dataset
"""
def build_data(en_filepath, zh_filepath, val_pct=0.1, test_pct=0.1, shuffle=True):
    df_en = pd.read_csv(en_filepath, sep="%%%%", header=None, names=["en"])
    print("en: %s", (df_en.shape,))
    df_zh = pd.read_csv(zh_filepath, sep="%%%%", header=None, names=["zh"])
    print("zh: %s", (df_zh.shape,))
    
    df = pd.concat([df_en, df_zh], axis=1)
    df = df.dropna()
    print("df: %s", (df.shape,))

    # save memory
    del df_en
    del df_zh
    data = np.array(df)
    del df
    return data

def write_data(data, filenames=['train', 'valid', 'test'], output_dir="./"):
    train, valid, test = split_dataset(data)
    for data, filename in list(zip([train, valid, test], filenames)):
        pd.DataFrame(data[:,0]).to_csv(output_dir+filename+".en")
        pd.DataFrame(data[:,1]).to_csv(output_dir+filename+".zh")


parser = argparse.ArgumentParser()
parser.add_argument('--en', required=True, help='tokenized en filepath')
parser.add_argument('--zh', required=True, help='tokenized zh filepath')
parser.add_argument('--output_dir', required=True, help="output dir")

if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    data = build_data(opt.en, opt.zh)
    write_data(data, output_dir=opt.output_dir)