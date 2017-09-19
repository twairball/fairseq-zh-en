"""
Download and prepare dataset for wmt17 zh-en
"""
import prepare

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

WMT17_TRAIN_ZHEN = {
    "url": "http://data.statmt.org/wmt17/translation-task/"
                          "training-parallel-nc-v12.tgz",
    "source": "training/news-commentary-v12.zh-en.en",
    "target": "training/news-commentary-v12.zh-en.zh",
    "data_source": "train.en",
    "data_target": "train.zh",
}

WMT17_DEV_ZHEN = {
    "url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    "source": "dev/newsdev2017-zhen-ref.en.sgm",
    "target": "dev/newsdev2017-zhen-src.zh.sgm",
    "data_source": "valid.en",
    "data_target": "valid.zh",
}

WMT17_TEST_ZHEN = {
    "url": "http://data.statmt.org/wmt17/translation-task/test.tgz",
    "source": "test/newstest2017-zhen-ref.en.sgm",
    "target": "test/newstest2017-zhen-src.zh.sgm",
    "data_source": "test.en",
    "data_target": "test.zh",
}

DATA_DIR  = "data/wmt17_en_zh/"
TMP_DIR = "./tmp/"

if __name__ == '__main__':
    for ds in [WMT17_TRAIN_ZHEN, WMT17_DEV_ZHEN, WMT17_TEST_ZHEN]:
        prepare.prepare_dataset(DATA_DIR, TMP_DIR, ds)

    