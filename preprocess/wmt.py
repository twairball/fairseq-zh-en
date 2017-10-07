# -*- coding: utf-8 -*-

"""
Download and prepare dataset for wmt17 zh-en
"""
from __future__ import unicode_literals, division

import sys
import codecs
import io
import argparse

import prepare
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
TMP_DIR = "tmp/wmt17_en_zh/"

if __name__ == '__main__':
    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    for ds in [WMT17_TRAIN_ZHEN, WMT17_DEV_ZHEN, WMT17_TEST_ZHEN]:
        prepare.prepare_dataset(DATA_DIR, TMP_DIR, ds)

    