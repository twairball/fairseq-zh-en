"""
Download and prepare dataset for challenger.ai en-zh nmt
"""
import prepare

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN = {
    "url": "https://www.dropbox.com/s/m38haw5rhz9wdm2/train_clean.tgz",
    "source": "train_clean.en",
    "target": "train_clean.zh",
    "data_source": "train.en",
    "data_target": "train.zh",
}

VALID = {
    "url": "https://www.dropbox.com/s/ft2evgnh8taeonf/valid_clean.tgz",
    "source": "valid_clean.en",
    "target": "valid_clean.zh",
    "data_source": "valid.en",
    "data_target": "valid.zh",
}

SAMPLE = {
    "url": "https://www.dropbox.com/s/11i3ccsizgq8lgt/sample_train.tgz",
    "source": "sample_train.en",
    "target": "sample_train.zh",
    "data_source": "train.en",
    "data_target": "train.zh",    
}

DATA_DIR  = "data/challenger_nmt/"
TMP_DIR = "tmp/challenger_nmt/"

if __name__ == '__main__':
    for ds in [SAMPLE, VALID]:
        # dataset is already tokenized
        prepare.prepare_dataset(DATA_DIR, TMP_DIR, ds, tokenize=False)

    