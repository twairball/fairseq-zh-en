# Chinese-English NMT

Experiments and reproduction of pretrained models trained on [WMT17 Chinese-English](http://www.statmt.org/wmt17/translation-task.html) using [fairseq](https://github.com/facebookresearch/fairseq)


#### Abstract 

A big pain point for any RNN/LSTM model training is that they are very time consuming, so `fairseq` proposed fully convolutional architecture is very appealing. Some cursory experiments show much faster training time for `fconv` (Fully Convolutional Sequence-to-Sequence) compared to `blstm` (Bi-LSTM), while yielding comparable results. While `fconv` measures slightly worse BLEU scores vs `blstm`, some manual tests seem to favor `fconv`. A hybrid model using `convenc` (Convolutional encoder, LSTM decoder) trains for much more epochs but performs much worse BLEU score. 


|Model | Epochs | Training Time | BLEU4 (beam1) | BLEU4 (beam5) | BLEU4 (beam10) | BLEU4 (beam20)|
|------|--------|---------------|---------------|---------------|----------------|---------------|
| fconv | 25 | ~4.5hrs | 63.49 | 62.22 | 62.52 | 62.74 |
| fconv_enc7 | 33 | ~5hrs | 66.40 | 65.52 | 65.8 | 65.96 |
| fconv_dec5 | 28 | ~5hrs | 65.65 | 64.71 | 64.91 | 64.98 |
| blstm | 30 | ~8hrs | 64.59 | 64.15 | 64.38 | 63.76 |
| convenc | 47 | ~7hrs | 50.91 | 56.71 | 56.83 | 53.66 |


# Download

Pretrained models:

- [wmt17.zh-en.fconv-cuda](https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/wmt17.zh-en.fconv-cuda.tgz): Pre-trained model for [WMT17 Chinese-English](http://www.statmt.org/wmt17/translation-task.html) 
- [wmt17.zh-en.fconv-float](https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/wmt17.zh-en.fconv-float.tgz): CPU version of the above

# Install

Follow `fairseq` installation, then:

````
# Chinese tokenizer
$ pip install jieba

# English tokenizer
$ pip install nltk
$ mkdir -p ~/nltk_data/tokenizers/
$ wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -o ~/nltk_data/tokenizers/punkt.zip
$ unzip ~/nltk_data/tokenizers/punkt.zip ~/nltk_data/tokenizers/

````

Additionally, we use scripts from Moses and Subword-nmt

````
git clone https://github.com/moses-smt/mosesdecoder
````

````
git clone https://github.com/rsennrich/subword-nmt
````

## Additional Setup

CUDA might need to link libraries to path. 

````
# Couldn't open CUDA library libcupti.so.8.0. LD_LIBRARY_PATH: /git/torch/install/lib:
$ cd $LD_LIBRARY_PATH; 
$ sudo ln -s  /usr/local/cuda-8.0/extras/CUPTI/lib64/* $LD_LIBRARY_PATH/
````


# Preprocessing

#### Word Token
We tokenize dataset, using `nltk.word_tokenizer` for English and `jieba` for Chinese word segmentation. 

#### Casing
We remove cases from English and converted all string to lower case. 

#### Merge blank lines
We note that dataset often has blank lines. In some cases, this is formatting, but there are cases where a long English sentence is translated to 2 Chinese sentences. This appears as a sentence followed by blank line on English corpus. To deal with this, we merge the 2 Chinese sentences onto same line, and then remove the blank line from both corpus. 

There are also formatting issues, where English corpus has blank line while Chinese corpus has a single `.`. We treat this as both blank lines and remove them. 

#### Additional data cleaning

We note that there are further work that can be added to data cleaning:
* remove non-English/Chinese sentences. I think there was a Russian sentence. 
* remove (HTML?) markup
* remove non-breaking white space. `\xa20` was found and converted to whitespace.


# Preprocessing

Preprocessing is run by `wmt_prepare.sh`.

1. We download, unzip, tokenize, and clean dataset in `preprocess/wmt.py`. 

2. We learn subword vocabulary using [apply_bpe](https://github.com/rsennrich/subword-nmt).

3. Then preprocess datasets to binary using `fairseq preprocess`

# Training

Run `wmt17_train.sh` which does the following: 

````
$ DATADIR=data-bin/wmt17_en_zh
$ TRAIN=trainings/wmt17_en_zh

# Standard bi-directional LSTM model
$ mkdir -p $TRAIN/blstm
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model blstm -nhid 512 -dropout 0.2 -dropout_hid 0 -optim adam -lr 0.0003125 \
    -savedir $TRAIN/blstm

# Fully convolutional sequence-to-sequence model
$ mkdir -p $TRAIN/fconv
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model fconv -nenclayer 4 -nlayer 3 -dropout 0.2 -optim nag -lr 0.25 -clip 0.1 \
    -momentum 0.99 -timeavg -bptt 0 -savedir $TRAIN/fconv

# Convolutional encoder, LSTM decoder
$ mkdir -p trainings/convenc
$ fairseq train -sourcelang en -targetlang zh -datadir $DATADIR \
    -model conv -nenclayer 6 -dropout 0.2 -dropout_hid 0 -savedir trainings/convenc
````

# Generate

Run `wmt17_generate.sh`, or run `generate-line` as follows:

````
$ DATADIR=data-bin/wmt17_en_zh

# Optional: optimize for generation speed (fconv only)
$ fairseq optimize-fconv -input_model trainings/fconv/model_best.th7 -output_model trainings/fconv/model_best_opt.th7

$ fairseq generate-lines -sourcedict $DATADIR/dict.en.th7 -targetdict $DATADIR/dict.zh.th7 -path trainings/fconv/model_best_opt.th7 -beam 10 -nbest 2
# you actually have to implement the solution
# <unk> 实际上 必须 实施 解决办法 。

$ fairseq generate-lines -sourcedict $DATADIR/dict.en.th7 -targetdict $DATADIR/dict.zh.th7 -path trainings/blstm/model_best.th7 -beam 10 -nbest 2
# you actually have to implement the solution
# <unk> ， 这些 方案 必须 非常 困难 

$ fairseq generate-lines -sourcedict $DATADIR/dict.en.th7 -targetdict $DATADIR/dict.zh.th7 -path trainings/convenc/model_best.th7 -beam 10 -nbest 2
# you actually have to implement the solution
# <unk> 这种 道德 又 能 实现 这些 目标 。 

````

---

# References

```
@article{gehring2017convs2s,
  author          = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title           = "{Convolutional Sequence to Sequence Learning}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1705.03122},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2017,
  month           = May,
}
```

```
@article{gehring2016convenc,
  author          = {Gehring, Jonas, and Auli, Michael and Grangier, David and Dauphin, Yann N},
  title           = "{A Convolutional Encoder Model for Neural Machine Translation}",
  journal         = {ArXiv e-prints},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1611.02344},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2016,
  month           = Nov,
}
```

# License
`fairseq` is licensed from its original repo. 

Pretrained models in this repo are BSD-licensed.