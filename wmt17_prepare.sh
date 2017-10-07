#!/bin/sh

TEXT=data/wmt17_en_zh
DATADIR=data-bin/wmt17_en_zh
TRAIN=trainings/wmt17_en_zh

# download, unzip, clean and tokenize dataset. 
python preprocess/wmt.py

# build subword vocab
SUBWORD_NMT=../subword-nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.en > $TEXT/${CODES}.en
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.zh > $TEXT/${CODES}.zh

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en < $TEXT/train.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh < $TEXT/train.zh > $TEXT/train.${NUM_OPS}.bpe.zh

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.en"
cat $TEXT/train.${NUM_OPS}.bpe.en | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.en

echo "Generating vocab: ${VOCAB}.zh"
cat $TEXT/train.${NUM_OPS}.bpe.zh | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.zh

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/valid.en > $TEXT/valid.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh --vocabulary $TEXT/${VOCAB}.zh < $TEXT/valid.zh > $TEXT/valid.${NUM_OPS}.bpe.zh

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/test.en > $TEXT/test.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.zh < $TEXT/test.zh > $TEXT/test.${NUM_OPS}.bpe.zh

# generate preprocessed data
echo "Preprocessing datasets..."
DATADIR=data-bin/wmt17_en_zh
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq preprocess -sourcelang en -targetlang zh \
    -trainpref $TEXT/train.${NUM_OPS}.bpe -validpref $TEXT/valid.${NUM_OPS}.bpe -testpref $TEXT/test.${NUM_OPS}.bpe \
    -thresholdsrc 3 -thresholdtgt 3 -destdir $DATADIR

