#!/bin/sh

TEXT=data/wmt17_en_zh
DATADIR=data-bin/wmt17_en_zh
TRAIN=trainings/wmt17_en_zh

# download, unzip, clean and tokenize dataset. 
python ./preprocess/wmt.py

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
MOSESDECODER=../mosesdecoder
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/train en zh $TEXT/train.clean 1 80
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/valid en zh $TEXT/valid.clean 1 80
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/test en zh $TEXT/test.clean 1 80

# build subword vocab
SUBWORD_NMT=../subword-nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.en > $TEXT/${CODES}.en
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.zh > $TEXT/${CODES}.zh

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en < $TEXT/train.clean.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh < $TEXT/train.clean.zh > $TEXT/train.${NUM_OPS}.bpe.zh

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.en"
cat $TEXT/train.${NUM_OPS}.bpe.en | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.en

echo "Generating vocab: ${VOCAB}.zh"
cat $TEXT/train.${NUM_OPS}.bpe.zh | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.zh

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/valid.clean.en > $TEXT/valid.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh --vocabulary $TEXT/${VOCAB}.zh < $TEXT/valid.clean.zh > $TEXT/valid.${NUM_OPS}.bpe.zh

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/test.clean.en > $TEXT/test.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.zh < $TEXT/test.clean.zh > $TEXT/test.${NUM_OPS}.bpe.zh

# generate preprocessed data
echo "Preprocessing datasets..."
DATADIR=data-bin/wmt17_en_zh
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq preprocess -sourcelang en -targetlang zh \
    -trainpref $TEXT/train.${NUM_OPS}.bpe -validpref $TEXT/valid.${NUM_OPS}.bpe -testpref $TEXT/test.${NUM_OPS}.bpe \
    -thresholdsrc 3 -thresholdtgt 3 -destdir $DATADIR

