#!/bin/sh

DATADIR=data-bin/wmt17_en_zh
TRAIN=trainings/wmt17_en_zh

fairseq optimize-fconv -input_model $TRAIN/fconv/model_best.th7 -output_model $TRAIN/fconv/model_best_opt.th7

fairseq generate -path $TRAIN/fconv/model_best_opt.th7 -datadir $DATADIR \
     -beam 10 -nbest 2 -dataset test -sourcelang en -targetlang zh | tee tmp/wmt17_en_zh/fconv_test.out
     