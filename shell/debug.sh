#!/bin/bash

cd ~/code/fairseq
#pip install --user fairseq
pip install --user --editable .
pip install --user tensorboardX

export PATH="$HOME/.local/bin:$PATH"


EXP_NAME=debug_9_albert_gan
TASK=masked_lm
CRITERION=masked_lm_gan
ARCH=adsbrain_albert_base


DATA_DIR=~/data/bert_pretrain/small-data-bin
SAVE_DIR=~/data/experiment/bert_base/${EXP_NAME}/checkpoints
TENSORBOARD_LOGDIR=~/tensorboard/${DLWS_JOB_ID}/logs/${EXP_NAME}
mkdir -p ~/tensorboard/${DLWS_JOB_ID}/logs/${EXP_NAME}

RESTORE_FILE=

#train from scrach 0.001, continue train 0.0001
PEAK_LR=0.001          # Peak learning rate, adjust as needed
MAX_EPOCH=36
TOTAL_UPDATES=125000
WARMUP_UPDATES=24000
TOKENS_PER_SAMPLE=512
MAX_SENTENCES=16
UPDATE_FREQ=2

NUM_WORKERS=8
FP16_INIT_SCALE=8



fairseq-train --fp16 $DATA_DIR  \
    --memory-efficient-fp16 \
    --task $TASK --criterion $CRITERION \
    --arch $ARCH --sample-break-mode complete \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    --save-dir $SAVE_DIR \
    --num-workers $NUM_WORKERS  \
    --fp16-init-scale $FP16_INIT_SCALE \
    --ddp-backend=no_c10d \
    --max-epoch $MAX_EPOCH \
    --max-update $TOTAL_UPDATES --log-format tqdm \
    --bpe gpt2 --gpt2-encoder-json ~/data/bert_pretrain/gpt2_bpe/encoder.json \
    --gpt2-vocab-bpe ~/data/bert_pretrain/gpt2_bpe/vocab.bpe \
    --mask-whole-words
