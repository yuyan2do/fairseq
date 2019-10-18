#!/bin/bash

cd ~/code/fairseq
#pip install --user fairseq
pip install --user --editable .
pip install --user tensorboardX

export PATH="$HOME/.local/bin:$PATH"


#train from scrach 0.001, continue train 0.0001
PEAK_LR=0.0001          # Peak learning rate, adjust as needed

DATA_DIR=~/data/bert_pretrain/data-bin
EXAP_NAME=continue_train_fill_blank_embed
SAVE_DIR=~/data/experiment/bert_base/${EXAP_NAME}/checkpoints
mkdir -p ~/tensorboard/${DLWS_JOB_ID}/logs/${EXAP_NAME}
TENSORBOARD_LOGDIR=~/tensorboard/${DLWS_JOB_ID}/logs/${EXAP_NAME}


TOTAL_UPDATES=3565
WARMUP_UPDATES=2857 
TOKENS_PER_SAMPLE=512   
MAX_SENTENCES=16      # MAX_SENTENCES=16  
UPDATE_FREQ=64 #UPDATE_FREQ=128
DATA_DIR=~/data/bert_pretrain/data-bin

ROBERTA_PATH=~/data/roberta.base/adsbrain_model_position_head.pt

if [ -z "$1" ]
then
    echo "training model"
else
    echo "debug model"
    DATA_DIR=~/data/bert_pretrain/small-data-bin
    UPDATE_FREQ=2 #UPDATE_FREQ=128
    EXAP_NAME=debug_tmp
fi

CUDA_VISIBLE_DEVICES=1,2 \
fairseq-train --fp16 $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --task masked_position --criterion masked_position \
    --arch adsbrain_roberta_base --sample-break-mode complete \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    --save-dir $SAVE_DIR \
    --num-workers 0  \
    --fp16-init-scale 2 \
    --ddp-backend=no_c10d \
    --max-epoch 10 \
    --max-update $TOTAL_UPDATES --log-format tqdm \
#    --fill-avg-position-weight
