export DATASET=FB15kET
export SAVE_DIR_NAME=FB15kET
export LOG_PATH=./logs/FB15kET.out
export HIDDEN_DIM=100
export TEMPERATURE=0.5
export LEARNING_RATE=0.001
export TRAIN_BATCH_SIZE=128
export MAX_EPOCH=500
export VALID_EPOCH=25
export BETA=1
export LOSS=SFNA

export PAIR_POOLING=avg
export SAMPLE_ET_SIZE=3
export SAMPLE_KG_SIZE=7
export SAMPLE_ENT2PAIR_SIZE=6
export WARM_UP_STEPS=50
export TT_ABLATION=all

CUDA_VISIBLE_DEVICES=0 python ./run.py --dataset $DATASET --save_path $SAVE_DIR_NAME --hidden_dim $HIDDEN_DIM --temperature $TEMPERATURE --lr $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE --cuda --max_epoch $MAX_EPOCH --valid_epoch $VALID_EPOCH --beta $BETA --loss $LOSS \
  --pair_pooling $PAIR_POOLING --sample_et_size $SAMPLE_ET_SIZE --sample_kg_size $SAMPLE_KG_SIZE --sample_ent2pair_size $SAMPLE_ENT2PAIR_SIZE --warm_up_steps $WARM_UP_STEPS \
  --tt_ablation $TT_ABLATION \
  > $LOG_PATH 2>&1 &

python3 run.py --dataset FB15kET --save_path FB15kET --hidden_dim 100 --temperature 0.5 --lr 0.001  --train_batch_size 128 --cuda --max_epoch 500 --valid_epoch 25 --beta 1 --loss SFNA  --pair_pooling avg --sample_et_size 3 --sample_kg_size 7 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all

--dataset FB15kET --save_path FB15kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 10 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 7 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all

export DATASET=YAGO43kET
export SAVE_DIR_NAME=YAGO43kET
export LOG_PATH=./logs/YAGO43kET.out
export HIDDEN_DIM=100
export TEMPERATURE=0.5
export LEARNING_RATE=0.001
export TRAIN_BATCH_SIZE=128
export MAX_EPOCH=500
export VALID_EPOCH=25
export BETA=1
export LOSS=SFNA

export PAIR_POOLING=avg
export SAMPLE_ET_SIZE=3
export SAMPLE_KG_SIZE=8
export SAMPLE_ENT2PAIR_SIZE=6
export WARM_UP_STEPS=50
export TT_ABLATION=all

CUDA_VISIBLE_DEVICES=1 python ./run.py --dataset $DATASET --save_path $SAVE_DIR_NAME --hidden_dim $HIDDEN_DIM --temperature $TEMPERATURE --lr $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE --cuda --max_epoch $MAX_EPOCH --valid_epoch $VALID_EPOCH --beta $BETA --loss $LOSS \
  --pair_pooling $PAIR_POOLING --sample_et_size $SAMPLE_ET_SIZE --sample_kg_size $SAMPLE_KG_SIZE --sample_ent2pair_size $SAMPLE_ENT2PAIR_SIZE --warm_up_steps $WARM_UP_STEPS \
  --tt_ablation $TT_ABLATION \
  > $LOG_PATH 2>&1 &

python3 run.py --dataset YAGO43kET --save_path YAGO43kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 20 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 8 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all