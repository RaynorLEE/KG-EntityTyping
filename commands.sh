# SEM BERT
python run.py --dataset FB15kET --save_path FB15kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 10 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 7 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all
python run.py --dataset YAGO43kET --save_path YAGO43kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 20 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 8 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all

# SEM Roberta
python run.py --dataset FB15kET --save_path FB15kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 10 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 7 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all
python run.py --dataset YAGO43kET --save_path YAGO43kET --hidden_dim 100 --temperature 0.5 --lr 5e-5 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 20 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 8 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all

# SKA
python run.py --dataset FB15kET --save_path FB15kET --hidden_dim 100 --temperature 0.5 --lr 1e-3 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 25 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 7 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all --nhop 2 --rerank_ratio 0.5 --rerank_scope 50 --lambda 1.0

python run.py --dataset YAGO43kET --save_path YAGO43kET --hidden_dim 100 --temperature 0.5 --lr 2e-3 --train_batch_size 32 --cuda --max_epoch 500 --valid_epoch 20 --beta 1 --loss SFNA --pair_pooling avg --sample_et_size 3 --sample_kg_size 8 --sample_2hop_kg_size 8 --sample_ent2pair_size 6 --warm_up_steps 50 --tt_ablation all --nhop 2 --rerank_ratio 0.5 --rerank_scope 50 --lambda 1.0