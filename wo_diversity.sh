export LANG="C.UTF-8"

CUDA_VISIBLE_DEVICES=1 /home/users/nus/dcshanw/miniconda3/envs/gec/bin/python -u /home/projects/11001764/wenjuan/gec_wj/beast19/software/fairseq-transformer/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --restore-file /home/projects/11001764/songyang/gec/beast19/transformer/ensemble/model_1_2_3_4/model4.pt --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 500  --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 19 --fp16 --save-dir wo_div/checkpoint_3 /home/projects/11001764/wenjuan/gec_wj/processed_sota
# /home/projects/11001764/wenjuan/gec_distillation/data/generate_debpe
# /home/projects/11001764/wenjuan/gec_wj/processed_toy

# CUDA_VISIBLE_DEVICES=3 /home/users/nus/dcshanw/miniconda3/envs/gec/bin/python -u /home/projects/11001764/wenjuan/gec_wj/beast19/software/fairseq-transformer/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 5000 --restore-file /home/projects/11001764//wenjuan/gec_wj/checkpoint3.pt --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 19 --fp16 --save-dir temp /home/projects/11001764/wenjuan/gec_distillation/data/generate_debpe
# CUDA_VISIBLE_DEVICES=0 /home/users/nus/dcshanw/miniconda3/envs/gec/bin/python -u /home/projects/11001764/wenjuan/gec_wj/beast19/software/fairseq-transformer/check_load.py \
#  --no-progress-bar  --path $models  --replace-unk --num-shards $threads $dictdir < $inputfile | tail -n +6 
 # > temp.loglog

# /home/projects/11001764/wenjuan/gec_wj/beast19/software/fairseq-transformer/load_weiqi_single_model.py   interactive
 # --beam $beam --nbest $nbest
 
