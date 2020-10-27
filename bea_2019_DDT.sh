export LANG="C.UTF-8"

device_id=2
main_path=/home/projects/11001764/wenjuan/gec_wj
train_path=$main_path/beast19/software/fairseq-transformer
python_path=/home/users/nus/dcshanw/miniconda3/envs/gec/bin
model_path=/home/projects/11001764/songyang/gec/beast19/transformer/ensemble/model_1_2_3_4
#$main_path/checkpoint_before_ddc/model_1_2_3_4
processed_path=$main_path/processed_sota
# processed_conll
output_path=$main_path/bea19_outputs
data_name=1minusbleu_1234_offline_2iter
temp_file=1mimusbea_1234_bleu_temp
checkpoint_file=checkpoints
# mkdir $main_path/${temp_file}1
# mkdir $main_path/${temp_file}2
# mkdir $main_path/${temp_file}3
# mkdir $main_path/${temp_file}4
# CUDA_VISIBLE_DEVICES=$device_id $python_path/python -u $train_path/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --restore-file $model_path/model1.pt --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 500  --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 16 --fp16  --component-path $output_path  --departure-args-path edin kakao toho --save-dir $main_path/${temp_file}1 $processed_path
# wait
# CUDA_VISIBLE_DEVICES=$device_id $python_path/python -u $train_path/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --restore-file $model_path/model2.pt --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 500  --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 16 --fp16  --component-path $output_path  --departure-args-path edin kakao toho --save-dir $main_path/${temp_file}2 $processed_path
# wait
# CUDA_VISIBLE_DEVICES=$device_id $python_path/python -u $train_path/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --restore-file $model_path/model3.pt --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 500  --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 16 --fp16  --component-path $output_path  --departure-args-path edin kakao toho --save-dir $main_path/${temp_file}3 $processed_path
# wait
# CUDA_VISIBLE_DEVICES=$device_id $python_path/python -u $train_path/train.py --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --restore-file $model_path/model4.pt --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0002 --min-lr 1e-09 --clip-norm 5 --weight-decay 0.0 --warmup-updates 8000 --max-tokens 4500 --update-freq 10 --max-sentences-valid 16 --save-interval-updates 500  --no-progress-bar --seed 0 --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 --criterion label_smooth_ce_rl --source-token-dropout 0.2 --target-token-dropout 0.1 --label-smoothing 0.1 --raw-text --max-epoch 16 --fp16  --component-path $output_path  --departure-args-path edin kakao toho --save-dir $main_path/${temp_file}4 $processed_path
# wait

mkdir $main_path/${checkpoint_file}/$data_name
wait
mv $main_path/${temp_file}1/checkpoint12.pt  $main_path/${checkpoint_file}/$data_name/
mv $main_path/${temp_file}2/checkpoint14.pt  $main_path/${checkpoint_file}/$data_name/
mv $main_path/${temp_file}3/checkpoint13.pt  $main_path/${checkpoint_file}/$data_name/
mv $main_path/${temp_file}4/checkpoint15.pt  $main_path/${checkpoint_file}/$data_name/
# mv $main_path/${temp_file}1/checkpoint11.pt  $main_path/${checkpoint_file}/$data_name/
# mv $main_path/${temp_file}2/checkpoint13.pt  $main_path/${checkpoint_file}/$data_name/
# mv $main_path/${temp_file}3/checkpoint12.pt  $main_path/${checkpoint_file}/$data_name/
# mv $main_path/${temp_file}4/checkpoint14.pt  $main_path/${checkpoint_file}/$data_name/
# # mv $main_path/conllu_online_temp1/checkpoint12.pt  $main_path/${checkpoint_file}/$data_name/
# # mv $main_path/conllu_online_temp2/checkpoint14.pt  $main_path/${checkpoint_file}/$data_name/
# # mv $main_path/conllu_online_temp3/checkpoint13.pt  $main_path/${checkpoint_file}/$data_name/
# # mv $main_path/conllu_online_temp4/checkpoint15.pt  $main_path/${checkpoint_file}/$data_name/
wait


mkdir $output_path/$data_name
wait
bash $main_path/predict/predict_wj_rerank.sh  $main_path/${checkpoint_file}/$data_name  wilocABCN-dev
wait
mv /home/projects/11001764/wenjuan/gec_distillation/weiqi_f/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/wilocABCN-dev.out.txt $output_path/$data_name/dev.txt
wait
bash $main_path/predict/predict_wj_rerank.sh $main_path/${checkpoint_file}/$data_name  wilocABCN-test
wait
mv /home/projects/11001764/wenjuan/gec_distillation/weiqi_f/ensemble/reranking+bert+numpunct.wilocABCN-dev.errant/wilocABCN-test.out.txt $output_path/$data_name/test.txt
wait
