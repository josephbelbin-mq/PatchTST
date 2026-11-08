seq_len=96
#model_name=PatchTST
model_name=DLinear

root_path_name=/home/josbel/uni_projects/itransformer/dataset_generation/
data_path_name=space_weather_clean.csv

model_id_name=SpaceWeather

data_name=custom
random_seed=2021

for pred_len in 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --target sunspot_number \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 20 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 10 \
      --patience 3 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.001 \
      > logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
done
