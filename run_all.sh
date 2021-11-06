
task_name=$1
gpu=$2


# alpha=$2
for alpha in 0.1 0.3 0.5 0.7 1
do
    CUDA_VISIBLE_DEVICES=$gpu python3 cli.py \
    --method pet \
    --pattern_ids 0 1 2 \
    --data_dir FewGLUE/$task_name \
    --model_type albert \
    --model_name_or_path albert-xxlarge-v2 \
    --task_name $task_name \
    --output_dir ./output_dir/${task_name}/alpha_${alpha}/ \
    --do_train \
    --do_eval \
    --pet_per_gpu_eval_batch_size 8 \
    --pet_per_gpu_train_batch_size 8 \
    --pet_gradient_accumulation_steps 2 \
    --pet_max_steps 250 \
    --pet_max_seq_length 256 \
    --pet_repetitions 3 \
    --sc_max_steps 0 \
    --alpha $alpha 

    rm -rf ./output_dir/${task_name}/alpha_${alpha}/p* ./output_dir/${task_name}/alpha_${alpha}/f* ../output_dir/${task_name}/alpha_${alpha}/u*
done

