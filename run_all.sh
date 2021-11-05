

CUDA_VISIBLE_DEVICES=3 python3 cli.py \
--method pet \
--pattern_ids 0 \
--data_dir FewGLUE/BoolQ \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name boolq \
--output_dir output_dir \
--do_train \
--do_eval \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 2 \
--pet_gradient_accumulation_steps 8 \
--pet_max_steps 250 \
--pet_max_seq_length 256 \
--pet_repetitions 3 \
--sc_per_gpu_train_batch_size 2 \
--sc_per_gpu_unlabeled_batch_size 2 \
--sc_gradient_accumulation_steps 8 \
--sc_max_steps 5000 \
--sc_max_seq_length 256 \
--sc_repetitions 1

