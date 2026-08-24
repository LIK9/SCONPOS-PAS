export CUDA_VISIBLE_DEVICES=1

python main_test.py \
  --inference_batch 4 \
  --do_sample 0 \
  --num_beam 1 \
  --temperature 1.0 \
  --top_p 1.0 \
  --test_data sampled_splited \
  --test_task eval_ovs \
  --dataset None \
  --model_name None \
  --decoding_strategy steering \
  --use_vector 1 \
  --target_layer_ratio 9999999999 \
  --normalize_vector 0 \
  --token_position last \
  --differnce_method mean \
  --output_path self/ours_matched \
  --contrastive_alpha 9999999999 \
  --token_position_steering all \
