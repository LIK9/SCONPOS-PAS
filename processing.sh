# Requires OPENAI_API_KEY to be set in the environment, e.g.:
#   export OPENAI_API_KEY=sk-...
export CUDA_VISIBLE_DEVICES=0

python processing.py \
  --api_key "${OPENAI_API_KEY:?set OPENAI_API_KEY before running this script}" \
  --test_main_sub main \
  --process_task persona_labeling \
  --test_data sampled_splited \
  --dataset None \
  --persona_labeling_target generated_response \
  --model_name None \
  --output_path None \
