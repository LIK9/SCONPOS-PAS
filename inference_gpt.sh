# Requires OPENAI_API_KEY to be set in the environment, e.g.:
#   export OPENAI_API_KEY=sk-...
export CUDA_VISIBLE_DEVICES=0

python gpt_inference.py \
  --api_key "${OPENAI_API_KEY:?set OPENAI_API_KEY before running this script}" \
  --test_data sampled_splited \
  --dataset MBTI_S2Conv \
  --model_name gpt4.1 \
  --test_task inference \
  --output_path self/default/all_persona_input \
