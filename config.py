import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--train_data', type=str, default="./original_dataset/train_self_original_duplicate_remove.json")
    parser.add_argument('--valid_data', type=str, default="./original_dataset/valid_self_original.json")
    parser.add_argument('--model_name', type=str, default="llama3")


    # inference
    parser.add_argument('--inference_batch', type=int, default=32)
    parser.add_argument('--num_beam', type=int, default=1)
    parser.add_argument('--do_sample', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0)

    parser.add_argument('--output_path', type=str, default="self/default/all_persona_input")
    parser.add_argument('--test_data', type=str, default="./dataset/processed/persona_labeled/panda_sample.json")
    parser.add_argument('--test_task', type=str, default="eval_ovs")
    
    # processing
    parser.add_argument('--process_task', type=str, default="None")
    parser.add_argument('--persona_labeling_target', type=str, default="generated_response")
    parser.add_argument('--api_key', type=str, default="None")
    parser.add_argument('--persona_labeling_order', type=int, default=1)
    parser.add_argument('--test_main_sub', type=str, default="main")
    parser.add_argument('--dataset', type=str, default="PersonaChat")

    parser.add_argument('--train_data_ratio', type=float, default="1.0")


    # steering vector
    parser.add_argument('--target_layer_ratio', type=float, default=0.5)
    # parser.add_argument('--target_layer_ratio', type=str, default="None")
    parser.add_argument('--normalize_vector', type=int, default=1)
    parser.add_argument('--token_position', type=str, default="last")
    parser.add_argument('--differnce_method', type=str, default="mean")

    # decoding
    parser.add_argument('--use_vector', type=int, default=1)
    parser.add_argument('--decoding_strategy', type=str, default="contrastive")
    parser.add_argument('--contrastive_alpha', type=float, default=2.0)
    parser.add_argument('--token_position_steering', type=str, default="prompt_all") # prompt_all, prompt_last, gen_all


    return parser.parse_args()