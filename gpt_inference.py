import json
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed, BertTokenizer, BertForSequenceClassification
import torch
import config
import json
from tqdm import tqdm
import copy
import os
import math
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from evaluate import load
from openai import OpenAI
import time
import re



import prompts.PersonaChat.self.sub_prompt as sub_personachat
import prompts.PersonaChat.self.main_prompt as main_personachat


import prompts.MBTI_S2Conv.self.sub_prompt as sub_mbtis2conv
import prompts.MBTI_S2Conv.self.main_prompt as main_mbtis2conv


import prompts.MBTI_S2Conv.partner.sub_prompt as sub_mbtis2conv_partner

import prompts.PersonaChat.partner.sub_prompt as sub_personachat_partner


FUNC_MAP_PERSONACHAT = {
    "default/all_persona_input": sub_personachat.all_persona_input_vanila,
    "default/no_persona_input": sub_personachat.no_persona_input_vanila,
    "default/certain_persona_input": sub_personachat.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_personachat.one_random_persona_input_vanila,

    "cot/rationale": main_personachat.cot_rationale,
    "cot/response": main_personachat.cot_response,
    "decom/rationale": main_personachat.decom_rationale,
    "decom/response": main_personachat.decom_response,
    "refine/response": main_personachat.refine_response,

}


FUNC_MAP_MBTIS2CONV = {
    "default/all_persona_input": sub_mbtis2conv.all_persona_input_vanila,
    "default/no_persona_input": sub_mbtis2conv.no_persona_input_vanila,
    "default/certain_persona_input": sub_mbtis2conv.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_mbtis2conv.one_random_persona_input_vanila,

    "cot/rationale": main_mbtis2conv.cot_rationale,
    "cot/response": main_mbtis2conv.cot_response,
    "decom/rationale": main_mbtis2conv.decom_rationale,
    "decom/response": main_mbtis2conv.decom_response,
    "refine/response": main_mbtis2conv.refine_response,

}


FUNC_MAP_MBTIS2CONV_PARTNER = {
    "default/all_persona_input": sub_mbtis2conv_partner.all_persona_input_vanila,
    "default/no_persona_input": sub_mbtis2conv_partner.no_persona_input_vanila,
    "default/certain_persona_input": sub_mbtis2conv_partner.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_mbtis2conv_partner.one_random_persona_input_vanila,

}


FUNC_MAP_PERSONACHAT_PARTNER = {
    "default/all_persona_input": sub_personachat_partner.all_persona_input_vanila,
    "default/no_persona_input": sub_personachat_partner.no_persona_input_vanila,
    "default/certain_persona_input": sub_personachat_partner.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_personachat_partner.one_random_persona_input_vanila,

}


def build_json_for_batch_api(args):
    self_or_partner = args.output_path.split('/')[0]

    
    if 'ours' in args.output_path or 'rge' in args.output_path:
        test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/rationale.json"

    elif 'decom/response' in args.output_path or 'refine/response' in args.output_path or 'cot/response' in args.output_path:
        test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale.json"

    elif 'sng/response_v' in args.output_path:
        version = args.output_path.split('/')[-1].replace('response_', '')
        test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale_{version}.json"

    elif 'retrieved' in args.output_path:
        threshold = args.output_path.split("/")[-1]
        test_data = f"./dataset/{args.dataset}/processed/persona_labeled/retrieved_{threshold}.json"

    else:
        test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}.json"


    print(f"test data: {test_data}")
    with open(test_data, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)

    max_token = 1000

    model_name = None
    if args.model_name == 'gpt3.5':
        model_name = "gpt-3.5-turbo-1106"
    elif args.model_name == 'gpt4.1':
        model_name = "gpt-4.1-2025-04-14"

    batch_requests = []
    request_idx = 0

    inner_batch_size = 10000

    input_fn_name = args.output_path.split("/", 1)[1]

    if self_or_partner == 'self':

        if args.dataset == 'PersonaChat':
            input_fn = FUNC_MAP_PERSONACHAT[input_fn_name]

        elif args.dataset == 'MBTI_S2Conv':
            input_fn = FUNC_MAP_MBTIS2CONV[input_fn_name]

    elif self_or_partner == 'partner':

        if args.dataset == 'PersonaChat':
            input_fn = FUNC_MAP_PERSONACHAT_PARTNER[input_fn_name]

        elif args.dataset == 'MBTI_S2Conv':
            input_fn = FUNC_MAP_MBTIS2CONV_PARTNER[input_fn_name]


    print(f"prompt function: {input_fn_name}")

    for persona_dict in original_panda_list:

        request_idx += 1

        if 'retrieved' not in args.output_path:
            message = input_fn(persona_dict, None)
        else:
            message = input_fn(persona_dict, None, None, None)


        system_prompt = message['text'][0]['content']
        user_prompt = message['text'][1]['content']


        request_obj = {
            "custom_id": f"request-{request_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{model_name}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_token
            }
        }
        batch_requests.append(request_obj)

        # break

    print(system_prompt)
    print(user_prompt)

    num_batches = math.ceil(request_idx / inner_batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * inner_batch_size
        end = min((batch_idx + 1) * inner_batch_size, request_idx)
        batch_requests_inner_batch = batch_requests[start:end]

        save_path = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}_before.jsonl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            for req in batch_requests_inner_batch:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")


def api_call(args):
    api_key = args.api_key


    client = OpenAI(api_key=api_key)


    batch_idx = 0

    batch_file = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}_before.jsonl"



    batch_input_file = client.files.create(
    file=open(batch_file, "rb"),
    purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "persona labeling"}
    )
    batch_id = batch_job.id


    while True:

        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Current batch status: {status}")

        if status in ["completed", "failed", "expired"]:
            break

        time.sleep(10)

    status = client.batches.retrieve(batch_id)
    stream = client.files.content(status.output_file_id)

    save_path = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}_after.jsonl"


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(stream.read())

def generated_response_processing(args):

    batch_idx = 0
    generated_response_list = []
    after_file = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}_after.jsonl"
    total_id = []

    with open(after_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_ = json.loads(line)  
            response = json_["response"]["body"]["choices"][0]["message"]["content"]

            custom_id = int(re.search(r"request-(\d+)", json_["custom_id"]).group(1))
            total_id.append(custom_id)
            generated_response_list.append(response)

    # if 'ours' in args.output_path:
    #     input_file = f"./result/{args.dataset}/{args.model_name}/main/response/ours/rationale.json"
    # elif 'decom/response' in args.output_path or 'refine/response' in args.output_path or 'cot/response' in args.output_path or 'cot_v02/response' in args.output_path:
    #     input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path.split('/')[0]}/rationale.json"
    # else:
    #     input_file = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}.json"


    self_or_partner = args.output_path.split('/')[0]
    if 'ours' in args.output_path or 'rge' in args.output_path:
        input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/rationale.json"

    elif 'decom/response' in args.output_path or 'refine/response' in args.output_path or 'cot/response' in args.output_path:
        input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale.json"

    elif 'sng/response_v01' in args.output_path:
        input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale_v01.json"

    elif 'sng/response_v02' in args.output_path:
        input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale_v02.json"

    elif 'sng/response_v03' in args.output_path:
        input_file = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale_v03.json"

    else:
        input_file = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}.json"

    print(input_file)
    print(len(total_id))

    with open(input_file, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)
    after_panda_list = copy.deepcopy(original_panda_list)


    save_list = []

    for idx, panda_idx in enumerate(total_id):


        panda_dict = after_panda_list[panda_idx-1]
        generated_response = generated_response_list[idx]


        # if 'refine/response' in args.output_path:

        #     parts = re.split(r'response:', generated_response.replace("\n", ""), flags=re.IGNORECASE)

        #     if parts[-1] == "":
        #         print(generated_response)

        #     if len(parts) != 1 and parts[-1] != "":
        #         generated_response = parts[-1]
        #     else:
        #         generated_response = None

        if generated_response != None:
            panda_dict["generated_response"] = generated_response

            save_list.append(panda_dict)


    output_file = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(len(save_list))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = config.get_args()
    set_seed(args.seed)

    if args.test_task == 'inference':
        build_json_for_batch_api(args)
        api_call(args)
        generated_response_processing(args)
    elif args.test_task == 'inference_processing':
        generated_response_processing(args)