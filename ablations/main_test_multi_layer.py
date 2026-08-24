import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed,  BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm
import json
import copy
from evaluate import load
import numpy as np
import config
from transformers import TrainingArguments as HFTrainingArguments
from datetime import datetime
from functools import partial
from math import ceil
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from collections import Counter
import re
from utility.utils import *
import time
import torch.nn.functional as F
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, Border, Side

import prompts.PersonaChat.self.sub_prompt as sub_personachat
import prompts.PersonaChat.self.main_prompt as main_personachat


import prompts.MBTI_S2Conv.self.sub_prompt as sub_mbtis2conv
import prompts.MBTI_S2Conv.self.main_prompt as main_mbtis2conv


import prompts.MBTI_S2Conv.partner.sub_prompt as sub_mbtis2conv_partner

import prompts.PersonaChat.partner.sub_prompt as sub_personachat_partner

import gc

FUNC_MAP_PERSONACHAT = {
    "default/all_persona_input": sub_personachat.all_persona_input_vanila,
    "default/no_persona_input": sub_personachat.no_persona_input_vanila,
    "default/certain_persona_input": sub_personachat.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_personachat.one_random_persona_input_vanila,
    "default/empty": sub_personachat.empty,

    "cot/rationale": main_personachat.cot_rationale,
    "cot/response": main_personachat.cot_response,
    "decom/rationale": main_personachat.decom_rationale,
    "decom/response": main_personachat.decom_response,
    "refine/rationale": sub_personachat.all_persona_input_vanila,
    "refine/response": main_personachat.refine_response,

    "ours": sub_personachat.all_persona_input_vanila,

    "ours/cot/rationale": main_personachat.cot_rationale,
    "ours/cot/response": main_personachat.cot_response,
    "ours/decom/rationale": main_personachat.decom_rationale,
    "ours/decom/response": main_personachat.decom_response,
    "ours/refine/rationale": sub_personachat.all_persona_input_vanila,
    "ours/refine/response": main_personachat.refine_response,

    "refine/response1": main_personachat.refine_response1_v02,
    "refine/response2": main_personachat.refine_response2_v02,

    "ours/refine/response1": main_personachat.refine_response1_v02,
    "ours/refine/response2": main_personachat.refine_response2_v02,

    "ours/diff_dataset/cot/rationale": main_personachat.cot_rationale,
    "ours/diff_dataset/cot/response": main_personachat.cot_response,
    "ours/diff_dataset": sub_personachat.all_persona_input_vanila,

}


FUNC_MAP_MBTIS2CONV = {
    "default/all_persona_input": sub_mbtis2conv.all_persona_input_vanila,
    "default/no_persona_input": sub_mbtis2conv.no_persona_input_vanila,
    "default/certain_persona_input": sub_mbtis2conv.certain_persona_input_vanila,
    "default/one_random_persona_input": sub_mbtis2conv.one_random_persona_input_vanila,
    "default/empty": sub_mbtis2conv.empty,


    "cot/rationale": main_mbtis2conv.cot_rationale,
    "cot/response": main_mbtis2conv.cot_response,
    "decom/rationale": main_mbtis2conv.decom_rationale,
    "decom/response": main_mbtis2conv.decom_response,
    "refine/rationale": sub_mbtis2conv.all_persona_input_vanila,
    "refine/response": main_mbtis2conv.refine_response,

    "ours": sub_mbtis2conv.all_persona_input_vanila,

    "ours/cot/rationale": main_mbtis2conv.cot_rationale,
    "ours/cot/response": main_mbtis2conv.cot_response,
    "ours/decom/rationale": main_mbtis2conv.decom_rationale,
    "ours/decom/response": main_mbtis2conv.decom_response,
    "ours/refine/rationale": sub_mbtis2conv.all_persona_input_vanila,
    "ours/refine/response": main_mbtis2conv.refine_response,

    "refine/response1": main_mbtis2conv.refine_response1_v02,
    "refine/response2": main_mbtis2conv.refine_response2_v02,
    
    "ours/refine/response1": main_mbtis2conv.refine_response1_v02,
    "ours/refine/response2": main_mbtis2conv.refine_response2_v02,

    "ours/diff_dataset/cot/rationale": main_mbtis2conv.cot_rationale,
    "ours/diff_dataset/cot/response": main_mbtis2conv.cot_response,
    "ours/diff_dataset": sub_mbtis2conv.all_persona_input_vanila,

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


def get_batch_output(model, tokenizer, batch_messages):
    enc = tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=args.do_sample,
            num_beams=args.num_beam,
            pad_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p
        )

    return outputs, input_ids

def get_model_name(args):
    if args.model_name == 'llama3':
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif args.model_name == 'deepseek':
        model_name = 'deepseek-ai/deepseek-llm-7b-chat'
    elif args.model_name == 'qwen':
        model_name = 'Qwen/Qwen2.5-7B-Instruct'
    else:
        model_name = 'None'
    return model_name



def get_hidden_states_for_dataset(args, model, tokenizer, texts, batch_size, layer_idx):
    """
    texts: list[str]
    return: mean hidden state vector of shape [hidden_size]
    """
    device = next(model.parameters()).device
    collected = []

    num_batches = ceil(len(texts) / batch_size)

    for batch_idx in tqdm(range(num_batches), desc=f"Collect layer {layer_idx}"):
        batch_messages = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        enc = tokenizer(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )


        hidden = outputs.hidden_states[layer_idx]   # [B, S, H]

        if args.token_position == 'last':
            hidden_vector = hidden[:, -1, :] # [B, H]
            # we use left padding


        collected.append(hidden_vector.detach().float().cpu())


        # break

    all_hidden = torch.cat(collected, dim=0)           # [N, H]


    return all_hidden


def get_all_layer_hidden_states_for_dataset(args, model, tokenizer, texts, batch_size, num_layers):
    """
    texts: list[str]
    return: 각 layer(1~num_layers, embedding 제외)의 마지막 토큰 hidden state, shape [N, num_layers, H]
    layer 개수만큼 반복 forward하지 않고, 배치당 한 번의 forward로 모든 layer의 hidden state를 모은다.
    """
    device = next(model.parameters()).device
    collected = []

    num_batches = ceil(len(texts) / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Collect all layers"):
        batch_messages = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        enc = tokenizer(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states[0]은 embedding 출력이므로 제외하고, 1~num_layers(각 decoder layer 출력)만 사용
        layer_hiddens = []
        for layer_idx in range(1, num_layers + 1):
            hidden = outputs.hidden_states[layer_idx]  # [B, S, H]

            if args.token_position == 'last':
                hidden_vector = hidden[:, -1, :]  # [B, H]
                # we use left padding

            layer_hiddens.append(hidden_vector.detach().float().cpu())

        batch_stacked = torch.stack(layer_hiddens, dim=1)  # [B, num_layers, H]
        collected.append(batch_stacked)

    all_hidden = torch.cat(collected, dim=0)  # [N, num_layers, H]

    return all_hidden


def get_mid3_layer_indices(num_layers):
    """
    중간 layer를 기준으로 앞/뒤 1개씩 포함한 3개 layer의 인덱스를 반환한다.
    hidden_states 기준 1-indexed (get_hidden_states_for_dataset의 layer_idx와 동일한 체계).
    """
    mid = int(num_layers * 0.5)
    return [max(1, mid - 1), mid, min(num_layers, mid + 1)]


def extract_vector(args):
    model_name = get_model_name(args)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token


    test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}_train.json"



    
    dataset = load_dataset("json", data_files={"test": test_data})            

    self_or_partner = args.output_path.split('/')[0]

    if self_or_partner == 'self':
        if args.dataset == 'PersonaChat':
            dataset_overuse = dataset.map(
                partial(sub_personachat.all_persona_input_vanila, tokenizer=tokenizer),
                load_from_cache_file=False
            )
            dataset_normal = dataset.map(
                partial(sub_personachat.certain_persona_input_vanila, tokenizer=tokenizer),
                load_from_cache_file=False
            )
        else:
            dataset_overuse = dataset.map(
                partial(sub_mbtis2conv.all_persona_input_vanila, tokenizer=tokenizer),
                load_from_cache_file=False
            )
            dataset_normal = dataset.map(
                partial(sub_mbtis2conv.certain_persona_input_vanila, tokenizer=tokenizer),
                load_from_cache_file=False
            )

    else:
        raise ValueError(f"Unsupported mode: {self_or_partner}")


    overuse_messages = []
    normal_messages = []

    for ex_idx in range(len(dataset_overuse["test"])):
        overuse_messages.append(dataset_overuse["test"][ex_idx]["text"])
        normal_messages.append(dataset_normal["test"][ex_idx]["text"])

    print("example overuse:", overuse_messages[0])
    print("example normal:", normal_messages[0])
    print("num examples:", len(overuse_messages), len(normal_messages))

    batch_size = args.inference_batch

    if args.target_layer_ratio == 'all':
        target_layer = 'all'
        num_layers = model.config.num_hidden_layers

        all_overuse = get_all_layer_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=overuse_messages,
            batch_size=batch_size,
            num_layers=num_layers,
        )  # [N, num_layers, H]

        all_normal = get_all_layer_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=normal_messages,
            batch_size=batch_size,
            num_layers=num_layers,
        )  # [N, num_layers, H]

        if args.differnce_method == 'mean':
            overuse_vector = all_overuse.mean(dim=0) - all_normal.mean(dim=0)  # [num_layers, H]

    elif args.target_layer_ratio == 'mid3':
        target_layer = 'mid3'
        num_layers = model.config.num_hidden_layers
        mid3_layer_indices = get_mid3_layer_indices(num_layers)  # 1-indexed, 예: [mid-1, mid, mid+1]

        all_overuse_full = get_all_layer_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=overuse_messages,
            batch_size=batch_size,
            num_layers=num_layers,
        )  # [N, num_layers, H]

        all_normal_full = get_all_layer_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=normal_messages,
            batch_size=batch_size,
            num_layers=num_layers,
        )  # [N, num_layers, H]

        # 1-indexed layer 번호 -> all_overuse_full의 배열 위치(0-indexed)로 변환해 3개만 선택
        layer_positions = [idx - 1 for idx in mid3_layer_indices]
        all_overuse = all_overuse_full[:, layer_positions, :]  # [N, 3, H]
        all_normal = all_normal_full[:, layer_positions, :]  # [N, 3, H]

        if args.differnce_method == 'mean':
            overuse_vector = all_overuse.mean(dim=0) - all_normal.mean(dim=0)  # [3, H]

    else:
        target_layer = int(model.config.num_hidden_layers * float(args.target_layer_ratio))

        all_overuse = get_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=overuse_messages,
            batch_size=batch_size,
            layer_idx=target_layer,
        )

        all_normal = get_hidden_states_for_dataset(
            args=args,
            model=model,
            tokenizer=tokenizer,
            texts=normal_messages,
            batch_size=batch_size,
            layer_idx=target_layer,
        )

        if args.differnce_method == 'mean':
            overuse_vector = all_overuse.mean(dim=0) - all_normal.mean(dim=0)  # [H]


    if overuse_vector.dim() == 1:
        raw_norm = overuse_vector.norm().item()

        if args.normalize_vector:
            print("vector normalized")
            overuse_vector = overuse_vector / (raw_norm + 1e-8)

        final_norm = overuse_vector.norm().item()
    else:
        # [num_layers, H] (또는 [3, H]) - layer별로 개별 정규화
        raw_norm = overuse_vector.norm(dim=-1).tolist()

        if args.normalize_vector:
            print("vector normalized (per layer)")
            overuse_vector = overuse_vector / (overuse_vector.norm(dim=-1, keepdim=True) + 1e-8)

        final_norm = overuse_vector.norm(dim=-1).tolist()

    print("vector shape:", overuse_vector.shape)
    print("raw norm:", raw_norm)
    print("final norm:", final_norm)
    

    save_dir = f"./result/{args.dataset}/{args.model_name}/vector"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}.pt"
    )


    torch.save(
        {
            "vector": overuse_vector.detach().cpu(),
            "raw_norm": raw_norm,
            "final_norm": final_norm,
            "normalized": bool(args.normalize_vector),
            "layer": target_layer,
            "model_name": model_name,
            "dataset": args.dataset,
            "source": "overuse - normal",
            "token_position": args.token_position,
            "differnce_method": args.differnce_method,
            "num_examples": len(overuse_messages),



        },
        save_path,
    )

    print(f"saved to: {save_path}")


def load_steering_vector(args, model):
    if args.target_layer_ratio == 'all':
        target_layer = 'all'
    elif args.target_layer_ratio == 'mid3':
        target_layer = 'mid3'
    else:
        target_layer = int(model.config.num_hidden_layers * float(args.target_layer_ratio))

    if 'diff_dataset' in args.output_path:
        if args.dataset == 'PersonaChat':
            vector_path = f"./result/MBTI_S2Conv/{args.model_name}/vector/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}.pt"
        else:
            vector_path = f"./result/PersonaChat/{args.model_name}/vector/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}.pt"
    else:
        vector_path = f"./result/{args.dataset}/{args.model_name}/vector/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}.pt"
    print(f"vector_path: {vector_path}")
    ckpt = torch.load(vector_path, map_location="cpu")

    vector = ckpt["vector"].to(device=next(model.parameters()).device, dtype=model.dtype)
    # target_layer = ckpt["layer"]

    return vector, target_layer



def make_amateur_steering_hook(v, alpha, direction, token_position, is_prefill):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        hidden = hidden.clone()
        steer = direction * alpha * v

        if token_position == 'last':
            hidden[:, -1, :] += steer

        elif token_position == "gen_all":
            # Generated Tokens After Input
            if not is_prefill:
                hidden += steer

        elif token_position == "prompt_all":
            # ALL Input Tokens
            if is_prefill:
                hidden += steer

        elif token_position == "prompt_last":
            # Last Input Token
            if is_prefill:
                hidden[:, -1:, :] += steer

        elif token_position == 'all':
            hidden += steer

        else:
            raise ValueError(f"Unsupported mode: {token_position}")


        if rest is None:
            return hidden
        return (hidden, *rest)

    return hook


def register_steering_hooks(model, v, alpha, direction, token_position, is_prefill, target_layer):
    """
    target_layer가 'all'이면 모든 decoder layer에, 'mid3'면 중간 layer 기준 앞/뒤 1개씩 포함한
    3개 decoder layer에, 그 외에는 target_layer-1번 layer 하나에만 hook을 건다.

    v가 [num_layers_selected, H] 형태(layer별 벡터)면 layer_indices와 같은 순서로 저장된 것으로
    보고 각 layer에 자신의 행(v[i])을 사용하고, v가 [H] 형태(단일 벡터)면 모든 layer에 동일하게 사용한다.

    등록된 handle 목록을 반환하므로, 호출부에서 finally에 전부 remove() 해줘야 한다.
    """
    if target_layer == 'all':
        layer_indices = list(range(model.config.num_hidden_layers))
    elif target_layer == 'mid3':
        # hidden_states 1-indexed 인덱스를 decoder layer 0-indexed 인덱스로 변환
        layer_indices = [idx - 1 for idx in get_mid3_layer_indices(model.config.num_hidden_layers)]
    else:
        layer_indices = [target_layer - 1]

    handles = []
    for i, layer_idx in enumerate(layer_indices):
        v_layer = v[i] if v.dim() > 1 else v

        handle = model.model.layers[layer_idx].register_forward_hook(
            make_amateur_steering_hook(
                v=v_layer,
                alpha=alpha,
                direction=direction,
                token_position=token_position,
                is_prefill=is_prefill,
            )
        )
        handles.append(handle)

    return handles

@torch.no_grad()
def contrastive_generate_batch(
    model,
    tokenizer,
    batch_messages,
    steering_vector,
    target_layer,
    alpha,
    beta,
    max_new_tokens,
    amateur_direction,
    do_sample,
    temperature,
    top_p,
):
    device = next(model.parameters()).device

    enc = tokenizer(
        batch_messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    prompt_input_ids = [seq.tolist() for seq in input_ids]
    generated = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    eos_token_id = tokenizer.eos_token_id
    batch_size = generated.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    v = steering_vector.to(device=device, dtype=model.dtype)

    for _ in range(max_new_tokens):
        # expert
        expert_out = model(
            input_ids=generated,
            attention_mask=generated_attention_mask,
            use_cache=False,
        )
        expert_logits = expert_out.logits[:, -1, :]
        logp_expert = F.log_softmax(expert_logits, dim=-1)

        # amateur
        handle = model.model.layers[target_layer].register_forward_hook(
            make_amateur_steering_hook(
                v=v,
                alpha=alpha,
                direction=amateur_direction,
                token_position=args.token_position,
            )
        )
        try:
            amateur_out = model(
                input_ids=generated,
                attention_mask=generated_attention_mask,
                use_cache=False,
            )
            amateur_logits = amateur_out.logits[:, -1, :]
            logp_amateur = F.log_softmax(amateur_logits, dim=-1)
        finally:
            handle.remove()

        # contrastive score
        scores = logp_expert - beta * logp_amateur

        # temperature
        scores = scores / max(temperature, 1e-6)

        if do_sample:
            # top-p
            if top_p < 1.0:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                probs = F.softmax(sorted_scores, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)

                remove_mask = cumulative_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False

                sorted_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))

                scores_filtered = torch.full_like(scores, float("-inf"))
                scores_filtered.scatter_(1, sorted_indices, sorted_scores)
                scores = scores_filtered

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(scores, dim=-1)

        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, eos_token_id),
            next_tokens,
        )

        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

        new_mask = torch.ones(
            (batch_size, 1),
            dtype=generated_attention_mask.dtype,
            device=device
        )
        generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

        finished = finished | (next_tokens == eos_token_id)
        if finished.all():
            break

    batch_outputs = [generated[i].tolist() for i in range(batch_size)]
    return batch_outputs, prompt_input_ids


@torch.no_grad()
def activation_steering_generate_batch(
    model,
    tokenizer,
    batch_messages,
    steering_vector,
    target_layer,
    alpha,
    max_new_tokens,
    steering_direction,
    do_sample,
    temperature,
    top_p,
):
    device = next(model.parameters()).device

    enc = tokenizer(
        batch_messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    prompt_input_ids = [seq.tolist() for seq in input_ids]
    generated = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    eos_token_id = tokenizer.eos_token_id
    batch_size = generated.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    v = steering_vector.to(device=device, dtype=model.dtype)

    # -----------------------------
    # 1) prefill
    # -----------------------------


    # handle = model.model.layers[target_layer-1].register_forward_hook(
    #     make_amateur_steering_hook(
    #         v=v,
    #         alpha=alpha,
    #         direction=steering_direction,
    #         token_position=args.token_position,
    #     )
    # )

    handles = register_steering_hooks(
        model=model,
        v=v,
        alpha=alpha,
        direction=steering_direction,
        token_position=args.token_position_steering,
        is_prefill=True,
        target_layer=target_layer,
    )

    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values
    finally:
        for handle in handles:
            handle.remove()

    # -----------------------------
    # 2) decoding loop
    # -----------------------------
    for step in range(max_new_tokens):
        scores = logits

        if temperature is not None and temperature > 0:
            scores = scores / max(temperature, 1e-6)

        if do_sample:
            if top_p < 1.0:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_scores, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove_mask = cumulative_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False

                sorted_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))

                filtered_scores = torch.full_like(scores, float("-inf"))
                filtered_scores.scatter_(1, sorted_indices, sorted_scores)
                scores = filtered_scores

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

            denom = probs.sum(dim=-1, keepdim=True)
            zero_mask = denom.squeeze(-1) == 0
            if zero_mask.any():
                probs[zero_mask] = 1.0 / probs.size(-1)
                denom = probs.sum(dim=-1, keepdim=True)

            probs = probs / denom
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(scores, dim=-1)

        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, eos_token_id),
            next_tokens,
        )

        generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

        new_mask = torch.ones(
            (batch_size, 1),
            dtype=generated_attention_mask.dtype,
            device=device,
        )
        generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

        finished = finished | (next_tokens == eos_token_id)
        if finished.all():
            break

        step_input_ids = next_tokens.unsqueeze(-1)

        # handle = model.model.layers[target_layer-1].register_forward_hook(
        #     make_amateur_steering_hook(
        #         v=v,
        #         alpha=alpha,
        #         direction=steering_direction,
        #         token_position=args.token_position,
        #     )
        # )

        handles = register_steering_hooks(
            model=model,
            v=v,
            alpha=alpha,
            direction=steering_direction,
            token_position=args.token_position_steering,
            is_prefill=False,
            target_layer=target_layer,
        )

        try:
            out = model(
                input_ids=step_input_ids,
                attention_mask=generated_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values
        finally:
            for handle in handles:
                handle.remove()

    batch_outputs = [generated[i].tolist() for i in range(batch_size)]
    return batch_outputs, prompt_input_ids



def inference(args):
    model_name = get_model_name(args)


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache=True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side = 'left')
    tokenizer.pad_token = tokenizer.eos_token
    


    self_or_partner = args.output_path.split('/')[0]

    if 'decom/response' in args.output_path or 'refine/response' in args.output_path or 'cot/response' in args.output_path:
        test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/rationale.json"
        if 'refine/response2' in args.output_path:
            test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/{args.output_path.split('/')[1]}/response1.json"

        if 'ours' in args.output_path: 
            test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/{args.output_path.split('/')[2]}/rationale/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"
            if 'refine/response2' in args.output_path:
                test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/{args.output_path.split('/')[2]}/response1/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"
        if 'diff_dataset' in args.output_path:
            test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/diff_dataset/{args.output_path.split('/')[3]}/rationale/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"
            if 'refine/response2' in args.output_path:
                test_data = f"./result/{args.dataset}/{args.model_name}/main/response/{self_or_partner}/ours/diff_dataset/{args.output_path.split('/')[3]}/response1/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"


    elif 'default' in args.output_path:
        test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}.json" # train+test dataset

    elif 'ours' in args.output_path: 
        # test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}_test.json" # test dataset only
        test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}_test.json" # test dataset only

    else:
        test_data = f"./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}_test.json" # test dataset only


    print(f"test data: {test_data}")

    dataset = load_dataset("json", data_files={"test": test_data})            

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

    dataset = dataset.map(partial(input_fn, tokenizer=tokenizer),load_from_cache_file=False)


    output_messages = []
    predictions = []

    with open(test_data, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)
    mapped_original_panda_list = []

    for ex_idx in range(len(dataset["test"])):
        ex = dataset["test"][ex_idx]
        output_messages.append(ex["text"])
        mapped_original_panda_list.append(original_panda_list[ex_idx])


    print(output_messages[2])

    print(len(mapped_original_panda_list))


    batch_size = args.inference_batch

    num_batches = ceil(len(output_messages) / batch_size)
    progress_bar = tqdm(range(num_batches))

    steering_vector = None
    target_layer = None
    if args.use_vector:
        print('use overuse vector')
        steering_vector, target_layer = load_steering_vector(args, model)
    

    for batch_idx in progress_bar:
        batch_messages = output_messages[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        if not args.use_vector and args.decoding_strategy == 'base':
            print('decoding_strategy=base')
            batch_outputs, input_ids = get_batch_output(model, tokenizer, batch_messages)

        elif args.use_vector and args.decoding_strategy == 'steering':
            print('decoding_strategy=steering')
            batch_outputs, input_ids = activation_steering_generate_batch(
                model=model,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                steering_vector=steering_vector,
                target_layer=target_layer,
                alpha=args.contrastive_alpha,   
                max_new_tokens=256,
                steering_direction=-1.0,
                # steering_direction=1.0,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
            )

        for i in range(len(batch_messages)):
            prompt_lens = len(input_ids[i])
            gen_ids = batch_outputs[i][prompt_lens:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predictions.append(pred)
            # print(pred)

        # break

    save_list = []
    for idx in range(len(predictions)):
        if predictions[idx] != None:
            if 'refine/response1' in args.output_path:
                item = mapped_original_panda_list[idx].copy()
                initial_response = item["generated_response"]
                item["initial_response"] = initial_response
                item["generated_response"] = predictions[idx]
                save_list.append(item)
            else:
                mapped_original_panda_list[idx]["generated_response"] = predictions[idx]
                save_list.append(mapped_original_panda_list[idx])

    print(f"save list len: {len(save_list)}")


    if 'ours' not in args.output_path:
        save_path = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}.json"
    else:
        save_path = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)




def get_test_file(args):
    test_file = []

    test_file_certain_persona_output = []
    test_file_no_persona_output = []
    test_file_all = []

    self_or_partner = args.output_path.split('/')[0]

    if args.test_task == 'eval_ovs':
        input_json = f"./result/{args.dataset}/{args.model_name}/main/response_labeled/{args.output_path}/processed/{args.output_path}.json"

    elif args.test_task == 'eval':
        input_json = f"./result/{args.dataset}/{args.model_name}/main/response/{args.output_path}.json"

    output_json_certain_persona_output = f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}/certain_persona_output.json"
    output_json_no_persona_output = f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}/no_persona_output.json"
    output_json_all = f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}.json"

    with open(input_json, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)

    for original_panda_dict in tqdm(original_panda_list, total=len(original_panda_list)):
        test_file_all.append(original_panda_dict)

        if self_or_partner == 'self':
            query_labeled_persona = get_query_labeled_persona(original_panda_dict["query_labeled_topic"], original_panda_dict["agent_persona"], original_panda_dict["self_labels"])
        elif self_or_partner == 'partner':
            query_labeled_persona = get_query_labeled_persona(original_panda_dict["query_labeled_topic"], original_panda_dict["agent_persona"]+original_panda_dict["user_persona"], original_panda_dict["self_labels"]+original_panda_dict["partner_labels"])


        if len(query_labeled_persona) != 0:
            test_file_certain_persona_output.append(original_panda_dict)
            # print(original_panda_dict["dialogues"][-1])
            # print()
        else:
            test_file_no_persona_output.append(original_panda_dict)
    
    dict_certain_persona_output = {'test_file': test_file_certain_persona_output, 'output_json': output_json_certain_persona_output}
    dict_no_persona_output = {'test_file': test_file_no_persona_output, 'output_json': output_json_no_persona_output}
    dict_all_output = {'test_file': test_file_all, 'output_json': output_json_all}

    test_file.append(dict_no_persona_output)
    test_file.append(dict_certain_persona_output)
    test_file.append(dict_all_output)

    return test_file

def compute_metrics(args):
    rouge = load("rouge")
    # bleu = load("bleu")
    bertscore = load("bertscore")
    # meteor = load("meteor")

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=True, padding_side = 'left')
    tokenizer.pad_token = tokenizer.eos_token

    # get_p_distance/get_c_score가 test_file_dict마다 3.6GB word2vec/NLI 모델을 매번 새로 디스크에서
    # 로드하던 것이 compute_metrics()가 느렸던 원인이라, 반복문 밖에서 한 번만 로드해서 재사용한다.
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        "./GoogleNews-vectors-negative300.bin",
        binary=True,
    )

    nli_model_dir = "./nli/model"
    nli_tokenizer = BertTokenizer.from_pretrained(nli_model_dir)
    nli_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nli_model = BertForSequenceClassification.from_pretrained(nli_model_dir, num_labels=3).to(nli_device)

    test_file = get_test_file(args)
    irrel_PAS = None
    rel_PAS = None
    irrel_PAS_both = None
    rel_PAS_both = None

    for test_file_dict in test_file:


        ovs_total = 0
        off_topic_value_total = 0
        excess_of_quaqntity_value_total = 0
        ovs_len = 0


        predictions = []
        references = []    
        agent_persona_list = []
        partner_persona_list = []
        both_persona_list = []

        agent_pas = 0.0
        # partner_pas = 0.0
        both_pas = 0.0
        response_len = 0.0

        output_json = test_file_dict["output_json"]
        original_panda_list = test_file_dict["test_file"]


        print(len(original_panda_list))
        print(output_json)
        print()


        for original_panda_dict in tqdm(original_panda_list, total=len(original_panda_list)):

            agent_persona = original_panda_dict["agent_persona"]
            user_persona = original_panda_dict["user_persona"]
            all_persona = agent_persona+user_persona

            self_labels = original_panda_dict["self_labels"]
            partner_labels = original_panda_dict["partner_labels"]
            generated_response = original_panda_dict["generated_response"]

            if args.test_task == 'eval_ovs' and "generated_response_labeled_persona" in original_panda_dict.keys():
                generated_response_labeled_persona = original_panda_dict["generated_response_labeled_persona"]

                ovs_len += 1
                response_entail_persona = original_panda_dict["generated_response_labeled_persona"]
                query_labeled_topic = original_panda_dict["query_labeled_topic"]
                response_topic_list = get_topic(generated_response_labeled_persona, agent_persona, user_persona, self_labels, partner_labels)
                agent_persona_topic_list = get_topic(agent_persona, agent_persona, user_persona, self_labels, partner_labels)
                partner_persona_topic_list = get_topic(user_persona, agent_persona, user_persona, self_labels, partner_labels)
                both_persona_topic_list = get_topic(all_persona, agent_persona, user_persona, self_labels, partner_labels)

                agent_topic_list = get_topic(response_entail_persona, agent_persona, user_persona, self_labels, partner_labels)

                ovs_val, off_topic_value, excess_of_quaqntity_value, ovs_type_list = ovs_eval(original_panda_dict["query_labeled_topic"], agent_topic_list)
                ovs_total += ovs_val
                off_topic_value_total += off_topic_value
                excess_of_quaqntity_value_total += excess_of_quaqntity_value

                agent_pas += p3_score_topic_level(query_labeled_topic, response_topic_list, agent_persona_topic_list, generated_response)
                # partner_pas += p3_score_topic_level(query_labeled_topic, response_topic_list, partner_persona_topic_list, generated_response)
                both_pas += p3_score_topic_level(query_labeled_topic, response_topic_list, both_persona_topic_list, generated_response)

                response_len += len(tokenizer(generated_response)['input_ids'])

            predictions.append(original_panda_dict["generated_response"])
            references.append(original_panda_dict["dialogues"][-1])
            agent_persona_list.append(agent_persona)
            partner_persona_list.append(user_persona)
            both_persona_list.append(all_persona)

        self_or_partner = args.output_path.split('/')[0]

        if self_or_partner == 'self':
            p_f1 = get_p_f1(predictions, agent_persona_list)
            p_distance = get_p_distance(predictions, agent_persona_list, word2vec_model)

            if args.dataset == 'PersonaChat':
                c_score = get_c_score(predictions, agent_persona_list, model=nli_model, tokenizer=nli_tokenizer)

            else:
                c_score = 0


        else:
            print('partner persona included')
            p_f1 = get_p_f1(predictions, both_persona_list)
            p_distance = get_p_distance(predictions, both_persona_list, word2vec_model)

            if args.dataset == 'PersonaChat':
                c_score = get_c_score(predictions, both_persona_list, model=nli_model, tokenizer=nli_tokenizer)

            else:
                c_score = 0


        bleu1_score = compute_bleu1(predictions, references)
        # bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["bleu"]
        rouge_score = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])["rougeL"]
        # meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]

        bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en", device = 'cuda:0', batch_size = 256)
        bertscore_f1 = np.mean(bertscore_result["f1"])


        if args.test_task == 'eval_ovs':
            ovs_avg = ovs_total / ovs_len
            off_topic_value_avg = off_topic_value_total / ovs_len
            excess_of_quaqntity_value_avg = excess_of_quaqntity_value_total / ovs_len

            agent_pas_avg = agent_pas / ovs_len
            both_pas_avg = both_pas / ovs_len

            if len(test_file) == 3:
                if test_file_dict["output_json"] == f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}/no_persona_output.json":
                    irrel_PAS = agent_pas_avg
                    irrel_PAS_both = both_pas_avg
                elif test_file_dict["output_json"] == f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}/certain_persona_output.json":
                    rel_PAS = agent_pas_avg
                    rel_PAS_both = both_pas_avg
                elif test_file_dict["output_json"] == f"./result/{args.dataset}/{args.model_name}/main/metrics/{args.output_path}.json":
                    agent_pas_avg = 2/(1/irrel_PAS + 1/rel_PAS)
                    both_pas_avg = 2/(1/irrel_PAS_both + 1/rel_PAS_both)


            xlsx_result = [
                ["OVS", "OT", "EoQ", "", "ROUGE-L", "BLEU", "BertScore", "", "P-Dist", "C-Score", "P-F1", "PAS", "PAS(Both)", "", "length"],
                [
                    round(ovs_avg, 4),
                    round(off_topic_value_avg, 4),
                    round(excess_of_quaqntity_value_avg, 4),
                    "",
                    round(rouge_score, 4),
                    round(bleu1_score, 4),
                    round(bertscore_f1, 4),
                    "",
                    round(p_distance, 4),
                    round(c_score, 4) if c_score != "x" else "x",
                    round(p_f1, 4),
                    round(agent_pas_avg, 4),
                    round(both_pas_avg, 4),
                    "",
                    round(response_len / ovs_len, 4),

                ],
            ]

            output_xlsx = output_json.replace(".json", ".xlsx")

            os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)

            wb = Workbook()
            ws = wb.active

            for row in xlsx_result:
                ws.append(row)

            thin = Side(style="thin", color="000000")
            border = Border(top=thin, bottom=thin)

            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border
                    if isinstance(cell.value, float):
                        cell.number_format = '0.0000'
                        
            for cell in ws[1]:
                cell.font = Font(bold=True)

            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 14

            wb.save(output_xlsx)
        
        # break





def get_p_f1(predictions, persona):
    mean_p_f1 = 0.0
    p_f1_list = []
    for idx, pred in enumerate(predictions):
        persona_list = persona[idx]
        p_fl = compute_p_f1(pred, persona_list)
        mean_p_f1 += (p_fl / len(predictions))
        p_f1_list.append(p_fl)
    return mean_p_f1






if __name__ == '__main__':

    args = config.get_args()
    set_seed(args.seed)

    if args.test_task == "extract_vector":
        args.inference_batch = 4

        target_layer_ratio_list = [
                # 'all',
                'mid3'
                              ]
        
        dataset_list = [
            "PersonaChat", 
                        "MBTI_S2Conv",
                        ]
        model_list = [
            'llama3',
            'qwen',
            # 'deepseek',
        ]
        contrastive_alpha_list = [
            # 0.25,
            # 0.5,
            # 0.75,
            # 1.0,
            # 2.0,

            3.0,
            4.0,
            # 5.0,
            # 6.0,
            # 7.0
        ]

        for contrastive_alpha in contrastive_alpha_list:
            args.contrastive_alpha = contrastive_alpha

            for model in model_list:
                args.model_name = model
                    
                for dataset in dataset_list:
                    args.dataset = dataset

                    for target_layer_ratio in target_layer_ratio_list:
                        args.target_layer_ratio = target_layer_ratio

                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(5)


                        # if args.dataset == 'PersonaChat':
                        #     if args.model_name == 'llama3':
                        #         args.contrastive_alpha = 5.0

                        #     elif args.model_name == 'qwen':
                        #         args.contrastive_alpha = 5.0

                        #     elif args.model_name == 'deepseek':
                        #         args.contrastive_alpha = 2.0

                        # else:
                        #     if args.model_name == 'llama3':
                        #         args.contrastive_alpha = 5.0

                        #     elif args.model_name == 'qwen':
                        #         args.contrastive_alpha = 5.0

                        #     elif args.model_name == 'deepseek':
                        #         args.contrastive_alpha = 2.0
                        



                        print(f"inference_batch: {args.inference_batch}")
                        print(f"dataset: {args.dataset}")
                        print(f"model_name: {args.model_name}")
                        print(f"target_layer_ratio: {args.target_layer_ratio}")

                        start_time = time.perf_counter()
                        extract_vector(args)
                        end_time = time.perf_counter()
                        elapsed = end_time - start_time

                        elapsed_path = f"./result/{args.dataset}/{args.model_name}/main/elapsed/self/ours/extract_vector/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"

                        # elapsed_path = f"./result/{args.dataset}/{args.model_name}/elapsed/{args.output_path}.json"
                        os.makedirs(os.path.dirname(elapsed_path), exist_ok=True)
                        with open(elapsed_path, "w", encoding="utf-8") as f:
                            json.dump({'elapsed': round(elapsed, 3)}, f, ensure_ascii=False, indent=4)




                        self_or_partner = args.output_path.split('/')[0]



                        print(f"inference_batch: {args.inference_batch}")
                        print(f"dataset: {args.dataset}")
                        print(f"model_name: {args.model_name}")
                        print(f"target_layer_ratio: {args.target_layer_ratio}")
                        print(f"token_position_steering: {args.token_position_steering}")
                        print(f"contrastive_alpha: {args.contrastive_alpha}")

                        start_time = time.perf_counter()
                        inference(args)
                        end_time = time.perf_counter()
                        elapsed = end_time - start_time

                        if 'ours' not in args.output_path:
                            elapsed_path = f"./result/{args.dataset}/{args.model_name}/main/elapsed/{args.output_path}.json"
                        else:
                            elapsed_path = f"./result/{args.dataset}/{args.model_name}/main/elapsed/self/ours/target_layer_ratio={args.target_layer_ratio},token_position={args.token_position},differnce_method={args.differnce_method},normalize_vector={args.normalize_vector}/use_vector={args.use_vector},decoding_strategy={args.decoding_strategy},contrastive_alpha={args.contrastive_alpha},token_position_steering={args.token_position_steering}.json"

                        # elapsed_path = f"./result/{args.dataset}/{args.model_name}/elapsed/{args.output_path}.json"
                        os.makedirs(os.path.dirname(elapsed_path), exist_ok=True)
                        with open(elapsed_path, "w", encoding="utf-8") as f:
                            json.dump({'elapsed': round(elapsed, 3)}, f, ensure_ascii=False, indent=4)



    if args.test_task == "eval" or args.test_task == "eval_ovs":

        # target_layer_ratio_list = [
        #         # 'all',
        #         'mid3'
        #                         ]
        
        dataset_list = [
            # "PersonaChat", 
                        "MBTI_S2Conv",
                        ]
        model_list = [
            'llama3',
            'qwen',
            'deepseek'
        ]
        # contrastive_alpha_list = [
        #     # 3.0,
        #     # 4.0,
        #     5.0,

        # ]


        for model in model_list:
            args.model_name = model
                
            for dataset in dataset_list:
                args.dataset = dataset


                if args.dataset == 'PersonaChat':
                    if args.model_name == 'llama3':
                        args.contrastive_alpha = 5.0
                        args.target_layer_ratio = 0.5

                    elif args.model_name == 'qwen':
                        args.contrastive_alpha = 5.0
                        args.target_layer_ratio = 0.7

                    elif args.model_name == 'deepseek':
                        args.contrastive_alpha = 2.0
                        args.target_layer_ratio = 0.5

                else:
                    if args.model_name == 'llama3':
                        args.contrastive_alpha = 5.0
                        args.target_layer_ratio = 0.5

                    elif args.model_name == 'qwen':
                        args.contrastive_alpha = 5.0
                        args.target_layer_ratio = 0.5

                    elif args.model_name == 'deepseek':
                        args.contrastive_alpha = 2.0
                        args.target_layer_ratio = 0.5
                
                        
                args.output_path = f'self/ours_matched/target_layer_ratio={args.target_layer_ratio},token_position=last,differnce_method=mean,normalize_vector=0/use_vector=1,decoding_strategy=steering,contrastive_alpha={args.contrastive_alpha},token_position_steering=all'

                compute_metrics(args)