import re
import string
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed,  BertTokenizer, BertForSequenceClassification
import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
import json
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import gensim

# PAS (Persona Appropriateness Score) lives in utility/pas.py (NumPy-only, usable on its own).
from utility.pas import (
    _kldiv, js_dist, _to_prob,
    get_topic, get_query_labeled_persona,
    persona_appropriateness_score, pas_from_record, is_persona_relevant, corpus_pas,
)


def p3_score_topic_level(query_labeled_topic, response_labeled_topic, agent_persona_topic_list, response=None):
    """Deprecated alias of ``persona_appropriateness_score`` (PAS), kept for backward compatibility."""
    return persona_appropriateness_score(query_labeled_topic, response_labeled_topic, agent_persona_topic_list)



def hellinger_distance(p, q):

    p = np.array(p)
    q = np.array(q)
    
    
    return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))


def _wasserstein_1d(p, q, step=1.0):

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    wasserstein_val = float(step * np.sum(np.abs(cdf_p - cdf_q)))

    return wasserstein_val

def _tvd(P, Q):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return 0.5 * np.sum(np.abs(P - Q))


def normalize_text(text):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(prediction, attribute):

    pred_tokens = prediction.split()
    truth_tokens = attribute.split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def compute_p_f1(pred, persona_list):
    norm_pred = normalize_text(pred)

    if persona_list is not None:
        f1 = 0
        for p in persona_list:
            att = normalize_text(p.strip())
            _f1 = compute_f1(norm_pred, att)
            f1 += _f1
        f1 = f1/len(persona_list)
    else:
        'persona list empty error'
        f1=None

    return f1


def get_p_distance(predictions, persona, word2vec_model):

    # nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    if word2vec_model == None:
    
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            "./GoogleNews-vectors-negative300.bin",
            binary=True
        )

    def tokenize(text: str):
        text = text.lower()
        # keep letters/numbers/apostrophes, remove other punctuation
        text = re.sub(r"[^a-z0-9'\s]+", " ", text)
        toks = [t for t in text.split() if t and t not in stop_words]
        return toks

    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        denom = (np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0:
            return 0.0
        return float(np.dot(u, v) / denom)

    per_example_scores = []
    per_example_keyword_max = []

    for pred, per in zip(predictions, persona):
        # persona is a list of sentences; join to one text
        persona_text = " ".join(per)

        p_tokens = tokenize(persona_text)
        r_tokens = tokenize(pred)

        # unique (optional but usually sensible for "keywords")
        p_tokens = list(dict.fromkeys(p_tokens))
        r_tokens = list(dict.fromkeys(r_tokens))

        # keep only in-vocab tokens
        p_in = [t for t in p_tokens if t in word2vec_model]
        r_in = [t for t in r_tokens if t in word2vec_model]

        # if nothing to compare, define score as 0
        if len(p_in) == 0 or len(r_in) == 0:
            per_example_scores.append(0.0)
            per_example_keyword_max.append({})
            continue

        # precompute response vectors for speed
        r_vecs = np.vstack([word2vec_model[t] for t in r_in])
        r_norms = np.linalg.norm(r_vecs, axis=1) + 1e-12

        keyword_max = {}
        max_vals = []

        for p in p_in:
            p_vec = word2vec_model[p]
            p_norm = np.linalg.norm(p_vec) + 1e-12

            # cosine with all response tokens, take max
            sims = (r_vecs @ p_vec) / (r_norms * p_norm)
            mx = float(np.max(sims))

            keyword_max[p] = mx
            max_vals.append(mx)

        score = float(np.mean(max_vals)) if max_vals else 0.0
        per_example_scores.append(score)
        per_example_keyword_max.append(keyword_max)

    overall_mean = float(np.mean(per_example_scores)) if per_example_scores else 0.0
    return overall_mean



def get_c_score(predictions, persona, model=None, tokenizer=None):
    nli_model_dir = "./nli/model"

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(nli_model_dir)

    if model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(nli_model_dir, num_labels=3).to(device)

    mean_nli_value = 0.0

    for idx, pred in tqdm(enumerate(predictions), total=len(predictions)):
        persona_list = persona[idx]
        nli_value = get_nli_value(response=pred, persona_list=persona_list, model=model, tokenizer=tokenizer)
        mean_nli_value += nli_value / len(predictions)

    return mean_nli_value

def get_nli_value(response, persona_list, model, tokenizer):
    label_map = {"entail": 0, "neutral": 1, "negative": 2}
    nli_value = 0
    premises = [response] * len(persona_list)
    hypotheses = persona_list
    encodings = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids = encodings['input_ids'], attention_mask = encodings['attention_mask'])
        preds = outputs.logits.argmax(dim=1).cpu().tolist() 
        
        for idx, pred in enumerate(preds):
            if pred == 0:
                nli_value += 1
            elif pred == 1:
                nli_value += 0
            else:
                nli_value += -1

    return nli_value

def compute_bleu1(predictions, references):
    smoothie = SmoothingFunction().method1
    tokenized_hypotheses = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references] 
    return corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothie)

def ovs_eval(tagged_u, tagged_pred):
    eps = 0.1
    hyper_k = 1.0

    u_count = Counter(tagged_u)
    pred_count = Counter(tagged_pred)

    ovs_off_topic_list = []
    ovs_excess_of_quantity_list = []

    ovs_list = []
    ovs_type_list = []
    for pc in pred_count:
        if pc in u_count.keys():
            len_u = u_count[pc]
            len_pred = pred_count[pc]
        else:
            len_u = 0
            len_pred = pred_count[pc]

        penalty_w, w_type = _ovs_penalty(len_u, len_pred)

        ovs = (len_pred / (eps + len_u)) * penalty_w
        ovs_list.append(ovs)
        ovs_type_list.append(w_type)
        if w_type == 'off_topic':
            ovs_off_topic_list.append((len_pred / (eps + len_u)))
        elif w_type == 'qty_exc':
            ovs_excess_of_quantity_list.append((len_pred / (eps + len_u)))

    if len(ovs_list) != 0:
        if len(ovs_off_topic_list) != 0:
            off_topic_value = _sigmoid(hyper_k * math.log(sum(ovs_off_topic_list)/len(ovs_off_topic_list)))
        else:
            off_topic_value = 0.0

        if len(ovs_excess_of_quantity_list) != 0:
            excess_of_quaqntity_value = _sigmoid(hyper_k * math.log(sum(ovs_excess_of_quantity_list)/len(ovs_excess_of_quantity_list)))
        else:
            excess_of_quaqntity_value = 0.0
    else:
        off_topic_value = 0.0
        excess_of_quaqntity_value = 0.0

    # both penalty
    if "off_topic" in ovs_type_list and "qty_exc" in ovs_type_list:
        k = hyper_k
    else:
        k = 1

    if len(ovs_list) == 0:
        OVS = 0.0
    else:
        OVS = _sigmoid(k * math.log(sum(ovs_list)/len(ovs_list)))

    # print(f"ovs: {OVS}")

    return OVS, off_topic_value, excess_of_quaqntity_value, ovs_type_list

def _ovs_penalty(len_u, len_pred):
    diff = (len_pred - len_u) + 1
    # off-topic
    if len_u == 0 and len_u < len_pred:
        w = math.exp(diff)
        w = math.log(w)
        typ = "off_topic"
    # excessive-quantity
    elif len_u != 0 and len_u < len_pred:
        w = diff * math.exp(1)
        w = math.log(w)
        typ = "qty_exc"
    else:  # not overuse
        w = math.exp(0)
        typ = "accordant"

    return w, typ


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))



def pad_prompt_to_match_token_length(tokenizer, render_fn, target_len, max_pad_chars=2000):
    # render_fn(num_pad_spaces)가 만든 텍스트의 토큰 길이가 target_len 이상이 될 때까지
    # 맨 앞의 공백 개수를 한 글자씩 늘려가며 재생성한다.
    num_pad = 0
    text = render_fn(num_pad)
    current_len = len(tokenizer(text)["input_ids"])

    while current_len < target_len and num_pad < max_pad_chars:
        num_pad += 1
        text = render_fn(num_pad)
        current_len = len(tokenizer(text)["input_ids"])

    return text


def split_paragraphs_mbti(text: str):
    sentences = []

    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue

        parts = re.split(r'(?<!\d)\.(\s+|$)', block)

        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)

    return sentences



def parse_to_list(s):
    result = []
    tokens = re.split(r"[,\s]+", s.strip())
    parsing_error = False

    for t in tokens:
        if not t:  
            continue
        try:
            result.append(int(t))   
        except ValueError:
            # if t.lower() == "none":
            #     continue
            # elif t.lower() == 'none.':
            #     continue

            if 'none' in t.lower():
                continue

            else:
                # print("parsing error")
                # if s == '6. Relationships(family)':
                #     return [6], False
                parsing_error = True
    return result, parsing_error
