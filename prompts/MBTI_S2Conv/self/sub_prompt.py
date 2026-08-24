import random
from utility.utils import *


def all_persona_input_vanila(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2]


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]

    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example


def all_persona_input_retrieved(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona_retrieved"])])

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2]


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]

    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example



def certain_persona_input_vanila(example, tokenizer):
    query_labeled_topic = example["query_labeled_topic"]
    self_labels = example["self_labels"]

    agent_persona = example["agent_persona"]
    query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)
    

    if len(query_labeled_persona) != 0:
        persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(query_labeled_persona)])

    else:
        persona_text = ""

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]



    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example

def certain_persona_input_vanila_padded(example, tokenizer):
    query_labeled_topic = example["query_labeled_topic"]
    self_labels = example["self_labels"]

    agent_persona = example["agent_persona"]
    query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)

    if len(query_labeled_persona) != 0:
        persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(query_labeled_persona)])
    else:
        persona_text = ""

    dialogue_history = example["dialogues"][:-2]
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n"

    last_utterance = example["dialogues"][-2]

    def render(persona_text_local, num_pad_spaces):
        system_prompt = (
            f"{' ' * num_pad_spaces}"
            "- Your persona:\n"
            f"{persona_text_local}\n\n"

            "- Dialogue history:\n"
            f"{dialogue_text}"

            "---\n"
            "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
            "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
            "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Partner: {last_utterance}"}
        ]

        if tokenizer is None:
            return messages
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if tokenizer is None:
        example['text'] = render(persona_text, 0)
        return example

    # all_persona_input_vanila와 프롬프트 길이(토큰 수)가 같아지도록, 맨 앞에 공백을 채워 넣는다.
    all_persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(agent_persona)])
    target_len = len(tokenizer(render(all_persona_text, 0))["input_ids"])

    example['text'] = pad_prompt_to_match_token_length(
        tokenizer=tokenizer,
        render_fn=lambda num_pad: render(persona_text, num_pad),
        target_len=target_len,
    )

    return example


def certain_persona_input_vanila_persona_matched(example, tokenizer):
    query_labeled_topic = example["query_labeled_topic"]
    self_labels = example["self_labels"]

    agent_persona = example["agent_persona"]
    query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)

    # certain의 persona 개수를 all(agent_persona)의 개수와 맞춘다.
    # 부족한 개수는 certain 자신의 persona(query_labeled_persona) 안에서 중복 샘플링으로 채운다.
    matched_persona = list(query_labeled_persona)
    num_missing = len(agent_persona) - len(matched_persona)
    if len(matched_persona) != 0 and num_missing > 0:
        matched_persona += random.choices(query_labeled_persona, k=num_missing)

    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(matched_persona)])

    dialogue_history = example["dialogues"][:-2]
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n"

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )

    last_utterance = example["dialogues"][-2]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]

    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example

def certain_persona_input_vanila_neg(example, tokenizer):
    query_labeled_topic = example["query_labeled_topic"]
    self_labels = example["self_labels"]

    agent_persona = example["agent_persona"]
    query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)

    not_query_labeled_persona = [p for p in agent_persona if p not in query_labeled_persona]

    num_neg = len(query_labeled_persona)

    if num_neg != 0 and len(not_query_labeled_persona) != 0:
        # query_labeled_persona의 여집합 중에서, query_labeled_persona와 동일한 개수만큼 무작위로 뽑는다
        sampled_neg_persona = random.sample(not_query_labeled_persona, min(num_neg, len(not_query_labeled_persona)))
        persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(sampled_neg_persona)])

    else:
        persona_text = ""

    dialogue_history = example["dialogues"][:-2]
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n"

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2]


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]



    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example

def one_random_persona_input_vanila(example, tokenizer):
    query_labeled_topic = example["query_labeled_topic"]
    self_labels = example["self_labels"]

    agent_persona = example["agent_persona"]
    query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)

    not_query_labeled_persona = [p for p in agent_persona if p not in query_labeled_persona]

    if len(not_query_labeled_persona) != 0: # all of agent persona in query_labeled_persona
        random_one_persona = random.choices(not_query_labeled_persona, k=1)
        persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(query_labeled_persona + random_one_persona)])

    else:
        persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(query_labeled_persona)])



    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]



    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example


def no_persona_input_vanila(example, tokenizer): 
    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]



    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example



def empty(example, tokenizer): 
    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = "" 

    system_prompt = (
        "- Your persona:\n"
        f"\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"}
    ]



    if tokenizer != None:
        example['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        example['text'] = messages

    return example


def all_persona_input_train(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    matched_response = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"},
        {"role": "assistant", "content": matched_response}

    ]

    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    text = tokenizer.apply_chat_template(messages,tokenize=False)

    cut_messages = messages[:-1]
    prompt_tokenized = tokenizer.apply_chat_template(cut_messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    prompt_len = len(prompt_tokenized)

    labels = [-100]*prompt_len + tokenized[prompt_len:]
    example['input_ids'] = tokenized
    example['text'] = text
    example['labels'] = labels

    return example




def all_persona_input_train_GT(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    matched_response = example["dialogues"][-1] 

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n"
    )


    last_utterance = example["dialogues"][-2] 


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Partner: {last_utterance}"},
        {"role": "assistant", "content": matched_response}

    ]

    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    text = tokenizer.apply_chat_template(messages,tokenize=False)

    cut_messages = messages[:-1]
    prompt_tokenized = tokenizer.apply_chat_template(cut_messages, return_tensors="pt", tokenize=True).squeeze(0).tolist()
    prompt_len = len(prompt_tokenized)

    labels = [-100]*prompt_len + tokenized[prompt_len:]
    example['input_ids'] = tokenized
    example['text'] = text
    example['labels'] = labels

    return example
