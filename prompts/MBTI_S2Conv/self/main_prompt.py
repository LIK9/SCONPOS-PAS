
def cot_rationale(example, tokenizer): 
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
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, you should generate reasoning path for an appropriate response. Let's think step-by-step.\n"
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

def cot_response(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    rationale = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}\n"


        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, you should generate reasoning path for an appropriate response. Let's think step-by-step.\n\n"

        "- Reasoning path:\n"
        f"{rationale}\n"
        "Respond appropriately referring to the given reasoning path.\n"
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


def decom_rationale(example, tokenizer): 
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
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, you should decompose the given claim into sub-claims for an appropriate response. Let's break down the claim!\n"
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

def decom_response(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    rationale = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}\n"

        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, you should decompose the given claim into sub-claims for an appropriate response. Let's break down the claim!\n\n"

        "- Sub-claims:\n"
        f"{rationale}\n"
        "Respond appropriately referring to the given sub-claims.\n"
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



def refine_response(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    rationale = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}\n"


        "- Your previous response:\n"
        f"{rationale}\n"
        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, please provide feedback on whether your previous response is appropriate.\n"
        "Also, refine your previous response based on this feedback.\n"
        "Please answer in the following format:\n"
        "- Feedback:\n"
        "- Refined response:\n"
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



def refine_response1_v02(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    rationale = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}\n"


        "- Your previous response:\n"
        f"{rationale}\n"
        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, please provide feedback on whether your previous response is appropriate.\n"
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




def refine_response2_v02(example, tokenizer): 
    persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(example["agent_persona"])])
    initial_response = example["initial_response"]
    rationale = example["generated_response"]

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{persona_text}\n\n"

        "- Dialogue history:\n"
        f"{dialogue_text}\n"


        "- Your previous response:\n"
        f"{initial_response}\n"
        "---\n"

        "You aim to chat with partner. The partner has encountered some problems recently, and you will give comfort and help to the partner through chatting.\n"
        "Respond to the partner's utterance considering the given dialogue history between your partner and you.\n"
        "Do not reveal in any way that you are an AI or that you are roleplaying, always remember that you are who you are.\n\n"

        "First, please provide feedback on whether your previous response is appropriate.\n\n"
        "- Feedback:\n"
        f"{rationale}\n"
        "Also, refine your previous response based on this feedback.\n"

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