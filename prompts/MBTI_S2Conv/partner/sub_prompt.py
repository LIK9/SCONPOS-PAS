import random
from utility.utils import *


def all_persona_input_vanila(example, tokenizer): 
    self_persona_text = "\n".join(
        [f"{i+1}. {p.strip()}" for i, p in enumerate(example['agent_persona'])]
    )
    user_persona_text = "\n".join(
        [f"{i + len(example['agent_persona']) + 1}. {p.strip()}" for i, p in enumerate(example['user_persona'])]
    )

    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{self_persona_text}\n\n"

        "- Partner's persona:\n"
        f"{user_persona_text}\n\n"

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

    user_labels = example["partner_labels"]
    user_persona = example["user_persona"]

    agent_query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)
    
    user_query_labeled_persona = get_query_labeled_persona(query_labeled_topic, user_persona, user_labels)


    if (len(agent_query_labeled_persona) + len(user_query_labeled_persona)) != 0:
        self_persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(agent_query_labeled_persona)])

        user_persona_text = "\n".join([f"{i+1+len(agent_query_labeled_persona)}. {p.strip()}" for i, p in enumerate(user_query_labeled_persona)])

    else:
        self_persona_text = "None"
        user_persona_text = "None"


    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{self_persona_text}\n\n"

        "- Partner's persona:\n"
        f"{user_persona_text}\n\n"

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

    user_labels = example["partner_labels"]
    user_persona = example["user_persona"]

    agent_query_labeled_persona = get_query_labeled_persona(query_labeled_topic, agent_persona, self_labels)
    agent_not_query_labeled_persona = [p for p in agent_persona if p not in agent_query_labeled_persona]


    user_query_labeled_persona = get_query_labeled_persona(query_labeled_topic, user_persona, user_labels)
    user_not_query_labeled_persona = [p for p in user_persona if p not in user_query_labeled_persona]



    if (len(agent_not_query_labeled_persona)+len(user_not_query_labeled_persona)) != 0: # all of agent persona in query_labeled_persona
        not_query_labeled_persona_dict = {
            "agent": agent_not_query_labeled_persona,
            "user": user_not_query_labeled_persona,
        }
        random_role, random_persona = random.choice([
            (role, persona)
            for role, personas in not_query_labeled_persona_dict.items()
            for persona in personas
        ])        

        if random_role == 'agent':
            new_agent_persona = agent_query_labeled_persona + [random_persona]

            self_persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(new_agent_persona)])
            user_persona_text = "\n".join([f"{i+1+len(new_agent_persona)}. {p.strip()}" for i, p in enumerate(user_query_labeled_persona)])
        else:
            new_user_persona = user_query_labeled_persona + [random_persona]

            self_persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(agent_query_labeled_persona)])
            user_persona_text = "\n".join([f"{i+1+len(agent_query_labeled_persona)}. {p.strip()}" for i, p in enumerate(new_user_persona)])

    else:
        self_persona_text = "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(agent_query_labeled_persona)])
        user_persona_text = "\n".join([f"{i+1+len(agent_query_labeled_persona)}. {p.strip()}" for i, p in enumerate(user_query_labeled_persona)])



    dialogue_history = example["dialogues"][:-2] 
    dialogue_text = ""
    for i, turn in enumerate(dialogue_history):
        speaker = "Partner" if i % 2 == 0 else "You"
        dialogue_text += f"{speaker}: {turn.strip()}\n" 

    system_prompt = (
        "- Your persona:\n"
        f"{self_persona_text}\n\n"

        "- Partner's persona:\n"
        f"{user_persona_text}\n\n"

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

        "- Partner's persona:\n"
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



