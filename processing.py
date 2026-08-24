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
from utility.utils import *
import random
from pydantic import BaseModel, Field

class PersonaTopicOutput(BaseModel):
    topic_labels: list[int] = Field(
        description="list of topic indexes (e.g. if both topic 0 and topic 2 are labeled in the target text, then output [0, 2])"
    )

    confidence_score: float = Field(
        ge=0,
        le=1,
        description="confidence score between 0 and 1",
    )
    created_topic: str = Field(
        description="only if a new topic was created, otherwise output an empty string",
    )

def process_labels(labels):
    processed_labesl = []

    for label in labels:
        if isinstance(label, list):
            processed_labesl.append(label)      
        else:
            processed_labesl.append([label]) 

    return processed_labesl


def build_json_for_batch_api(args):
    if args.persona_labeling_target == 'generated_response':
        with open(f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response/{args.output_path}.json", 'r', encoding='utf-8') as f:
            original_json = json.load(f)
    
    elif args.persona_labeling_target == 'persona_topic':
        with open(f"./dataset/{args.dataset}/processed/{args.test_data}.json", 'r', encoding='utf-8') as f:
            original_json = json.load(f)
    elif args.persona_labeling_target == 'query_topic':
        with open(f"./dataset/{args.dataset}/processed/{args.test_data}.json", 'r', encoding='utf-8') as f:
            original_json = json.load(f)

    max_token = 3000
    model_name = "gpt-4o-2024-08-06"
    batch_requests = []
    request_idx = 0

    inner_batch_size = 10000

    test_cnt = 0


    for persona_dict in original_json:

        request_idx += 1
        
        system_prompt = None
        user_prompt = None

        if args.persona_labeling_target == 'generated_response':
            agent_persona_text = "\n".join([f"{i}. {p.strip()}" for i, p in enumerate(persona_dict["agent_persona"])])
            user_persona_dict = "\n".join([f"{i+len(persona_dict['agent_persona'])}. {p.strip()}" for i, p in enumerate(persona_dict["user_persona"])])
            user_prompt = persona_dict["generated_response"]
            system_prompt = (
                "- Your persona:\n"
                f"{agent_persona_text}\n\n"

                "- Partner's persona:\n"
                f"{user_persona_dict}\n"

                "---\n"
                "You aim to label which persona attributes are used in an utterance.\n"
                "When labeling an utterance, you can refer to the given persona attributes.\n"
                "Provide appropriate numbers only, for example, \"0, 3, 5\".\n"
                "Also, there can be an utterance in which none of the given persona is included: \"None\".\n"
            )

        elif args.persona_labeling_target == 'GT':
            agent_persona_text = "\n".join([f"{i}. {p.strip()}" for i, p in enumerate(persona_dict["agent_persona"])])
            user_persona_dict = "\n".join([f"{i+len(persona_dict['agent_persona'])}. {p.strip()}" for i, p in enumerate(persona_dict["user_persona"])])
            user_prompt = persona_dict["dialogues"][-1]
            system_prompt = (
                "- Your persona:\n"
                f"{agent_persona_text}\n\n"

                "- Partner's persona:\n"
                f"{user_persona_dict}\n"

                "---\n"
                "You aim to label which persona attributes are used in an utterance.\n"
                "When labeling an utterance, you can refer to the given persona attributes.\n"
                "Provide appropriate numbers only, for example, \"0, 3, 5\".\n"
                "Also, there can be an utterance in which none of the given persona is included: \"None\".\n"
            )


        elif args.persona_labeling_target == 'query_topic':

            if args.dataset == 'PersonaChat':
                user_prompt = f"- Target response:\n{ persona_dict['dialogues'][-2] }"
                system_prompt = (
                    "Your task is to label which topics are used in the given target response based on the predefined topic list.\n\n"

                    "Start with clear Instructions describing how to label the topics.\n\n"

                    "Instructions:\n"
                    "- First, carefully read the predefined topic list to fully understand the definitions of each topic category.\n"
                    "- Analyze the target response in its entirety to capture the main meaning and details.\n"
                    "- Compare the content of the target response against each topic definition in the list.\n"
                    "- If the target response contains content that matches a topic, label the corresponding topic.\n"
                    "- If multiple topics are labeled, include all of them in numerical order separated by commas.\n"
                    "- If the target response does not contain any of the listed topics, output exactly \"None\".\n"
                    "- Do not invent or infer topics that are not explicitly present in the target response.\n"
                    "- Ensure the labeling is accurate, unambiguous, and strictly follows the provided instructions.\n\n"

                    "Topic List:\n"
    "0. Preferences(hobby/habit): Topics about an individual's preferences regarding hobbies, habits, and voluntarily maintained periodic routines.\n"
    "1. Preferences(food): The topic of personal preferences in food.\n"
    "2. Preferences(others): Topics on preferences for things other than habit and food, such as technology, animals, and plants.\n"
    "3. Characteristics(appearance): Topics related to the physical characteristics of an individual.\n"
    "4. Characteristics(personality): Topics related to an individual's personality traits.\n"
    "5. Characteristics(others): Individual's other personal features such as nationality, age, voice, and phobia.\n"
    "6. Relationships(family): Topics related to an individual's family relationships.\n"
    "7. Relationships(friend): Topics related to an individual's friends and friendships.\n"
    "8. Relationships(others): Topics related to an individual's social relationships outside of family and friends.\n"
    "9. Experiences(past/present): Topics related to an individual's past experiences or current experiences.\n"
    "10. Experiences(future/plan): Topics about an individual's plans and goals that are feasible and concrete.\n"
    "11. Possessions: Topics related to tangible assets possessed externally by an individual rather than the individual's external/internal characteristics.\n"
    "12. Occupations: Topics on an individual's professional status, position, and professional conduct.\n"
    "13. Beliefs/Values: Attributes consisting of abstract wish verbs (wish, hope, believe, etc.), such as desires and religious beliefs. It is distinguished from 'Experiences(future/plan)' in feasibility and specificity.\n\n"

                    "## Output Format\n"
                    "Return only one of the following:\n"
                    "- A string of topic numbers separated by commas. For example, if both topic 0 and topic 2 are labeled in the target response, then output \"0, 2\".\n"
                    "- Or the string \"None\" if no labeled topics are present.\n\n"
                )

            elif args.dataset == 'PRODIGy':
                user_prompt = f"- Target response:\n{ persona_dict['dialogues'][-2] }"
                system_prompt = (
                    "Your task is to label which topics are used in the given target response based on the predefined topic list.\n\n"

                    "Start with clear Instructions describing how to label the topics.\n\n"

                    "Instructions:\n"
                    "- First, carefully read the predefined topic list to fully understand the definitions of each topic category.\n"
                    "- Analyze the target response in its entirety to capture the main meaning and details.\n"
                    "- Compare the content of the target response against each topic definition in the list.\n"
                    "- If the target response contains content that matches a topic, label the corresponding topic.\n"
                    "- If multiple topics are labeled, include all of them in numerical order separated by commas.\n"
                    "- If the target response does not contain any of the listed topics, output exactly \"None\".\n"
                    "- Do not invent or infer topics that are not explicitly present in the target response.\n"
                    "- Ensure the labeling is accurate, unambiguous, and strictly follows the provided instructions.\n\n"

                    "Topic List:\n"
"0. Preferences(hobby/habit): Topics about an individual's preferences regarding hobbies, habits, and voluntarily maintained periodic routines.\n"
"1. Preferences(food): The topic of personal preferences in food.\n" 
"2. Preferences(others): Topics on preferences for things other than habit and food, such as technology, animals, and plants.\n"
"3. Characteristics(appearance): Topics related to the physical characteristics of an individual.\n"
"4. Characteristics(personality): Topics related to an individual's personality traits.\n" 
"5. Characteristics(others): Individual's other personal features such as nationality, age, voice, and phobia.\n"
"6. Relationships(family): Topics related to an individual's family relationships.\n"
"7. Relationships(friend): Topics related to an individual's friends and friendships.\n"
"8. Relationships(romantic): Topics related to an individual's romantic or dating relationships outside of marriage.\n"
"9. Relationships(others): Topics related to an individual's social relationships outside of family and friends and romantic.\n"
"10. Experiences(past/present): Topics related to an individual's past experiences or current experiences.\n"
"11. Experiences(future/plan): Topics about an individual's plans and goals that are feasible and concrete.\n"
"12. Conditions(work): Topics related to an individual's current work conditions, professional status, position, and professional conduct.\n"
"13. Conditions(study/education): Topics related to an individual's current educational and academic status or achievement.\n"
"14. Conditions(living/environment): Topics related to an individual's current living environment, housing situation and daily lifestyle.\n"
"15. Possessions: Topics related to tangible assets possessed externally by an individual rather than the individual's external/internal characteristics.\n"
"16. Beliefs/Values: Attributes consisting of abstract wish verbs (wish, hope, believe, etc.), such as desires and religious beliefs. It is distinguished from 'Experiences(future/plan)' in feasibility and specificity.\n\n"

                    "## Output Format\n"
                    "Return only one of the following:\n"
                    "- A string of topic numbers separated by commas. For example, if both topic 0 and topic 2 are labeled in the target response, then output \"0, 2\".\n"
                    "- Or the string \"None\" if no labeled topics are present.\n\n"
                )


        # elif args.persona_labeling_target == 'persona_topic':
        #     if args.dataset == 'MBTI-S2Conv':
        #         user_prompt = f"- Target text:\n{ persona_dict['persona'] }"
        #         system_prompt = (
        #             "Your task is to label which topics are related to the given target text based on the predefined topic list.\n"
        #             "The given target text is a sentence describing a persona trait of a virtual character, and the topic list contains each topic category's index and its description.\n"
        #             "If none of the topics are related to the target text, you must create a new topic category.\n"
        #             "You also provide a confidence score indicating how certain you are about the labeled topics.\n"
        #             "Start with clear Instructions how to label the topics.\n\n"
                    
        #             "Instructions:\n"
        #             "- First, carefully read the predefined topic list to fully understand the definitions of each topic category.\n"
        #             "- Analyze the target text in its entirety to capture the main meaning and details.\n"
        #             "- Compare the content of the target text against each topic definition in the list.\n"
        #             "- If the target text contains content that matches a topic, label the corresponding topic.\n"
        #             "- If multiple topics are labeled, output all topic indexes as a list of integers in numerical order.\n"
        #             "- Label the target text using the existing topic list as much as possible. Only if the target text does not match the meaning of any existing topic in the predefined topic list, you must create a new topic category.\n"
        #             "- The newly created topic must be related to target text, and match the style and format of the topics in predefined topic list closely.\n"
        #             "- Ensure the labeling is accurate, unambiguous, and strictly follows the provided instructions.\n"
        #             "- Provide a confidence score between 0 and 1 representing how certain you are of your labeling.\n\n"

        #             "Topic List:\n"
        #             "0. Preferences(hobby/habit): Topics about an individual's preferences regarding hobbies, habits, and voluntarily maintained periodic routines.\n"
        #             "1. Preferences(food): The topic of personal preferences in food.\n"
        #             "2. Preferences(others): Topics on preferences for things other than habit and food, such as technology, animals, and plants.\n"
        #             "3. Characteristics(appearance): Topics related to the physical characteristics of an individual.\n"
        #             "4. Characteristics(personality): Topics related to an individual's personality traits.\n"
        #             "5. Characteristics(advantages/disadvantages): Topics related to an individual's strengths and weaknesses.\n"
        #             "6. Characteristics(communication): Topics related to an individual's style and manner of communication.\n"
        #             "7. Characteristics(others): Individual's other personal features such as name, gender, nationality, age, voice.\n"
        #             "8. Relationships(family): Topics related to an individual's family relationships.\n"
        #             "9. Relationships(friend): Topics related to an individual's friends and friendships.\n"
        #             "10. Relationships(others): Topics related to an individual's social relationships outside of family and friends.\n"
        #             "11. Experiences(past/present): Topics related to an individual's past experiences or current experiences.\n"
        #             "12. Experiences(future/plan): Topics about an individual's plans and goals that are feasible and concrete.\n"
        #             "13. Possessions: Topics related to tangible assets possessed externally by an individual rather than the individual's external/internal characteristics.\n"
        #             "14. Occupations: Topics on an individual's professional status, position, and professional conduct.\n"
        #             "15. Work/Study Conditions: Topics related to an individual's current work or study environment, including their role, responsibilities, work habits, attitudes toward their job or studies, and the nature of the environment they operate in.\n"
        #             "16. Living Conditions/Life Environment: Topics related to an individual's current living environment, housing situation, daily lifestyle, surroundings, emotional attachment to their living space, and the way their home reflects their preferences or personality.\n"
        #             "17. Beliefs/Values: Attributes consisting of abstract wish verbs (wish, hope, believe, etc.), such as desires and religious beliefs. It is distinguished from 'Experiences(future/plan)' in feasibility and specificity.\n"
        #             "18. Emotional State: Topics related to an individual's current emotional and psychological condition, including temporary emotions, mood patterns, emotional reactions within relationships, and ongoing emotional development.\n"
        #             "19. Concerns/Stressors: Topics related to an individual's current worries, anxieties, and sources of psychological stress. This includes specific concerns about career, relationships, self-expectations, personal growth, or any ongoing pressure that affects the individual's emotional or mental state.\n"
        #             "20. Concerns/Stressors Management: Topics related to an individual's proactive behaviors, strategies, or actions taken to manage, reduce, or cope with anxiety or stress.\n"
        #             "21. Social/Environmental Advocacy: Topics related to an individual's involvement in social or environmental causes, including actions or initiatives aimed at promoting positive social change, sustainability, and community well-being.\n"

        #         )

        #     elif args.dataset == 'PRODIGy':
        #         user_prompt = f"- Target text:\n{ persona_dict['persona'] }"
        #         system_prompt = (
        #             "Your task is to label which topics are related to the given target text based on the predefined topic list.\n"
        #             "The given target text is a sentence describing a persona trait of a virtual character, and the topic list contains each topic category's index and its description.\n"
        #             "If none of the topics are related to the target text, you must create a new topic category.\n"
        #             "You also provide a confidence score indicating how certain you are about the labeled topics.\n"
        #             "Start with clear Instructions how to label the topics.\n\n"
                    
        #             "Instructions:\n"
        #             "- First, carefully read the predefined topic list to fully understand the definitions of each topic category.\n"
        #             "- Analyze the target text in its entirety to capture the main meaning and details.\n"
        #             "- Compare the content of the target text against each topic definition in the list.\n"
        #             "- If the target text contains content that matches a topic, label the corresponding topic.\n"
        #             "- If multiple topics are labeled, output all topic indexes as a list of integers in numerical order.\n"
        #             "- Label the target text using the existing topic list as much as possible. Only if the target text does not match the meaning of any existing topic in the predefined topic list, you must create a new topic category.\n"
        #             "- The newly created topic must be related to target text, and match the style and format of the topics in predefined topic list closely.\n"
        #             "- Ensure the labeling is accurate, unambiguous, and strictly follows the provided instructions.\n"
        #             "- Provide a confidence score between 0 and 1 representing how certain you are of your labeling.\n\n"

        #             "Topic List:\n"
        #         "0. Preferences(hobby/habit): Topics about an individual's preferences regarding hobbies, habits, and voluntarily maintained periodic routines.\n"
        #         "1. Preferences(food): The topic of personal preferences in food.\n"
        #         "2. Preferences(others): Topics on preferences for things other than habit and food, such as technology, animals, and plants.\n"
        #         "3. Characteristics(appearance): Topics related to the physical characteristics of an individual.\n"
        #         "4. Characteristics(personality): Topics related to an individual's personality traits.\n"
        #         "5. Characteristics(others): Individual's other personal features such as nationality, age, voice, and phobia.\n"
        #         "6. Relationships(family): Topics related to an individual's family relationships.\n"
        #         "7. Relationships(friend): Topics related to an individual's friends and friendships.\n"
        #         "8. Relationships(others): Topics related to an individual's social relationships outside of family and friends.\n"
        #         "9. Experiences(past/present): Topics related to an individual's past experiences or current experiences.\n"
        #         "10. Experiences(future/plan): Topics about an individual's plans and goals that are feasible and concrete.\n"
        #         "11. Possessions: Topics related to tangible assets possessed externally by an individual rather than the individual's external/internal characteristics.\n"
        #         "12. Occupations: Topics on an individual's professional status, position, and professional conduct.\n"
        #         "13. Beliefs/Values: Attributes consisting of abstract wish verbs (wish, hope, believe, etc.), such as desires and religious beliefs. It is distinguished from 'Experiences(future/plan)' in feasibility and specificity.\n"
                    
        #         )

        # else:
        #     break

        if args.persona_labeling_target != 'persona_topic':
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
        
        
        elif args.persona_labeling_target == 'persona_topic':
            # schema = PersonaTopicOutput.model_json_schema()
            # schema["additionalProperties"] = False

            # request_obj = {
            #     "custom_id": f"request-{request_idx}",
            #     "method": "POST",
            #     "url": "/v1/chat/completions",
            #     "body": {
            #         "model": f"{model_name}",
            #         "messages": [
            #             {"role": "system", "content": system_prompt},
            #             {"role": "user", "content": user_prompt}
            #         ],
            #         "max_tokens": max_token,

            #         "response_format": {
            #             "type": "json_schema",
            #             "json_schema": {
            #                 "name": "persona_topic_output",
            #                 "strict": True,
            #                 "schema": schema
            #             }
            #         }
            #     }
            # }

            persona_list = persona_dict["persona"]
            persona_str = "Persona: [\n" + ",\n".join(
                f'"{p}"' for p in persona_list
            ) + "\n]"
            
            input_text = [{
                "role": "user",
                "content": persona_str
            }]

            input_prompt={
            "id": "pmpt_693baf5ef9608194a01a98db3d29f7930f32ea50cd703e57",
            "version": "4"
            }


            output_template = {
                "format": {
                    "type": "json_schema",
                    "name": "persona_topics",
                    "strict": True,
                    "schema": {
                            "type": "object",
                            "properties": {
                            "result": {
                                "type": "array",
                                "description": "A list of extracted persona items and their associated topics.",
                                "items": {
                                "type": "object",
                                "properties": {
                                    "index": {
                                    "type": "integer",
                                    "description": "The 0-based index of the persona statement in the input array."
                                    },
                                    "persona": {
                                    "type": "string",
                                    "description": "A string reflecting the persona or statement to be analyzed."
                                    },
                                    "topics": {
                                    "type": "array",
                                    "description": "List of extracted topics for this persona.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                        "topic_label": {
                                            "type": "string",
                                            "description": "The topic label describing the extracted topic."
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Explanation for why this topic was assigned."
                                        },
                                        "confidence_score": {
                                            "type": "integer",
                                            "description": "Confidence score from 1 to 10 indicating strength of the topic assignment.",
                                            "minimum": 1,
                                            "maximum": 10
                                        },
                                        "created_topic": {
                                            "type": "string",
                                            "description": "Extra user-created topic label or an empty string if not applicable."
                                        }
                                        },
                                        "required": [
                                        "topic_label",
                                        "reason",
                                        "confidence_score",
                                        "created_topic"
                                        ],
                                        "additionalProperties": False
                                    }
                                    }
                                },
                                "required": [
                                    "index",
                                    "persona",
                                    "topics"
                                ],
                                "additionalProperties": False
                                }
                            }
                            },
                            "required": [
                            "result"
                            ],
                            "additionalProperties": False
            }
                }
            }       

            request_obj = {
                "custom_id": f"request-{request_idx}",
                "method": "POST", 
                "url": "/v1/responses",
                "body" : {
                    "model": model_name,
                    "prompt" : input_prompt,
                    "input" : input_text,
                    "text" : output_template,
                    "max_output_tokens" : max_token,
                    "temperature" : 1.0,
                    "top_p" : 1.0,
                    "store" : True,
                }
            }


        batch_requests.append(request_obj)

        # test_cnt += 1
        # if test_cnt == 10:
        #     break

    print(system_prompt)
    print(user_prompt)

    num_batches = math.ceil(request_idx / inner_batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * inner_batch_size
        end = min((batch_idx + 1) * inner_batch_size, request_idx)
        batch_requests_inner_batch = batch_requests[start:end]

        if args.persona_labeling_target == 'generated_response':
            output_path = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/before/idx{batch_idx}.jsonl"
        else:
            output_path = f"./dataset/{args.dataset}/persona_labeling/{args.persona_labeling_target}/before/idx{batch_idx}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for req in batch_requests_inner_batch:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")




def api_call(args):
    api_key = args.api_key
    client = OpenAI(api_key=api_key)
    batch_idx = 0

    if args.persona_labeling_target == 'generated_response':
        batch_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/before/idx{batch_idx}.jsonl"
    else:
        batch_file = f"./dataset/{args.dataset}/persona_labeling/{args.persona_labeling_target}/before/idx{batch_idx}.jsonl"


    batch_input_file = client.files.create(
    file=open(batch_file, "rb"),
    purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    # batch_job = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/responses",
    #     completion_window="24h",
    #     metadata={"description": "topic labeling"}
    # )

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "topic labeling"}
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

    # output_file_id = batch.output_file_id
    # result = client.files.content(output_file_id).content

    if args.persona_labeling_target == 'generated_response':
        save_path = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/after/idx{batch_idx}.jsonl"
    else:
        save_path = f"./dataset/{args.dataset}/persona_labeling/{args.persona_labeling_target}/after/idx{batch_idx}.jsonl"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(stream.read())
    # with open(save_path, 'wb') as file:
    #     file.write(result)


def persona_labeling(args):
    build_json_for_batch_api(args)
    api_call(args)



# def persona_labeling_processing_query_topic():
#     gpt_output = f"./dataset/persona_labeling/query_topic/after/idx0.jsonl"
    
#     query_topic_list = []
#     with open(gpt_output, 'r', encoding='utf-8') as f:
#         for line in f:
#             json_ = json.loads(line)  
#             response = json_["response"]["body"]["choices"][0]["message"]["content"]
#             parsed_list, parsing_error = parse_to_list(response)
#             if parsing_error:
#                 query_topic_list.append([int(re.match(r"\d+", response).group())])
#             else:
#                 query_topic_list.append(parsed_list)

#     print(len(query_topic_list))

#     input_file = './dataset/processed/persona_labeled/panda_sample_copy.json'
#     output_file = './dataset/processed/persona_labeled/panda_sample.json'

#     with open(input_file, 'r', encoding='utf-8') as f:
#         original_panda_list = json.load(f)
#     cnt = 0
#     after_panda_list = []
#     mapping = {
#         3: 4,
#         4: 5,
#         5: 6,
#         6: 8,
#         7: 10,
#         8: 11,
#         9: 12,
#         10: 13,
#         11: 7,
#         12: 3,
#         13: 14
#     }

#     for panda_idx in range(len(original_panda_list)):
#         panda_dict = original_panda_list[panda_idx]
#         query_topic = query_topic_list[panda_idx]

#         new_query_topic = [mapping.get(x, x) for x in query_topic]  
#         query_labeled_persona = get_query_labeled_persona(new_query_topic, panda_dict["agent_persona"], panda_dict["self_labels"])
#         if len(query_labeled_persona) == 0:
#             cnt += 1
        
#         after_panda_list.append({"agent_persona": panda_dict["agent_persona"], "user_persona": panda_dict["user_persona"] , "dialogues": panda_dict["dialogues"],
#                                  "query_labeled_topic": new_query_topic, "self_labels": panda_dict["self_labels"], "partner_labels": panda_dict["partner_labels"], 'agent_labeled_persona': panda_dict["agent_labeled_persona"] })

#     # os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     # with open(output_file, "w", encoding="utf-8") as f:
#     #     json.dump(after_panda_list, f, ensure_ascii=False, indent=4)
#     print(cnt)



def persona_labeling_processing_query_topic(args):
    gpt_output = f"./dataset/{args.dataset}/persona_labeling/query_topic/after/idx0.jsonl"
    
    query_topic_list = []
    cnt = 0
    with open(gpt_output, 'r', encoding='utf-8') as f:
        for line in f:
            json_ = json.loads(line)  
            response = json_["response"]["body"]["choices"][0]["message"]["content"]
            parsed_list, parsing_error = parse_to_list(response)
            if parsing_error:
                query_topic_list.append([int(re.match(r"\d+", response).group())])
            else:
                query_topic_list.append(parsed_list)
            if len(parsed_list) == 0:
                cnt += 1
    print(cnt)

    print(len(query_topic_list))

    input_file = f'./dataset/{args.dataset}/processed/{args.test_data}.json'
    output_file = f'./dataset/{args.dataset}/processed/persona_labeled/{args.test_data}.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)
    print(len(original_panda_list))

    for panda_idx in range(len(original_panda_list)):
        panda_dict = original_panda_list[panda_idx]
        query_topic = query_topic_list[panda_idx]
        panda_dict["query_labeled_topic"] = query_topic


    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(original_panda_list, f, ensure_ascii=False, indent=4)




def persona_labeling_processing_generated_response(args):

    batch_idx = 0
    generated_response_list = []
    after_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/after/idx{batch_idx}.jsonl"

    parsed_list_cnt = 0
    total_id = []

    with open(after_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_ = json.loads(line)  
            response = json_["response"]["body"]["choices"][0]["message"]["content"]

            custom_id = int(re.search(r"request-(\d+)", json_["custom_id"]).group(1))

            parsed_list, parsing_error = parse_to_list(response)
            if parsing_error:
                parsed_list_cnt += 1
                print(response)
            else:
                total_id.append(custom_id)

                parsed_response = sorted(parsed_list)

                generated_response_list.append(parsed_response)


    input_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response/{args.output_path}.json"
    output_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/processed/{args.output_path}.json"
    with open(input_file, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)
    after_panda_list = copy.deepcopy(original_panda_list)
    save_list = []

    print(f"original request id: {len(after_panda_list)}")
    print(f"parsing error cnt: {parsed_list_cnt}")
    print(f"total request id: {len(total_id)}")

    # for panda_idx in range(len(after_panda_list)):
    for idx, panda_idx in enumerate(total_id):


        # panda_dict = after_panda_list[panda_idx]
        panda_dict = after_panda_list[panda_idx-1]
        labeled_generated_response_persona_idxs = generated_response_list[idx]

        # try:
        #     # labeled_generated_response_persona_idxs = generated_response_list[panda_idx]
        #     labeled_generated_response_persona_idxs = generated_response_list[idx]
        # except:
        #     print('list index out of range')


        agent_persona = panda_dict["agent_persona"]
        user_persona = panda_dict["user_persona"]

        labeled_generated_response_persona = []


        if len(labeled_generated_response_persona_idxs) != 0:

            for labeled_generated_response_persona_idx in labeled_generated_response_persona_idxs:
                total_persona = agent_persona + user_persona
                try:
                    labeled_generated_response_persona.append(total_persona[labeled_generated_response_persona_idx])
                except (IndexError, TypeError):
                    print('labeled persona index error')
                    pass

        panda_dict["generated_response_labeled_persona"] = labeled_generated_response_persona
        save_list.append(panda_dict)


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(len(save_list))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)



def collect_all_persona_mbti():
    all_state_data = []
    cnt = 0

    topic_index_dict = {
        'advantages_and_disadvantages':1,
        'hobby':2,
        'growth_experience':3,
        'family_relationship':4,
        'working_conditions':5,
        'social_relationship':6,
        'emotional_state':7,
        'living_conditions':8,
        'recent_worry_or_anxiety':9
    }


    with open("./dataset/MBTI_S2Conv/original/mbti_1024_bank.json", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            persona_dict = json.loads(line)
            persona_list = []
            labels_list = []

            biography = persona_dict["supporter_statement"]


            profile_trans = persona_dict["profile_trans"]

            is_append = True

            for (key, value) in profile_trans.items():
                if key not in ['name', 'gender', 'age', 'region', 'tone', 'job', 'personality', 'additional_information']:
                    if key in topic_index_dict.keys():
                        print(key)
                        topic_idx = int(topic_index_dict[key])
                        persona_list.append(value)
                        labels_list.append([topic_idx])
                    else:
                        is_append = False

            if is_append:
                state_dict = {
                    'index': persona_dict["index"],
                    'persona': persona_list,
                    'labels': labels_list,
                    'persona_sentence': biography
                }

                all_state_data.append(state_dict)
            
    output_file_path = './dataset/MBTI_S2Conv/processed/all_persona_sentence.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(len(all_state_data))
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_state_data, f, ensure_ascii=False, indent=4)



def sample_dataset_mbti():
    with open("./dataset/MBTI_S2Conv/processed/all_persona_sentence.json", 'r', encoding='utf-8') as f:
        all_state_data = json.load(f)
    all_char_index = []
    for state_dict in all_state_data:
        char_index = state_dict["index"]
        all_char_index.append(char_index)

    total_data = []
    topic_index_dict = {
        'advantages_and_disadvantages':1,
        'hobby':2,
        'growth_experience':3,
        'family_relationship':4,
        'working_conditions':5,
        'social_relationship':6,
        'emotional_state':7,
        'living_conditions':8,
        'recent_worry_or_anxiety':9
    }
    with open("./dataset/MBTI_S2Conv/original/mbti_s2conv.json", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            dialogue_dict = json.loads(line)
            

            agent_idx = int(dialogue_dict["supporter_info"]["supporter_index"])
            partner_idx = int(dialogue_dict["seeker_info"]["seeker_index"])
            is_append = True


            if agent_idx in all_char_index and partner_idx in all_char_index:
                agent_info = all_state_data[all_char_index.index(agent_idx)]
                partner_info = all_state_data[all_char_index.index(partner_idx)]

                dialogue = []
                labeled_topic_list = []

                for dial_idx in range(len(dialogue_dict["dial"])):
                    dial = dialogue_dict["dial"][dial_idx]
                    aspect = dial["choose_info"]["aspect"]


                    if aspect not in topic_index_dict.keys() and aspect != '':
                        is_append = False
                    else:

                        if aspect == "":
                            topic_idx = []
                        else:
                            topic_idx = [topic_index_dict[aspect]]

                        labeled_topic_list.append(topic_idx)
                        dialogue.append(dial["utterance"])



                # if len(labeled_topic_list) != len(dialogue):
                #     print()

                if is_append:
                    total_data.append(
                        { "agent_persona": agent_info["persona"], "user_persona": partner_info["persona"], "dialogues": dialogue, 
                        "self_labels":agent_info["labels"], "partner_labels": partner_info["labels"],
                         "labeled_topic_list": labeled_topic_list,
                          "agent_description": agent_info["persona_sentence"], "user_description": partner_info["persona_sentence"] }
                    )
    
    output_file_path = './dataset/MBTI_S2Conv/processed/total_mbti_s2conv.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(len(total_data))
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(total_data, f, ensure_ascii=False, indent=4)

    sample_num = 300
    sampled_data = random.sample(total_data, sample_num)
    output_file_path = './dataset/MBTI_S2Conv/processed/sampled_mbti_s2conv.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(len(sampled_data))
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)


def collect_sampled_statement_mbti():
    with open('./dataset/MBTI-S2Conv/processed/sampled_mbti_s2conv.json', 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)

    agent_idx_set = []
    partner_idx_set = []

    for sampled_dict in sampled_data:
        agent_idx = sampled_dict["agent_idx"]
        partner_idx = sampled_dict["partner_idx"]

        if agent_idx not in agent_idx_set:
            agent_idx_set.append(agent_idx)
        if partner_idx not in partner_idx_set:
            partner_idx_set.append(partner_idx)

    print(len(agent_idx_set))
    print(len(partner_idx_set))


    with open("./dataset/MBTI-S2Conv/processed/all_persona_sentence.json", 'r', encoding='utf-8') as f:
        all_state_data = json.load(f)

    sampled_state = []

    for idx in agent_idx_set:
        persona_list = all_state_data[idx]["supporter_statement"]

        for persona in persona_list:
            sampled_state.append(
                {'idx': f'agent {idx}', 'persona': persona}
            )


    for idx in partner_idx_set:
        persona_list = all_state_data[idx]["seeker_statement"]

        for persona in persona_list:
            sampled_state.append(
                {'idx': f'user {idx}', 'persona': persona}
            )

    output_file_path = './dataset/MBTI-S2Conv/processed/sampled_persona_sentence.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    print(len(sampled_state))
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(sampled_state, f, ensure_ascii=False, indent=4)


def turn_split_mbti():

    input_file_path = "./dataset/MBTI_S2Conv/processed/sampled_mbti_s2conv.json"
    output_file_path = "./dataset/MBTI_S2Conv/processed/sampled_mbti_s2conv_splited.json"

    mapped_panda_list = []


    with open(input_file_path, 'r', encoding='utf-8') as f:
        original_list = json.load(f)
        
    for original_panda_dict in tqdm(original_list, total=len(original_list)):

        agent_persona = original_panda_dict["agent_persona"]
        user_persona = original_panda_dict["user_persona"]

        agent_description = original_panda_dict["agent_description"]
        user_description = original_panda_dict["user_description"]


        dialogues = original_panda_dict["dialogues"]

        self_labels = original_panda_dict["self_labels"]
        partner_labels = original_panda_dict["partner_labels"]

        labeled_topic_list = original_panda_dict["labeled_topic_list"]



        mapped_dialogues = []

        for utterance_idx in range(len(dialogues)):
            utterance = dialogues[utterance_idx]
            mapped_dialogues.append(utterance)

            if utterance_idx % 2 == 0:
                query_labeled_topic = labeled_topic_list[utterance_idx]
            else:
                GT_labeled_topic = labeled_topic_list[utterance_idx]
                mapped_panda_dict = {'agent_persona': agent_persona,
                                     'user_persona': user_persona, 
                                     'dialogues': mapped_dialogues, 
                                     'self_labels': self_labels, 'partner_labels': partner_labels,
                                     'query_labeled_topic': query_labeled_topic, 'GT_labeled_topic': GT_labeled_topic,
                                     'agent_description': agent_description, 'user_description': user_description }
                

                mapped_panda_list.append(copy.deepcopy(mapped_panda_dict))


    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(mapped_panda_list, f, ensure_ascii=False, indent=4)


def persona_labeling_processing_persona_topic(args):
    batch_idx = 0
    
    original_file_path = f"./dataset/{args.dataset}/processed/{args.test_data}.json"
    result_file_path = f"./dataset/{args.dataset}/persona_labeling/{args.persona_labeling_target}/after/idx{batch_idx}.jsonl"
    output_file_path = f"./dataset/{args.dataset}/processed/{args.test_data}_labeled.json"

    with open(original_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    save_list = []

    with open(result_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            body = row["response"]["body"]
            text_json_str = body["output"][0]["content"][0]["text"]
            response_dict = json.loads(text_json_str)["result"]
            custom_id = int(re.search(r"request-(\d+)", row["custom_id"]).group(1))

            char_idx = original_data[custom_id-1]["idx"]

            response_dict.insert(0, f"char_idx: {char_idx}")

            save_list.append(response_dict)

            # original_data[custom_id-1]["topic_labels"] = response_dict["topic_labels"]
            # original_data[custom_id-1]["confidence_score"] = response_dict["confidence_score"]
            # original_data[custom_id-1]["created_topic"] = response_dict["created_topic"]

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)


# NOTE: this published subset covers PersonaChat and MBTI-S2Conv only (the two
# datasets used in the paper). The original research repo also explored
# PRODIGy/PESConv as candidate datasets during early development; that
# exploratory, unused preprocessing code (revise_dataset_prodigy,
# sample_dataset_prodigy, turn_split_pesconv, turn_split_prodigy) has been
# removed here for clarity.


def train_test_split(dataset):
    with open(f"./dataset/{dataset}/processed/persona_labeled/sampled_splited.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    split_idx = int(len(data) * 0.8)

    train_data = data[:split_idx]
    test_data = data[split_idx:]

    with open(f"./dataset/{dataset}/processed/persona_labeled/sampled_splited_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(f"./dataset/{dataset}/processed/persona_labeled/sampled_splited_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)



def turn_split_personachat():
    json_file_name = 'panda_sample'
    input_json = f"./dataset/original/{json_file_name}.json"
    output_json = f"./dataset/processed/{json_file_name}.json"

    mapped_panda_list = []


    with open(input_json, 'r', encoding='utf-8') as f:
        original_panda_list = json.load(f)
        
    for original_panda_dict in tqdm(original_panda_list, total=len(original_panda_list)):

        agent_persona = original_panda_dict["self_text"]
        user_persona = original_panda_dict["partner_text"]


        dialogues = original_panda_dict["utterances"]

        self_labels = original_panda_dict["self_labels"]
        partner_labels = original_panda_dict["partner_labels"]

        processed_self_labels = process_labels(self_labels)
        processed_partner_labels = process_labels(partner_labels)

        
        mapped_dialogues = []

        for utterance_idx in range(len(dialogues)):
            utterance = dialogues[utterance_idx]
            mapped_dialogues.append(utterance)

            if utterance_idx % 2 != 0:

                mapped_panda_dict = {'agent_persona': agent_persona, 'user_persona': user_persona, 'dialogues': mapped_dialogues, 
                                     'self_labels': processed_self_labels, 'partner_labels': processed_partner_labels}
                mapped_panda_list.append(copy.deepcopy(mapped_panda_dict))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mapped_panda_list, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':
    args = config.get_args()
    set_seed(args.seed)
    # train_test_split("MBTI_S2Conv")


    # turn_split_personachat()


    # collect_all_persona_mbti()
    # sample_dataset_mbti()
    # turn_split_mbti()


    # turn_split_pesconv()


    # revise_dataset_prodigy()
    # sample_dataset_prodigy()
    # turn_split_prodigy()


    # train_data_ratio_list = [
    #         # 0.05,
    #         # 0.1,
    #         0.25,
    #         0.5,
    #                                 ]
    # for train_data_ratio in train_data_ratio_list:


    dataset_list = [
        # "PersonaChat", 
                    "MBTI_S2Conv",
                    ]
    model_list = [
        'llama3',
        'qwen',
        'deepseek',
    ]


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


            if args.process_task == 'persona_labeling':
                print(f"dataset: {args.dataset}")
                print(f"model_name: {args.model_name}")
                print(f"output_path: {args.output_path}")

                persona_labeling(args)
                if args.persona_labeling_target == 'generated_response':
                    persona_labeling_processing_generated_response(args)

                elif args.persona_labeling_target == 'persona_topic':
                    persona_labeling_processing_persona_topic(args)

                elif args.persona_labeling_target == 'query_topic':
                    persona_labeling_processing_query_topic(args)


    # # dataset_list = [
    # #     'PersonaChat', 
    # #                 'MBTI_S2Conv'
    # #                 ]
    # # model_name_list = [
    # #     'llama3', 
    # #                    'qwen', 
    # #                 #    'deepseek'
    # #                    ]
    # # target_layer_ratio_list = [
    # #     #     0.1, 
    # #     # 0.2, 
    # #     # 0.3,
    # #     # 0.4,
    # #     0.6,
    # #     0.7,
    # #     0.8,
    # #     0.9,
    # #     1.0,
    # #                            ]
    # # token_position_steering_list = [
    # #     'all',
    # #     # 'prompt_all', 
    # #     #                             'prompt_last', 
    # #     #                             'gen_all'
    # #                                 ]
    # # for token_position_steering in token_position_steering_list:

    # contrastive_alpha_list = [-1.0, -2.0, -3.0, -4.0]
    # for alpha in contrastive_alpha_list:


    #     # if args.dataset == 'PersonaChat':
    #     #     if args.model_name == 'llama3':
    #     #         alpha = 5.0
    #     #     elif args.model_name == 'qwen':
    #     #         alpha = 7.0
    #     #     elif args.model_name == 'deepseek':
    #     #         alpha = 2.0
    #     # else:
    #     #     if args.model_name == 'llama3':
    #     #         alpha = 5.0
    #     #     elif args.model_name == 'qwen':
    #     #         alpha = 5.0
    #     #     elif args.model_name == 'deepseek':
    #     #         alpha = 2.0

    #     args.output_path = f"self/ours/target_layer_ratio=0.5,token_position=last,differnce_method=mean,normalize_vector=0/use_vector=1,decoding_strategy=steering,contrastive_alpha={alpha},token_position_steering=all"

    #     # args.output_path = f"self/ours/target_layer_ratio={target_layer_ratio}{args.output_path}"

    #     # for dataset  in dataset_list:
    #     #     args.dataset = dataset

    #     #     for model_name in model_name_list:
    #     #         args.model_name = model_name

    #     if args.process_task == 'persona_labeling':
    #         print(f"dataset: {args.dataset}")
    #         print(f"model_name: {args.model_name}")
    #         print(f"output_path: {args.output_path}")

    #         if args.persona_labeling_target == 'generated_response':

    #             # output_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/processed/{args.output_path}.json"
    #             # if os.path.exists(output_file):
    #             #     print(f"[SKIP] Already exists: {output_file}")
    #             # else:
    #             #     print(f"save_path: {output_file}")

    #             persona_labeling(args)
    #             persona_labeling_processing_generated_response(args)

    #         # elif args.persona_labeling_target == 'persona_topic':
    #         #     persona_labeling_processing_persona_topic(args)

    #         # elif args.persona_labeling_target == 'query_topic':
    #         #     persona_labeling_processing_query_topic(args)











    # dataset_list = [
    #     'PersonaChat', 
    #                 'MBTI_S2Conv'
    #                 ]
    # model_name_list = [
    #     'llama3', 
    #                    'qwen', 
    #                    'deepseek'
    #                    ]


    # # output_path_list = [
    # #     # "self/SFT/refine/response",

    # #     # # "self/SFT/refine/response2",

    # #     # "self/SFT/cot/response",

    # #     # "self/SFT/decom/response",

    # #     # "self/SFT/diff_dataset",


    # #     "self/SFT_GT",

    # # ]
    # # for output_path in output_path_list:
    # #     args.output_path = output_path
    # # for dataset in dataset_list:
    # #     args.dataset = dataset
    # #     for model_name in model_name_list:
    # #         args.model_name = model_name



    # if args.dataset == 'PersonaChat':
    #     if args.model_name == 'llama3':
    #         alpha = 5.0
    #     elif args.model_name == 'qwen':
    #         alpha = 7.0
    #     elif args.model_name == 'deepseek':
    #         alpha = 2.0
    # else:
    #     if args.model_name == 'llama3':
    #         alpha = 5.0
    #     elif args.model_name == 'qwen':
    #         alpha = 5.0
    #     elif args.model_name == 'deepseek':
    #         alpha = 2.0

    # args.output_path = f"self/ours/diff_dataset/cot/response/target_layer_ratio=0.5,token_position=last,differnce_method=mean,normalize_vector=0/use_vector=1,decoding_strategy=steering,contrastive_alpha={alpha},token_position_steering=all"

    # if args.process_task == 'persona_labeling':
    #     print(f"dataset: {args.dataset}")
    #     print(f"model_name: {args.model_name}")
    #     print(f"output_path: {args.output_path}")

    #     persona_labeling(args)
    #     if args.persona_labeling_target == 'generated_response':
    #         persona_labeling_processing_generated_response(args)

    #     elif args.persona_labeling_target == 'persona_topic':
    #         persona_labeling_processing_persona_topic(args)

    #     elif args.persona_labeling_target == 'query_topic':
    #         persona_labeling_processing_query_topic(args)

    # if args.process_task == 'persona_labeling_processing':
    #     if args.persona_labeling_target == 'generated_response':
    #         persona_labeling_processing_generated_response(args)    

    #     elif args.persona_labeling_target == 'persona_topic':
    #         persona_labeling_processing_persona_topic(args)

    #     elif args.persona_labeling_target == 'query_topic':
    #         persona_labeling_processing_query_topic()

        













    # output_path_list = [
    #     'self/ours/refine/response2',

    # ]


    # original_output_path = args.output_path

    # for output_path in output_path_list:

    #     prefix_path = f"{output_path}/target_layer_ratio=0.5"
    #     args.output_path = f"{prefix_path},{original_output_path}"

    #     # args.output_path = f"self/ours/target_layer_ratio={target_layer_ratio}{args.output_path}"



    #     if args.process_task == 'persona_labeling':
    #         print(f"dataset: {args.dataset}")
    #         print(f"model_name: {args.model_name}")
    #         print(f"output_path: {args.output_path}")

    #         if args.persona_labeling_target == 'generated_response':
    #             output_file = f"./result/{args.dataset}/{args.model_name}/{args.test_main_sub}/response_labeled/{args.output_path}/processed/{args.output_path}.json"
    #             if os.path.exists(output_file):
    #                 print(f"[SKIP] Already exists: {output_file}")
    #             else:
    #                 print(f"save_path: {output_file}")

    #                 persona_labeling(args)
    #                 persona_labeling_processing_generated_response(args)
   