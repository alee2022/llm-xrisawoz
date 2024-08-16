from llm import llm_generate
import logging
import json, argparse
import time

def predict_domains(input, lang, eng):
    prompt_schema = "prompts_schema/dst_" + lang + ".prompt"
    output = llm_generate(
        prompt_schema,
        prompt_parameter_values={
            "input_from_rg": input
        },
        engine=eng,  # gpt-4, gpt-35-turbo, or text-davinci-003
        temperature=0,
        max_tokens=500,
        stop_tokens=["\n"],
        ban_line_break_start=False,
    )
    return output.split(", ")

def create_header(domains, schema):
    header = ""
    for dom in domains:
        header += "( "
        header += dom
        header += " ): "
        header += ", ".join(schema[dom])
        header += "; "
    return header

def create_body(domains, lang):
    user_start = "<|user_start|>"
    user_end = "<|user_end|>"
    assistant_start = "<|assistant_start|>"
    assistant_end = "<|assistant_end|>"
    with open("samples_idx.json", "r") as json_file:
        fs_examples = json.load(json_file)
    fs_examples = fs_examples[0]
    fewshot_filename = "data/multi-llama/" + lang + "fewshot.json"
    with open(fewshot_filename, "r") as json_file:
        data = json.load(json_file)
    body = ""
    for dom in domains:
        indices = fs_examples[dom]
        for idx in indices:
            body += user_start
            body += data[idx]["input"]
            body += user_end
            body += "\n"
            body += assistant_start
            body += data[idx]["output"]
            body += assistant_end
            body += "\n\n"
    return body

        
