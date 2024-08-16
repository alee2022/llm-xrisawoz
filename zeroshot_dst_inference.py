from llm import llm_generate
import logging
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
)

def predict_domains(input, lang, eng):
    prompt_schema = "prompts_schema/" + sub + "_" + lang + ".prompt"
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

base_lang = "en"
langs = ["ko"]
subtasks = ["dst"]
engine_model = ["gpt-4"]
for eng in engine_model:
    for lang in langs:
        for sub in subtasks:
            filename = "samples/" + sub + "_" + lang + "_samples.json"
            prompt_inf = "prompts_dom/" + sub + ".prompt"
            with open(filename, "r") as f:
                samples = json.load(f)
            with open("schemas.json", "r") as json_file:
                schemas = json.load(json_file)
            schema = schemas[0]
            results = []
            count = 0
            for sample in samples:
                doms_pred = predict_domains(sample["input"], base_lang, eng)
                flag = False
                for dom in doms_pred:
                    if dom not in schema:
                        flag = True
                        break
                if flag:
                    sample["prediction"] = "invalid domain"
                    results.append(sample)
                    count += 1
                    print(eng, lang, sub, "completed (invalid):", count)
                    continue
                header = create_header(doms_pred, schema)
                body = create_body(doms_pred, base_lang)
                output = llm_generate(
                    prompt_inf,
                    prompt_parameter_values={
                        "input_from_rg": sample["input"],
                        "header": header,
                        "body": body
                    },
                    engine=eng,  # gpt-4, gpt-35-turbo, or text-davinci-003
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
                sample["prediction"] = output
                results.append(sample)
                count += 1
                print(eng, lang, sub, "completed:", count)
                time.sleep(2)
            outname = "results_zero/" + sub + "_" + lang + "_results.json"
            with open(outname, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
