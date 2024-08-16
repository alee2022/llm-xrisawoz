from llm import llm_generate
from util_dst import predict_domains, create_header, create_body
import logging
import json, argparse
import time

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
# )

# Build input by replacing the label slots with previous predictions
def parse_dst(input, belief, acts_prev, acts):
    state_tag = "<state>"
    eos_tag = "<endofstate>"
    prev_tag = "AGENT_ACTS_PREV:"
    act_tag = "AGENT_ACTS:"
    user_tag = "USER:"
    
    state_idx = input.find(state_tag)
    eos_idx = input.find(eos_tag)
    prev_idx = input.find(prev_tag)
    act_idx = input.find(act_tag)
    user_idx = input.find(user_tag)

    if act_idx != -1:
        replaced = input[act_idx + len(act_tag) + 1 : user_idx - 1]
        input = input.replace(replaced, acts)
        if prev_idx != -1:
            replaced = input[prev_idx + len(prev_tag) + 1 : act_idx - 1]
            input = input.replace(replaced, acts_prev)
    replaced = input[state_idx + len(state_tag) + 1 : eos_idx - 1]
    if replaced == "null":
        return "null"
    input = input.replace(replaced, belief)
    return input

def parse_rg(input, acts):
    action_tag = "<actions>"
    end_tag = "<endofactions>"

    action_idx = input.find(action_tag)
    end_idx = input.find(end_tag)
    replaced = input[action_idx + len(action_tag) + 1 : end_idx - 1]
    input = input.replace(replaced, acts)
    return input
    
def e2e_inference(model, lang, split, filename, outname):
    # Loading Data
    with open(filename, "r") as f:
        turns = json.load(f)
    if split == "sampletest":
        # 10 percent of samples
        turns = turns[:400] + turns[1164:1564] + turns[4656:5056] + turns[5752:6152]
        turns = turns[:12]
    # Loading Schema for DST
    with open("schemas.json", "r") as json_file:
        schemas = json.load(json_file)
    schema = schemas[0]
    # Keeping track of results
    results_dst = []
    results_api = []
    results_da = []
    results_rg = []
    # Keeping Record of Predictions
    state = "null"
    # TODO: knowledge = ""
    acts_prev = "null"
    acts = "null"
    for idx in range(len(turns)):
        turn = turns[idx]
        # DST Hierarchical Prompting
        if idx % 4 == 0:
            val = parse_dst(turn["input"], state, acts_prev, acts)
            if val == "null":
                acts_prev = "null"
                acts = "null"
            else:
                turn["input"] = val
            # Domain Classification
            doms_pred = predict_domains(turn["input"], lang, model)
            flag = False
            for dom in doms_pred:
                if dom not in schema:
                    flag = True
                    break
            if flag:
                turn["prediction"] = ",".join(doms_pred)
                state = ",".join(doms_pred)
                results_dst.append(turn)
                continue
            # Parsing with schema
            header = create_header(doms_pred, schema)
            body = create_body(doms_pred, lang)
            try:
                output = llm_generate(
                    "prompts_dom/dst.prompt",
                    prompt_parameter_values={
                        "input_from_rg": turn["input"],
                        "header": header,
                        "body": body
                    },
                    engine=model,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
            except:
                output = "null"
            # Post-processing
            if lang == "zh":
                turn["prediction"] = output
                results_dst.append(turn)
                continue
            with open("classification_" + lang + ".json", "r") as json_file:
                enums = json.load(json_file)
            header = ""
            for dom in enums:
                header += "Domain: " + dom + "\n"
                header += enums[dom]
            try:
                output = llm_generate(
                    "prompts_post/postproc_" + lang + ".prompt",
                    prompt_parameter_values={
                        "input_from_rg": output,
                        "header": header,
                        "body": body
                    },
                    engine=model,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
            except:
                output = "Posprocessing error"
            turn["prediction"] = output
            state = output
            results_dst.append(turn)
        # API CALL DETECTION
        # TODO: implement code from GenieNLP which passes in parameters from the belief state
        elif idx % 4 == 1:
            turn["input"] = parse_dst(turn["input"], state, acts_prev, acts)
            try:
                output = llm_generate(
                    "prompts_new/api_" + lang + ".prompt",
                    prompt_parameter_values={
                        "input_from_rg": turn["input"],
                    },
                    engine=model,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
            except:
                output = "error"
            turn["prediction"] = output
            results_api.append(turn)
        # Dialogue Act Generation (DA)
            # TODO: Replace <knowledge> with the API result from the previous turn
        elif idx % 4 == 2:
            turn["input"] = parse_dst(turn["input"], state, acts_prev, acts)
            try:
                output = llm_generate(
                    "prompts_new/da_" + lang + ".prompt",
                    prompt_parameter_values={
                        "input_from_rg": turn["input"],
                    },
                    engine=model,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
            except:
                output = "null"
            turn["prediction"] = output
            results_da.append(turn)
            acts_prev = acts
            acts = output
        # Response Generation
        elif idx % 4 == 3:
            # Swapping the dialogue act to the one generated in the previous turn.
            turn["input"] = parse_rg(turn["input"], acts)
            try:
                output = llm_generate(
                    "prompts_new/rg_" + lang + ".prompt",
                    prompt_parameter_values = {
                    "input_from_rg": turn["input"],
                    },
                    engine=model,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
            except:
                output = "LLM_GENERATE FAILED!"
            turn["prediction"] = output
            results_rg.append(turn)
    outname_dst = outname + "dst_" + lang + ".json"
    outname_api = outname + "api_" + lang + ".json"
    outname_da = outname + "da_" + lang + ".json"
    outname_rg = outname + "rg_" + lang + ".json"
    with open(outname_dst, "w") as f:
        json.dump(results_dst, f, indent=2, ensure_ascii=False)
    with open(outname_api, "w") as f:
        json.dump(results_api, f, indent=2, ensure_ascii=False)
    with open(outname_da, "w") as f:
        json.dump(results_da, f, indent=2, ensure_ascii=False)
    with open(outname_rg, "w") as f:
        json.dump(results_rg, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--langs', 
        nargs='+', 
        type=str, 
        default=["en", "ko", "zh", "enhi", "fr", "hi"], 
        choices=["en", "ko", "zh", "enhi", "fr", "hi"]
    )
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        default=["gpt-4"],
        choices=["gpt-35-turbo", "gpt-4"]
    )
    parser.add_argument(
        '--split', 
        nargs='+',
        type=str,
        choices=["test", "sampletest"]
    )
    args = parser.parse_args()

    for model in args.models:
        for lang in args.langs:
            for split in args.split:
                filename = "data/multi-llama/" + lang + "test.json"
                outname = "e2e_results/" + split + "/"
                e2e_inference(model, lang, split, filename, outname)