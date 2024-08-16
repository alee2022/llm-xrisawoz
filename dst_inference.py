from llm import llm_generate
from util_dst import predict_domains, create_header, create_body
import logging
import json, time, argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
)

def predict_state(model, lang, filename, outname):
    prompt_inf = "prompts_dom/dst.prompt"
    # filename = "dst_hi_results.json"
    # outname = "dst_hi_errors.json"
    with open(filename, "r") as f:
        samples = json.load(f)
    with open("schemas.json", "r") as json_file:
        schemas = json.load(json_file)
    schema = schemas[0]
    results = []
    count = 0
    for sample in samples:
        doms_pred = predict_domains(sample["input"], lang, model)
        flag = False
        for dom in doms_pred:
            if dom not in schema:
                flag = True
                break
        if flag:
            sample["prediction"] = "invalid domain"
            results.append(sample)
            count += 1
            print(model, lang, "completed (invalid):", count)
            continue
        header = create_header(doms_pred, schema)
        body = create_body(doms_pred, lang)
        try:
            output = llm_generate(
                prompt_inf,
                prompt_parameter_values={
                    "input_from_rg": sample["input"],
                    "header": header,
                    "body": body
                },
                engine=model,  # gpt-4, gpt-35-turbo, or text-davinci-003
                temperature=0,
                max_tokens=500,
                stop_tokens=["\n"],
                ban_line_break_start=False,
            )
        except:
            output = "ERROR: llm_generate failed! line 52"
        if lang == "zh":
            sample["prediction"] = output
            results.append(sample)
            count += 1
            print(model, lang, "completed:", count, "total:", len(samples))
            # time.sleep(2)
            continue
        # Post-processing after inference
        # time.sleep(2)
        prompt_post = "prompts_post/postproc_" + lang + ".prompt"
        file_class = "classification_" + lang + ".json"
        with open(file_class, "r") as json_file:
            enums = json.load(json_file)
        header = ""
        for dom in enums:
            header += "Domain: " + dom + "\n"
            header += enums[dom]
        try:
            output = llm_generate(
                prompt_post,
                prompt_parameter_values={
                    "input_from_rg": output,
                    "header": header,
                    "body": body
                },
                engine=model,  # gpt-4, gpt-35-turbo, or text-davinci-003
                temperature=0,
                max_tokens=500,
                stop_tokens=["\n"],
                ban_line_break_start=False,
            )
        except:
            output = "ERROR: llm_generate failed! line 86"
        sample["prediction"] = output
        results.append(sample)
        count += 1
        print(model, lang, "completed:", count, "total:", len(samples))
        # time.sleep(2)
    with open(outname, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
        choices=["val", "test", "sampleval", "sampletest"]
    )
    args = parser.parse_args()

    for model in args.models:
        for lang in args.langs:
            for split in args.split:
                filename = split + "/dst_" + lang + ".json"
                outname = "results_" + split + "/dst_" + lang + ".json"
                predict_state(model, lang, filename, outname)