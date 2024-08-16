from llm import llm_generate
import logging
import json, argparse
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
)

def inference(model, lang, sub, filename, outname):
    with open(filename, "r") as f:
        samples = json.load(f)
    prompt = "prompts_new/" + sub + "_" + lang + ".prompt"
    results = []
    samples = samples[:2]
    for sample in samples:
        output = llm_generate(
            prompt,
            prompt_parameter_values={
                "input_from_rg": sample["input"],
            },
            engine=model,  # gpt-4, gpt-35-turbo, or text-davinci-003
            temperature=0,
            max_tokens=500,
            stop_tokens=["\n"],
            ban_line_break_start=False,
        )
        sample["prediction"] = output
        results.append(sample)
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
        '--subtasks',
        nargs='+',
        type=str,
        default=["api", "da", "rg"],
        choices=["api", "da", "rg"]
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
                for sub in args.subtasks:
                    filename = split + "/" + sub + "_" + lang + ".json"
                    outname = "results_" + split + "/" + sub + "_" + lang + ".json"
                    inference(model, lang, sub, filename, outname)