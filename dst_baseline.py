from llm import llm_generate
import logging
import json
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
)

langs = ["en", "ko", "zh", "enhi", "fr", "hi"]
subtasks = ["dst"]
engine_model = ["gpt-4"]
for eng in engine_model:
    for lang in langs:
        for sub in subtasks:
            filename = "samples/" + sub + "_" + lang + "_validfull.json"
            template = "prompts_new/" + sub + "_" + lang + ".prompt"
            with open(filename, "r") as f:
                samples = json.load(f)
            results = []
            for sample in samples:
                output = llm_generate(
                    template,
                    prompt_parameter_values={
                        "input_from_rg": sample["input"]
                    },
                    engine=eng,
                    temperature=0,
                    max_tokens=500,
                    stop_tokens=["\n"],
                    ban_line_break_start=False,
                )
                sample["prediction"] = output
                results.append(sample)
            outname = "results_baseline/" + sub + "_" + lang + "_results.json"
            with open(outname, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)