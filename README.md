This repository contains the code to run LLM inference on X-RiSAWOZ, multilingual Task-oriented Dialogue Dataset. 

## Installation

```bash
conda install --file ./conda_env.yaml
```

## Usage

The folders containing the ToD data are divided into "val", "sampleval", "test" and "sampletest" according to the namesake splits. After completing the inference of the given splits, the outcomes are stored in "results_{split}". All instruction prompts for DST are contained in the "prompts_DST" directory, while prompts for the other subtasks are located in the "prompts" directory.

### Turn-by-turn Evaluation
1. DST Inference
The following command runs DST inference on LLMs for belief state tracking after user utterance. 
```bash
python dst_inference.py --langs {langs} --models {gpt-35-turbo, gpt-4} --split {val, sampleval} --post {none, dictionary, llm}
```
The output is stored in "results_{split}/dst_{lang}.json". You can also run two ablation studies in the paper. The following code runs naive-prompting DST inference.
```bash
python dst_baseline.py --langs {langs} --models {gpt-35-turbo, gpt-4} --split {val, sampleval}
```
You can also run the normalization ablations with the "--post" argument.

2. ACD, DAG, RG Inference
The following command runs inference for the remaining ToD subtasks
```bash
python rg_inference.py --langs {langs} --models {gpt-35-turbo, gpt-4} --split {val, sampleval} --subtasks {api, da, rg}
```

### End-to-end Evaluation
Start the inference server:
`make start-inference-server`

Test the inference server:
curl http://127.0.0.1:7878/generate -d '{"language": "en", "model": "gpt-4", "task_input": "DST: <state> null <endofstate> <history> USER: Im from Suzhou but I dont go out much. My friend is coming to Suzhou to visit me and Im thinking about taking her to a water town. <endofhistory>"}' -X POST -H 'Content-Type: application/json'

Run evaluation:
`make run-e2e language=en engine=gpt-4`

## Format Conversion
All the result files above are stored in "json" format and we need to convert them into TSV files to compute various metrics using GenieNLP evaluation. The following code converts the given JSON file into TSV file with appropriate format.
```bash
python format_conversion.py --result_dir <path-to-JSON> --out_dir <path-to-TSV>
```

## GenieNLP Evaluation
After converting the LLM output to TSV, use the following command to compute automated metrics on different subtasks. The computed metrics are stored as JSON file. Also, for dst_em, all the mismatching indices are saved in "errors.tsv".
```bash
python evaluate_file.py --pred_file <path-to-TSV> --pred_tgt_languages {langs} --tasks <path-to-JSON> --extra_metrics {dst_em, da_em, bleu, sacre_bleu}
```

## Citation

TODO: Include citation for the paper.