include ./API_KEYS

engine ?= gpt-4# gpt-4 or gpt-35-turbo
language ?= en

dst-inference:
	python dst_inference.py --langs hi --split val

rg-inference:
	python rg_inference.py --langs en --split val 

zeroshot-dst:
	python zeroshot_dst_inference.py

dst-postprocess:
	python postprocess.py

e2e-eval:
	python e2e_test.py --langs en ko --split sampletest

dst-baseline:
	python dst_baseline.py

start-inference-server:
	python generic_inf.py

run-e2e:
	genienlp predict \
		--data xrisawoz_data/$(language) \
		--task risawoz \
		--eval_dir ./pred_e2e/ \
		--evaluate valid \
		--e2e_dialogue_evaluation \
		--language $(language) \
		--llm $(engine) \
		--llm_url http://127.0.0.1:7878 \
		--subsample 100

	python compute_e2e.py \
		--reference_file_path xrisawoz_data/$(language)/original_valid.json \
        --prediction_file_path ./pred_e2e/valid/e2e_dialogue_preds.json \
        --experiment risawoz \
        --setting $(language)

run-tbt:
	genienlp predict \
		--data xrisawoz_data/$(language) \
		--task risawoz \
		--eval_dir ./pred/ \
		--evaluate valid \
		--language $(language) \
		--llm $(engine) \
		--llm_url http://127.0.0.1:7878 \
		--subsample 100