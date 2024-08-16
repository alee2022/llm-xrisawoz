from llm import llm_generate
from util_dst import predict_domains, create_header, create_body
import logging
import json
from flask import Flask
from flask_restful import Api, reqparse

app = Flask(__name__)
api = Api(app)

logger = logging.getLogger(__name__)

req_parser = reqparse.RequestParser()
req_parser.add_argument("task_input", type=str)
req_parser.add_argument("language", type=str)
req_parser.add_argument("model", type=str, choices=["gpt-4", "gpt-35-turbo"])


def predict_dst(input, lang, model):
    with open("schemas.json", "r") as json_file:
        schemas = json.load(json_file)
    schema = schemas[0]
    domains = predict_domains(input, lang, model)
    # Detecting invalid domains
    flag = False
    for dom in domains:
        if dom not in schema:
            flag = True
            break
    if flag:
        return "ERROR: invalid domain prediction"

    header = create_header(domains, schema)
    body = create_body(domains, lang)

    # DST inference
    prompt_inference = "prompts_dom/dst.prompt"
    try:
        output = llm_generate(
            prompt_inference,
            prompt_parameter_values={
                "input_from_rg": input,
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
        return "ERROR: llm_generate failed with inference!"
    if lang != "zh":
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
                engine=model,
                temperature=0,
                max_tokens=500,
                stop_tokens=["\n"],
                ban_line_break_start=False,
            )
        except:
            return "ERROR: llm_generate failed with postprocessing!"
    return output

'''
TODO: this function should be changed to take in dst parameters for database query 
'''
def predict_others(input, lang, sub, model):
    assert sub == "api" or sub == "da" or sub == "rg"
    prompt = "prompts_new/" + sub + "_" + lang + ".prompt"
    output = llm_generate(
        prompt,
        prompt_parameter_values={
            "input_from_rg": input,
        },
        engine=model,
        temperature=0,
        max_tokens=500,
        stop_tokens=["\n"],
        ban_line_break_start=False,
    )
    return output

@app.route("/generate", methods=["GET", "POST"])
def predict_output():
    args = req_parser.parse_args()
    task_input = args["task_input"]
    language = args["language"]
    model = args["model"]

    # print("task_input = ", task_input)

    if task_input[:2] == "DS":
        ret =  predict_dst(task_input, language, model)
    elif task_input[:2] == "AP":
        ret =  predict_others(task_input, language, "api", model)
    elif task_input[:2] == "DA":
        ret =  predict_others(task_input, language, "da", model)
    elif task_input[:2] == "RG":
        ret =  predict_others(task_input, language, "rg", model)
    else:
        logger.error("Invalid task: %s", task_input)
        return {}, 500
    
    # print("task_output = ", ret)
    return {"task_output": ret}
        
if __name__ == "__main__":
    app.run(port=7878, debug=False, use_reloader=False)