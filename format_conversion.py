import json, argparse

def convert(filename, outname):
    with open(filename, "r") as json_file:
        results = json.load(json_file)
    with open(outname, "w") as f:
        for i in range(len(results)):
            idx = i
            pred = results[i]["prediction"]
            gold = results[i]["output"]
            inp = results[i]["input"]
            print("{idx}\t{pred}\t{gold}\t{inp}".format(idx=idx, pred=pred, gold=gold, inp=inp), file=f)

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
        '--split', 
        nargs='+',
        type=str,
        choices=["val", "sampleval", "test", "sampletest"]
    )
    parser.add_argument(
        '--subtasks', 
        nargs='+',
        type=str,
        default=["dst", "api", "da", "rg"],
        choices=["dst", "api", "da", "rg"]
    )
    args = parser.parse_args()
    for lang in args.langs:
        for split in args.split:
            for sub in args.subtasks:
                filename = split
                if split == "test" or split == "sampletest":
                    filename = "e2e_results/" + split
                filename = filename + "/" + sub + "_" + lang + ".json"
                outname = "tsv_" + split + "/" + sub + "_" + lang + ".tsv"
                convert(filename, outname)