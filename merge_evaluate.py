"""Script for actually merging models."""
import os
from datetime import datetime
import collections
import json

from itertools import combinations
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification

from model_merging import data
from model_merging import evaluation
from model_merging import hdf5_util
from model_merging import merging
from model_evaluator import ModelEvaluator

def load_models(tasks):
    models = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for i, model_str in enumerate(tasks):
        model_str = config["checkpoint_names"][model_str]
        model_str = os.path.expanduser(model_str)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_str, from_pt=config["from_pt"]
        )
        models.append(model)
    return models, tokenizer


def load_fishers(tasks):
    if config["fishers"] == "None":
        return None
    fishers = []
    for task in tasks:
        fisher_str = os.path.join(
            "fishers_"+config["fishers"]+"_"+str(config["n_examples"]), task + "_fisher.h5")
        fisher_str = os.path.join(config["base_model_name"], fisher_str)
        fisher_str = os.path.join("fishers", fisher_str)
        fisher_str = os.path.expanduser(fisher_str)
        fisher = hdf5_util.load_variables_from_hdf5(
            fisher_str, trainable=False)
        fishers.append(fisher)
    print(f"loaded fishers for {'_'.join(tasks)}")
    return fishers


def get_coeffs_set(
    coefficient_ratio="best",
    n_models=0
):
    if coefficient_ratio == "equal":
        return [(0.5, 0.5)]
    elif coefficient_ratio == "best":
        if config["coeff_mode"] == "grid":
            assert n_models == 2
            return merging.create_pairwise_grid_coeffs(config["n_coeffs"])
        elif config["coeff_mode"] == "random":
            return merging.create_random_coeffs(n_models, config["n_coeffs"])
        else:
            raise ValueError
    else:
        raise ValueError


def merge_models():
    # if config["fishers"] != "None":
    #     assert len(config["fishers"]) == len(config["checkpoint_names"])

    for tasks in combinations(all_tasks, config["num_at_once"]):
        metrics['metrics']["_".join(tasks)] = {}

        models, tokenizer = load_models(tasks)
        fishers = load_fishers(tasks)

        coefficients_set = get_coeffs_set(
            coefficient_ratio=config["coefficient_ratio"],
            n_models=len(models)
        )

        print(f"Merging {'_'.join(tasks)}")

        merged_models = merging.merging_coefficients_search(
            models,
            coefficients_set=coefficients_set,
            fishers=fishers,
            fisher_floor=config["fisher_floor"],
            favor_target_model=config["favor_target_model"],
            normalize_fishers=config["normalize_fishers"],
        )
        os.makedirs("merged_models", exist_ok=True)
        
        for coeffs, merged_model in merged_models:
            merged_model.save_pretrained(os.path.join("merged_models", config["base_model_name"], '_'.join(tasks)))


if __name__ == "__main__":
    # get config
    with open("config.json") as f:
        config = json.load(f)

    all_tasks = config["checkpoint_names"].keys()
    all_checkpoints = config["checkpoint_names"].values()

    metrics = {
        "desc": config["desc"],
        "tasks": tuple(all_tasks),
        "checkpoints": tuple(all_checkpoints),
        "seed": config["seed"],
        "fisher_samples": config["n_examples"],
        "metrics": {}
    }

    # load the evaluator
    evaluator = ModelEvaluator(device=config["device"])

    # merge models
    if config["use_saved"]:
        print("Using saved models")
    else: 
        print("Merging models")
        merge_models()
    
    # evaluate merged models
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for tasks in combinations(all_tasks, config["num_at_once"]):
        print(f"{'_'.join(tasks)}")
        merged_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join("merged_models",  config["base_model_name"], '_'.join(tasks)), from_tf=True)
        metric = {}
        for task in tasks:
            print(f"{task}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config["checkpoint_names"][task])
            # print(f"{task} Base Model")
            # res = evaluator.evaluate(base_model, tokenizer, task, batch_size=8)
            # print(res)
            base_model.base_model.load_state_dict(merged_model.base_model.state_dict())
            print(f"{'_'.join(tasks)} with {task} head")
            res = evaluator.evaluate(base_model, tokenizer, task, batch_size=8)
            print(res)
            metric[task] = res
        metrics["metrics"]['_'.join(tasks)] = metric
    # save metrics in a json file
    os.makedirs("metrics", exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metric_filename = "_".join([current_time, config["run_name"]]) + ".json"
    with open(os.path.join("metrics", metric_filename), "w") as f:
        json.dump(metrics, f, indent=4)
