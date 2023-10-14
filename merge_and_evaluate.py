"""Script for actually merging models."""
import os
<<<<<<< HEAD
from datetime import datetime
import collections
import json
=======
>>>>>>> 83377bb (fisher computation fix)

from absl import app
from absl import flags
from absl import logging
<<<<<<< HEAD
from itertools import combinations
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
=======
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
>>>>>>> 83377bb (fisher computation fix)

from model_merging import data
from model_merging import evaluation
from model_merging import hdf5_util
from model_merging import merging
<<<<<<< HEAD
from model_evaluator import ModelEvaluator

# get config
with open("config.json") as f:
    config = json.load(f)

config["checkpoint_names"] = {
      "rte": "textattack/bert-base-uncased-RTE",
      "mrpc": "textattack/bert-base-uncased-MRPC"
}

all_tasks = config["checkpoint_names"].keys()
all_checkpoints = config["checkpoint_names"].values()

metrics = {
    "desc": config["desc"],
    "tasks": tuple(all_tasks),
    "checkpoints": tuple(all_checkpoints),
    "seed": config["seed"],
    "metrics": {}
    }

# load the evaluator 
evaluator = ModelEvaluator(device="not-cuda", seed=config["seed"], split=config["split"])

def load_models(tasks):
    models = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for i, model_str in enumerate(tasks):
        model_str = config["checkpoint_names"][model_str]
        print("model_str: ", model_str)
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
        fisher_str = os.path.join("fishers_"+config["fishers"], task + "_fisher.h5")
=======

FLAGS = flags.FLAGS

# TODO: Add descriptions to flags

# The target model will be first.
flags.DEFINE_list("models", None, "")
flags.DEFINE_string("glue_task", None, "")

flags.DEFINE_list("fishers", None, "")

flags.DEFINE_bool("from_pt", True, "")

flags.DEFINE_string("split", "validation", "")
flags.DEFINE_integer("n_examples", 4096, "")
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_integer("sequence_length", 128, "")

flags.DEFINE_integer("n_coeffs", 51, "")
flags.DEFINE_enum("coeff_mode", "grid", ["grid", "random"], "")

flags.DEFINE_float("fisher_floor", 1e-6, "")
flags.DEFINE_bool("favor_target_model", True, "")
flags.DEFINE_bool("normalize_fishers", True, "")


def load_models():
    models = []
    for i, model_str in enumerate(FLAGS.models):
        model_str = os.path.expanduser(model_str)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_str, from_pt=FLAGS.from_pt
        )
        models.append(model)
        if i == 0:
            tokenizer = AutoTokenizer.from_pretrained(model_str)
    return models, tokenizer


def load_fishers():
    if not FLAGS.fishers:
        return None
    fishers = []
    for fisher_str in FLAGS.fishers:
>>>>>>> 83377bb (fisher computation fix)
        fisher_str = os.path.expanduser(fisher_str)
        fisher = hdf5_util.load_variables_from_hdf5(fisher_str, trainable=False)
        fishers.append(fisher)
    return fishers


<<<<<<< HEAD
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

def main():
    # if config["fishers"] != "None":
    #     assert len(config["fishers"]) == len(config["checkpoint_names"])

    for tasks in combinations(all_tasks, config["num_at_once"]):
      metrics['metrics']["_".join(tasks)] = {}

      models, tokenizer = load_models(tasks)
      fishers = load_fishers(tasks)

      coefficients_set = get_coeffs_set(
            coefficient_ratio=config["coefficient_ratio"],
            n_models = len(models)
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

      for coeffs, merged_model in merged_models:
        merged_model.save_pretrained(f"{'_'.join(tasks)}")

if __name__ == "__main__":
    # app.run(main)
    main()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for tasks in combinations(all_tasks, config["num_at_once"]):
        print(f"{'_'.join(tasks)}")
        model = AutoModelForSequenceClassification.from_pretrained(f"{'_'.join(tasks)}", from_tf=True)
        metric = {}
        for task in tasks:
            print(f"{task}")
            base_model = AutoModelForSequenceClassification.from_pretrained( config["checkpoint_names"][task])
            print("Base Model")
            res = evaluator.evaluate(base_model, tokenizer, task, batch_size=8)
            print(res)
            base_model.base_model.load_state_dict(model.base_model.state_dict())
            print("Merged Model with base model head")
            res = evaluator.evaluate(base_model, tokenizer, task, batch_size=8)
            print(res)
            metric[task] = res
        metrics["metrics"]['_'.join(tasks)] = metric
    print(metrics)
    # save metrics in a json file
    os.makedirs("metrics", exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metric_filename = "_".join([current_time, config["run_name"]]) + ".json"
    with open(os.path.join("metrics", metric_filename), "w") as f:
        json.dump(metrics, f, indent=4)
=======
def get_coeffs_set():
    n_models = len(FLAGS.models)
    if FLAGS.coeff_mode == "grid":
        assert n_models == 2
        return merging.create_pairwise_grid_coeffs(FLAGS.n_coeffs)
    elif FLAGS.coeff_mode == "random":
        return merging.create_random_coeffs(n_models, FLAGS.n_coeffs)
    else:
        raise ValueError


def get_best_results(results):
    return max(results, key=lambda r: evaluation.average_score(r.score))


def main(_):
    if FLAGS.fishers:
        assert len(FLAGS.fishers) == len(FLAGS.models)

    models, tokenizer = load_models()

    fishers = load_fishers()

    ds = data.load_glue_dataset(
        task=FLAGS.glue_task,
        split=FLAGS.split,
        tokenizer=tokenizer,
        max_length=FLAGS.sequence_length,
    )
    ds = ds.take(FLAGS.n_examples).batch(FLAGS.batch_size)

    metric = evaluation.load_metric_for_glue_task(FLAGS.glue_task)

    coefficients_set = get_coeffs_set()

    results = merging.merging_coefficients_search(
        models,
        coefficients_set=coefficients_set,
        dataset=ds,
        metric=metric,
        fishers=fishers,
        fisher_floor=FLAGS.fisher_floor,
        favor_target_model=FLAGS.favor_target_model,
        normalize_fishers=FLAGS.normalize_fishers,
    )

    best = get_best_results(results)
    print(80 * "*")
    print(" Best Merge")
    print(80 * "*")
    merging.print_merge_result(best)


if __name__ == "__main__":
    app.run(main)
>>>>>>> 83377bb (fisher computation fix)
