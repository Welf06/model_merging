"""Script for actually merging models."""
import os
import json
from datetime import datetime
import collections


from absl import app
from absl import flags
from absl import logging
from itertools import combinations
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from model_merging import data
from model_merging import evaluation
from model_merging import hdf5_util
from model_merging import merging

# FLAGS = flags.FLAGS

# TODO: Add descriptions to flags

# The target model will be first.
# flags.DEFINE_list("models", None, "")
# flags.DEFINE_string("glue_task", None, "")

# flags.DEFINE_list("fishers", None, "")

# flags.DEFINE_bool("from_pt", True, "")

# flags.DEFINE_string("split", "validation", "")
# flags.DEFINE_integer("n_examples", 4096, "")
# flags.DEFINE_integer("batch_size", 32, "")
# flags.DEFINE_integer("sequence_length", 128, "")

# flags.DEFINE_integer("n_coeffs", 51, "")
# flags.DEFINE_enum("coeff_mode", "grid", ["grid", "random"], "")

# flags.DEFINE_float("fisher_floor", 1e-6, "")
# flags.DEFINE_bool("favor_target_model", True, "")
# flags.DEFINE_bool("normalize_fishers", True, "")


# get config from config.json
with open("config.json") as f:
    config = json.load(f)
all_tasks = config["checkpoint_names"].keys()
all_checkpoints = config["checkpoint_names"].values()

metrics = {
    "desc": config["desc"],
    "tasks": tuple(all_tasks), 
    "checkpoints": tuple(all_checkpoints), 
    "seed": config["seed"],
    "metrics": {}
    }

MergeResult = collections.namedtuple("MergeResult", ["coefficients", "score"])

def load_models(tasks):
    models = []
    tokenizers = []
    for i, model_str in enumerate(tasks):
        model_str = config["checkpoint_names"][model_str]
        print("model_str: ", model_str)
        model_str = os.path.expanduser(model_str)
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_str, from_pt=config["from_pt"]
        )
        models.append(model)
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        tokenizers.append(tokenizer)
    return models, tokenizers


def load_fishers():
    if config["fishers"] == "None":
        return None
    fishers = []
    for fisher_str in config["fishers"]:
        fisher_str = os.path.expanduser(fisher_str)
        fisher = hdf5_util.load_variables_from_hdf5(fisher_str, trainable=False)
        fishers.append(fisher)
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


def average_score(score):
    return sum(score.values()) / len(score.values())
 
def get_best_results(results):
    return max(results, key=lambda r: average_score(r.score))

def get_best_average_results(results):
      results = [(average_score(r[0].score), average_score(r[1].score)) for r in results]
      print(results)
      return max(enumerate(results), key=lambda x: (x[1][0] + x[1][1]) / 2)[0]
 
def print_merge_result(result: MergeResult):
    print(f"Merging coefficients: {result.coefficients}")
    print("Scores:")
    for name, value in result.score.items():
        print(f"  {name}: {value}")
        
def main():
    if config["fishers"] != "None":
        assert len(config["fishers"]) == len(config["checkpoint_names)"])
    for tasks in combinations(all_tasks, config["num_at_once"]):
      metrics['metrics']["_".join(tasks)] = {}
      models, tokenizers = load_models(tasks)

      fishers = load_fishers()
      
      coefficients_set = get_coeffs_set(
            coefficient_ratio=config["coefficient_ratio"],
            n_models = len(models)
         )
      
      results = []
      print('_'*80)
      print(f"    Evaluating {'_'.join(tasks)}")
      print('_'*80)
      for i, task in enumerate(tasks):
         ds = data.load_glue_dataset(
            task=task,
            split=config["split"],
            tokenizer=tokenizers[i],
            max_length=config["sequence_length"],
         )
         ds = ds.take(config["n_examples"]).batch(config["batch_size"])

         metric = evaluation.load_metric_for_glue_task(task)
         print(80 * "*")
         print(task)
         print(80 * "*")
         
         merged_models = merging.merging_coefficients_search(
            models,
            coefficients_set=coefficients_set,
            dataset=ds,
            metric=metric,
            fishers=fishers,
            fisher_floor=config["fisher_floor"],
            favor_target_model=config["favor_target_model"],
            normalize_fishers=config["normalize_fishers"],
         )
         result = []
         for coeffs, merged_model in merged_models:
            score = evaluation.evaluate_model(merged_model, ds, metric)
            res = MergeResult(coefficients=coeffs, score=score)
            result.append(res)
            print_merge_result(res)
            
         models = models[1:] + models[:1]
         print(result)
         results.append(result)
         best = get_best_results(result)
         print("Best Merge")
         print(f"Merging coefficients: {best.coefficients}")
         print("Scores:")
         for name, value in best.score.items():
            print(f"  {name}: {value}")
      temp = [(r1, r2) for r1, r2 in zip(results[0], results[1])]
      print(results)
      best_idx = get_best_average_results(temp)
      best_t1 = results[0][best_idx]
      best_t2 = results[1][best_idx]
      metrics['metrics']["_".join(tasks)][tasks[0]] = best_t1.score
      metrics['metrics']["_".join(tasks)][tasks[1]] = best_t2.score
      
      print(80 * "*")
      print(" Best Average Merge")
      print(80 * "*")
      print(f"Merging coefficients: {best_t1.coefficients}")
      print(tasks[0])
      for name, value in best_t1.score.items():
            print(f"  {name}: {value}")
      print(tasks[1])
      for name, value in best_t2.score.items():
            print(f"  {name}: {value}")
      
    print(metrics)
    # os.makedirs("metrics", exist_ok=True)
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # metric_filename = "_".join([current_time, config["run_name"]]) + ".json"
    # with open(os.path.join("metrics", metric_filename), "w") as f:
    #     json.dump(metrics, f, indent=4)
      
if __name__ == "__main__":
    # app.run(main)
    main()