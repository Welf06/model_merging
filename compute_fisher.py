"""Script for computing the diagonal Fisher of a model."""
import os

from absl import app
from absl import flags
from absl import logging
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from model_merging import data
from model_merging import fisher
from model_merging import hdf5_util

FLAGS = flags.FLAGS

# TODO: Add descriptions to flags
# flags.DEFINE_string("model", None, "")
# flags.DEFINE_string("glue_task", None, "")
# flags.DEFINE_string("fisher_path", None, "Path of hdf5 file to save Fisher to.")

# flags.DEFINE_bool("from_pt", True, "")

# flags.DEFINE_string("split", "train", "")
# flags.DEFINE_integer("n_examples", 4096, "")
# flags.DEFINE_integer("batch_size", 2, "")
# flags.DEFINE_integer("sequence_length", 128, "")

# flags.mark_flags_as_required(["model", "glue_task", "fisher_path"])


import json

with open("config_fischer.json") as f:
    config = json.load(f)
    
    
def main():
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-RTE", from_pt=config["from_pt"]
    )
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-RTE")
    print("Model loaded")

    ds = data.load_glue_dataset(
        task="rte",
        split=config["split"],
        tokenizer=tokenizer,
        max_length=config["sequence_length"],
    )
    ds = ds.take(config["n_examples"]).batch(config["batch_size"])
    print("Dataset loaded")

    print("Starting Fisher computation")
    fisher_diag = fisher.compute_fisher_for_model(model, ds)

    print("Fisher computed. Saving to file...")
    fisher_path = os.path.expanduser("fishers/rte.hdf5")
    hdf5_util.save_variables_to_hdf5(fisher_diag, fisher_path)
    print("Fisher saved to file")


if __name__ == "__main__":
    # app.run(main)
    main()
