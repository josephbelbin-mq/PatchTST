#!/usr/bin/env python

from pathlib import Path
import subprocess
import numpy as np
import argparse
import pandas as pd
import time
import gc
import torch
import time

from run_longExp import get_parser, process_args


# ============================================================================
# Experiment configuration
# ============================================================================

sweep_parser = argparse.ArgumentParser()

sweep_parser.add_argument(
    "--dataset_dir",
    required=True,
)

sweep_parser.add_argument(
    "--model",
    required=True,
)

sweep_args = sweep_parser.parse_args()

MODEL = sweep_args.model

DATASET_DIR = Path(sweep_args.dataset_dir)

ARGS = [
    "--is_training", "1",
    "--model_id", "synthetic_96_96",
    "--model", MODEL,
    "--data", "Synthetic",
    "--features", "M",
    "--seq_len", "96",
    #"--seq_len", "336",
    "--label_len", "48",
    "--pred_len", "96",
    "--patch_len", "16",
    "--stride", "8",
    "--fc_dropout", "0.2",
    "--head_dropout", "0.0",
    "--patience", "3",
    "--lradj", "TST",
    "--pct_start", "0.2",
    "--e_layers", "3",
    "--n_heads", "8",
    "--enc_in", "8",
    "--dec_in", "8",
    "--c_out", "8",
    "--des", "Exp",
    "--d_model", "128",
    "--d_ff", "256",
    "--batch_size", "16",
    "--learning_rate", "0.0001",
    "--itr", "1",
    "--train_epochs", "10",
]

def build_current_args(dataset_path):
    """
    Construct exactly the same argparse Namespace
    that run.py will see.
    """
    parser = get_parser()
    cli_args = (
        ARGS
        + [
            "--root_path", str(DATASET_DIR),
            "--data_path", dataset_path.name,
        ]
    )

    args = parser.parse_args(cli_args)
    args = process_args(args)

    return vars(args)


def load_completed_runs(results_root="results"):
    """
    Returns:
        set(
            (data_source, args_tuple)
        )
    """
    completed_keys = {}
    completed_datasets = set()
    for result_file in Path(results_root).rglob("*_results.npz"):
        try:
            d = np.load(result_file, allow_pickle=True)
            data_source = str(d["data_source"])
            saved_args = d["args"].item()
            key = (
                data_source,
                frozenset(saved_args.items()),
            )
            completed_keys[key] = result_file
            completed_datasets.add((data_source, saved_args["model"]))

        except Exception as e:
            print(f"WARNING: failed to read {result_file}")
            print(e)

    return completed_keys, completed_datasets

def run_dataset(dataset_path):
    cmd = [
        "python",
        "-u",
        "run_longExp.py",
        *ARGS,
        "--root_path", str(DATASET_DIR),
        "--data_path", dataset_path.name,
    ]

    print()
    print("=" * 80)
    print("RUN:", dataset_path.name)
    print("=" * 80)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    print("running")
    completed_keys, completed_datasets = load_completed_runs()
    datasets = sorted(DATASET_DIR.glob("*.npz"))

    rows = []

    print(f"Found {len(datasets)} datasets")
    print(f"Found {len(completed_keys)} completed runs")

    for i, dataset_path in enumerate(datasets, start=1):
        n_remaining = len(datasets) - i + 1

        print(
            f"\n[{i}/{len(datasets)}] "
            f"remaining={n_remaining} "
            f"{dataset_path.name}"
        )
        current_args = build_current_args(dataset_path)
        data_source = str(dataset_path.resolve())
        key = (
            data_source,
            frozenset(current_args.items()),
        )
        if "rep000_lambda1.00_persistence1.00_gamma1.00" not in dataset_path.name:
            continue
        if key in completed_keys:
            print(f"SKIP      {dataset_path.name}")

            result_file = completed_keys[key]
            d = np.load(result_file, allow_pickle=True)

            rows.append({
                "input_file": str(dataset_path),
                "result_file": str(result_file),
                "status": "SKIP",
                "processing_time": 0.0,
                "mse": float(d["mse"]),
                "mae": float(d["mae"]),
                "rmse": float(d["rmse"]),
                "mape": float(d["mape"]),
                "mspe": float(d["mspe"]),
            })
            continue
        elif (data_source, MODEL) in completed_datasets:
            #print("MODEL =", MODEL)
            #print("data_source =", data_source)
            #print("lookup =", (data_source, MODEL))
            #print("completed_datasets contains:")
            #for x in completed_datasets:
            #    print(x)
            status = "RERUN"
        else:
            status = "NEW"
        print(f"{status:<9} {dataset_path.name}")

        t0 = time.time()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_dataset(dataset_path)        
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - t0

        completed_keys_after, _ = load_completed_runs()

        result_file = completed_keys_after[key]

        d = np.load(result_file, allow_pickle=True)

        rows.append({
            "input_file": str(dataset_path),
            "result_file": str(result_file),
            "status": status,
            "processing_time": elapsed,
            "mse": float(d["mse"]),
            "mae": float(d["mae"]),
            "rmse": float(d["rmse"]),
            "mape": float(d["mape"]),
            "mspe": float(d["mspe"]),
        })
        print("Just napping")
        time.sleep(30)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    summary_file = f"sweep_summary_{MODEL}.csv"
    df.to_csv(summary_file, index=False)

