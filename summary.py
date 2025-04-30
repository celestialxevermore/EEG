import os
import json
import csv
import numpy as np

RESULT_DIR = "./result"
TARGETS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CASE_IDS = list(range(1, 11))

case_description = {
    1: "n_band=1, no mutual, no KL, no DA",
    2: "n_band=1, mutual only",
    3: "n_band=1, mutual + KL",
    4: "n_band=3, no mutual, no KL, no DA",
    5: "n_band=3, mutual only",
    6: "n_band=3, mutual + KL",
    7: "n_band=1, DA only",
    8: "n_band=3, DA only",
    9: "n_band=1, mutual + KL + DA",
    10: "n_band=3, mutual + KL + DA",
}

def extract_acc(mode):
    summary = []
    for case_id in CASE_IDS:
        row = []
        for target_id in TARGETS:
            path = os.path.join(RESULT_DIR, str(target_id), f"c{case_id}", mode, "args.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    acc = json.load(f).get("acc", None)
                    if acc is not None:
                        row.append(round(acc, 4))
                    else:
                        row.append("NaN")
            else:
                row.append("NaN")
        valid = [v for v in row if isinstance(v, float)]
        row.append(round(np.mean(valid), 4) if valid else "NaN")
        summary.append(row)
    return summary

finetune_table = extract_acc("finetune")
pretrain_table = extract_acc("pretrain")

with open(os.path.join(RESULT_DIR, "summary_with_pretrain.csv"), "w", newline='') as f:
    writer = csv.writer(f)

    header = ["Case\\Target"] + [f"T{t}" for t in TARGETS] + ["Mean"]

    writer.writerow(["[FINETUNE ACCURACY]"])
    writer.writerow(header)
    for i, row in enumerate(finetune_table):
        writer.writerow([f"Case{i+1}"] + row)

    writer.writerow([])

    writer.writerow(["[PRETRAIN ACCURACY]"])
    writer.writerow(header)
    for i, row in enumerate(pretrain_table):
        writer.writerow([f"Case{i+1}"] + row)

    writer.writerow([])
    writer.writerow(["[CASE DESCRIPTIONS]"])
    for cid, desc in case_description.items():
        writer.writerow([f"Case{cid}: {desc}"])
