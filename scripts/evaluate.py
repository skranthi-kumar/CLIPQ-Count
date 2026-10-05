#!/usr/bin/env python
"""Evaluate a trained CLIPQ-Count checkpoint on an FSC-147 split.

Example:
    python scripts/evaluate.py --data-root FSC147 --checkpoint checkpoints/clipq_count_best.pth --split test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from clipq_count import CLIPCountingModel, FSC147Dataset, collate_fn
from clipq_count.engine import run_eval
from clipq_count.utils import get_device, logger


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="FSC147")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    ds = FSC147Dataset(os.path.join(args.data_root, "images_384_VarV2"),
                       os.path.join(args.data_root, "annotation_FSC147_384.json"),
                       os.path.join(args.data_root, "Train_Test_Val_FSC_147.json"),
                       shots=args.shots, split=args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=True, collate_fn=collate_fn)

    model = CLIPCountingModel(shots=args.shots).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    mae, prec, loss, preds, targets = run_eval(model, loader, device, desc=f"Evaluating {args.split}")
    rmse = float(((preds - targets) ** 2).mean() ** 0.5)
    logger.info("%s  MAE %.2f  RMSE %.2f  precision(+/-1) %.3f  loss %.4f", args.split.upper(), mae, rmse, prec, loss)


if __name__ == "__main__":
    main()
