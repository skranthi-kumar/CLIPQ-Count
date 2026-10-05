#!/usr/bin/env python
"""Train CLIPQ-Count on FSC-147 and report test MAE.

Example:
    python scripts/train.py --data-root FSC147 --shots 3 --epochs 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from clipq_count import CLIPCountingModel, FSC147Dataset, collate_fn, verify_dataset
from clipq_count.engine import run_eval, train
from clipq_count.utils import get_device, logger, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="FSC147", help="folder containing images_384_VarV2/ and the two JSON files")
    p.add_argument("--shots", type=int, default=3, help="exemplar boxes per image (max 5)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default="checkpoints/clipq_count_best.pth")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    data_dir = os.path.join(args.data_root, "images_384_VarV2")
    anno = os.path.join(args.data_root, "annotation_FSC147_384.json")
    split = os.path.join(args.data_root, "Train_Test_Val_FSC_147.json")
    if not verify_dataset(data_dir, anno, split):
        sys.exit(1)

    def loader(name, shuffle, augment):
        ds = FSC147Dataset(data_dir, anno, split, shots=args.shots, split=name, augment=augment)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=True, collate_fn=collate_fn)

    train_loader, val_loader, test_loader = loader("train", True, True), loader("val", False, False), loader("test", False, False)
    logger.info("Sizes - train %d, val %d, test %d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    model = CLIPCountingModel(shots=args.shots)
    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
          patience=args.patience, checkpoint_path=args.checkpoint)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint (val MAE %.2f)", ckpt["best_val_mae"])

    mae, prec, _, preds, targets = run_eval(model, test_loader, device, desc="Testing")
    rmse = float(((preds - targets) ** 2).mean() ** 0.5)
    logger.info("TEST  MAE %.2f  RMSE %.2f  precision(+/-1) %.3f", mae, rmse, prec)


if __name__ == "__main__":
    main()
