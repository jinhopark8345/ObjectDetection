import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from data_loader import CarsDatasetAdaptor, EfficientDetDataModule, get_valid_transforms
from model import EfficientDetModel


def main():
    dataset_path = Path("./data/CarObjectDetection")
    # list(dataset_path.iterdir())
    train_data_path = dataset_path / "training_images"
    df = pd.read_csv(dataset_path / "train_solution_bounding_boxes (1).csv")

    cars_train_ds = CarsDatasetAdaptor(train_data_path, df)

    dm = EfficientDetDataModule(
        train_dataset_adaptor=cars_train_ds,
        validation_dataset_adaptor=cars_train_ds,
        num_workers=4,
        batch_size=2,
    )

    model = EfficientDetModel(
        num_classes=1,
        img_size=512,
        inference_transforms=get_valid_transforms(target_img_size=512),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=5,
        num_sanity_val_steps=1,
    )
    trainer.fit(model, dm)

    model_save_dir = "./saved/models/"
    torch.save(model.state_dict(), os.path.join(model_save_dir, "trained_effdet.ckpt"))


main()
