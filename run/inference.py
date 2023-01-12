import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from data_loader import CarsDatasetAdaptor, get_valid_transforms
from model import EfficientDetModel
from utils import draw_pascal_voc_bboxes


def compare_bboxes_for_image(
    image,
    predicted_bboxes,
    actual_bboxes,
    draw_bboxes_fn,
    figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    plt.show()


def main():
    model_save_dir = "./saved/models/"

    model = EfficientDetModel(
        num_classes=1,
        img_size=512,
        inference_transforms=get_valid_transforms(target_img_size=512),
    )

    model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "trained_effdet.ckpt"))
    )
    model.eval()

    dataset_path = Path("./data/CarObjectDetection")
    train_data_path = dataset_path / "training_images"
    df = pd.read_csv(dataset_path / "train_solution_bounding_boxes (1).csv")

    cars_train_ds = CarsDatasetAdaptor(train_data_path, df)
    image1, truth_bboxes1, _, _ = cars_train_ds.get_image_and_labels_by_idx(0)
    image2, truth_bboxes2, _, _ = cars_train_ds.get_image_and_labels_by_idx(1)

    images = [image1, image2]

    predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)

    compare_bboxes_for_image(
        image1,
        predicted_bboxes=predicted_bboxes[0],
        actual_bboxes=truth_bboxes1.tolist(),
        draw_bboxes_fn=draw_pascal_voc_bboxes,
    )
    compare_bboxes_for_image(
        image2,
        predicted_bboxes=predicted_bboxes[1],
        actual_bboxes=truth_bboxes2.tolist(),
        draw_bboxes_fn=draw_pascal_voc_bboxes,
    )


main()
