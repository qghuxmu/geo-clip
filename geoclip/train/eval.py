import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic as GD
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from geoclip import GeoCLIP


class GeoDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.targets = self.df[["LAT", "LON"]].values.astype(np.float32)

        self.img_paths = self.df["IMG_ID"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
            img = img.squeeze(0)

        target = self.targets[idx]
        return img, target

def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            # imgs = model.image_encoder.preprocess_image(imgs).to(device)
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1] # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results


if __name__ == "__main__":

    # model
    model = GeoCLIP()

    csv_path = "/home/hqg/data/datasets/im2gps3ktest/im2gps3k_places365.csv"
    img_dir = "/home/hqg/data/datasets/im2gps3ktest"

    # Dataset & DataLoader
    dataset = GeoDataset(csv_path, img_dir, transform=model.image_encoder.preprocess_image)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Eval
    results = eval_images(val_dataloader, model, device)
    print(results)
