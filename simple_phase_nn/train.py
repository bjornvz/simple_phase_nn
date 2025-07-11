import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pprint
from torcheval.metrics.functional import r2_score

from simple_phase_nn.models import *
from simple_phase_nn.utils import *
from simple_phase_nn.datasets import *
from simple_phase_nn.plots import *

def train(cf):
    cf.logdir = create_path(cf.logdir, overwrite=False)
    print(f"\nTRAINING {cf.logdir}: --seed {cf.seed} --device {DEVICE}\n")
    pprint.pprint(cf)

    set_seeds(cf.seed)
    start_time = datetime.now()

    # LOAD DATASET
    train_dataset, val_dataset = XXZDataset(train=True, normalize=True), XXZDataset(train=False, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) # batch_size = len(val_dataset) is required to compute goodness_metric correclty

    criterion = nn.MSELoss()
    goodness_function = r2_score

    # Strings for plotting & logging  
    cf.goodness_str = "r2"
    cf.phase_indicator_str = r"${\partial\hat{\gamma}}/{\partial\gamma}$"

    net1 = Custom1DCNN().to(DEVICE) 

    cf.logdir = str(cf.logdir)
    save_json(cf, cf.logdir, "config.json")

    # OPTIMIZER
    optimizer = optim.Adam( list(net1.parameters()) , 
                           lr=cf.learning_rate , weight_decay=cf.weight_decay)
    
    early_stopper = EarlyStopper(patience=cf.patience, min_delta=0)

    metrics = {"train_loss": [], "train_l1": [], f"train_{cf.goodness_str}": [],
               "val_loss": [], "val_l1": [], f"val_{cf.goodness_str}": [], 
               "corr": [], "out0": [], "labels": [], "unique_labels" : [], "phase_indicator" : []}


    # TRAINING LOOP
    print(f"\nTraining for {cf.epochs} epochs with {len(train_loader)} training batches and {len(val_loader)} validation batches.\n")
    for epoch in tqdm(range(cf.epochs)):
        net1.train()
        # net2.train()

        total_loss = 0
        train_outputs, train_targets = [], [] # for goodness metric calculation

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, train_dataset.no_classes)
            optimizer.zero_grad()

            out = net1(x)

            loss = criterion( out, y )  
            
            total_loss += loss.item() 

            train_outputs.append(out.cpu())
            train_targets.append(y.cpu())

            loss.backward()
            optimizer.step()

        # Concatenate all outputs and targets for the whole epoch
        train_outputs = torch.cat(train_outputs, dim=0)
        train_targets = torch.cat(train_targets, dim=0)

        metrics["train_loss"].append( total_loss/len(train_loader) )      # loss is mean by default, so we need to divide by len(train_loader)
        metrics[f"train_{cf.goodness_str}"].append( goodness_function( train_outputs, train_targets ).item() )

        # VALIDATION
        net1.eval()
        total_loss = 0
        with torch.no_grad(): # this enables larger batch size for validation; less memory
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(DEVICE), y.to(DEVICE).view(-1, train_dataset.no_classes)

                out = net1(x)
                
                total_loss += criterion( out, y ).item() 

            
            # METRICS FOR EACH EPOCH
            # 1) Loss, l1, goodness metric
            metrics["val_loss"].append( total_loss/len(val_loader) ) 
            metrics[f"val_{cf.goodness_str}"].append( goodness_function( out, y ).item() )
            print(f"\nEPOCH {epoch+1}/{cf.epochs}, TRAIN loss: {metrics['train_loss'][-1]:.2e}, {cf.goodness_str}: {metrics[f'train_{cf.goodness_str}'][-1]:.2f}, VAL loss: {metrics['val_loss'][-1]:.2e}, {cf.goodness_str}: {metrics[f'val_{cf.goodness_str}'][-1]:.2f}")

            # 2) Phase indicator
            metrics["phase_indicator"].append( val_dataset.get_phase_indicator(out) ) 
            
            if early_stopper.early_stop(metrics["val_loss"][-1]):
                print(f"\nEarly stopping at epoch {epoch+1}")                                 
                break

    # METRICS FOR ONLY LAST EPOCH
    out = out.cpu()
    y = y.cpu()   
    if cf.normalize_labels: # we want to save unnormalized labels
        out[:,0] = denormalize01(out[:,0], val_dataset.y_min, val_dataset.y_max)
        y = denormalize01(y, val_dataset.y_min, val_dataset.y_max)
    metrics["out0"] = out[:,0].numpy().tolist()
    metrics["labels"] = y.numpy().tolist()


    metrics["unique_labels"] = val_dataset.unique_labels_unnormalized.tolist()
    metrics["train_time"] = str(datetime.now() - start_time)

    print(f"\nTraining time: {metrics['train_time']}")

    save_json(metrics, cf.logdir, "metrics.json")