import os

import numpy as np
import torch
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
import tensorboard

import configurations
import settings

log_writer_train = SummaryWriter(os.path.join(configurations.logs_dir, "train"))
log_writer_val = SummaryWriter(os.path.join(configurations.logs_dir, "val"))


def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, features in enumerate(data_loader):
        images = features["image"]
        targets = features["classification"]
        images = images.to(settings.DEVICE)
        targets = targets.to(settings.DEVICE)

        logits, probas = model(images)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features in data_loader:
            images = features["image"]
            targets = features["classification"]
            images = images.to(settings.DEVICE)
            targets = targets.to(settings.DEVICE)
            logits, probas = model(images)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def run(model, train_loader, optimizer):
    start_time = time.time()
    for epoch in range(settings.num_epochs):

        model.train()
        for batch_idx, features in enumerate(train_loader):
            images = features["image"]
            targets = features["classification"]
            images = images.to(settings.DEVICE)
            targets = targets.to(settings.DEVICE)

            # show(torchvision.utils.make_grid(images))
            # FORWARD AND BACK PROP
            optimizer.zero_grad()
            logits, probas = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            # if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | Time: %.2f'
                  % (epoch + 1, settings.num_epochs, batch_idx,
                     len(train_loader), loss, (time.time() - start_time) / 60))
            log_writer_train.add_scalar("Loss_Epoch_" + str(epoch + 1), loss, batch_idx)

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%% | Loss: %.3f' % (
                epoch + 1, settings.num_epochs,
                compute_accuracy(model, train_loader),
                compute_epoch_loss(model, train_loader)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    torch.save(model, "trained_image_sentiment")


import matplotlib.pyplot as plt
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()