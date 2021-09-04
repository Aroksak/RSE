import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_optimizer import RAdam
from tqdm.auto import tqdm

from rse import ResidualShuffleExchangeNetwork


class SeqModel(nn.Module):
    def __init__(self, m, n_classes, n_blocks=1):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, m)
        self.rse = ResidualShuffleExchangeNetwork(m, n_blocks=n_blocks)
        self.linear = nn.Linear(m, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.rse(x)
        x = self.linear(x)
        return x


def plot_training_stats(losses, accuracies):
    fig, ax1 = plt.subplots()
    fig.set_facecolor("white")
    ax2 = ax1.twinx()
    ax1.plot(losses, 'r-')
    ax2.plot(accuracies, 'b-')
    ax1.set_title("Training stats")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    plt.show()


def training_loop(model, batch_generator, device):
    model.to(device)
    optimizer = RAdam(model.parameters(), lr=0.000883883)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    max_n = 6  # train on sequences up to 2**6=64 items long
    batch_size = 8

    t = tqdm(range(50_000), desc="Training. Loss: , Acc: ")

    losses = []
    accuracies = []

    for _ in t:
        optimizer.zero_grad()
        n = np.random.randint(2, max_n + 1)
        out_length = 1 << n
        inp, res = batch_generator(out_length, batch_size)
        output = model(inp.to(device))
        flat_output = torch.flatten(output, end_dim=1)
        flat_res = torch.flatten(res).to(device)
        loss = loss_fn(flat_output, flat_res)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(flat_output, dim=-1)
        acc = (preds == flat_res).sum() / len(flat_res)
        losses.append(loss.item())
        accuracies.append(acc.item())
        t.set_description(f"Training. Loss: {np.mean(losses[-100:]):.2f}, Acc: {np.mean(accuracies[-100:]): .2%}")
        if np.mean(accuracies[-500:]) > 0.9999:
            break

    plot_training_stats(losses, accuracies)


def plot_test_errplot(max_out_lengths, accuracy_medians, accuracy_errs):
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    ax.errorbar(np.arange(len(max_out_lengths)), accuracy_medians, yerr=accuracy_errs, fmt='o-')
    ax.xaxis.set_ticks(np.arange(len(max_out_lengths)))
    ax.xaxis.set_ticklabels(max_out_lengths)
    ax.set_title("Testing generalization to different seq lenghts")
    ax.set_xlabel("Seq length")
    ax.set_ylabel("Accuracy")
    plt.show()


def test_lengths(model, batch_generator, device):
    model.to(device)
    model.eval()
    max_out_lengths = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 8
    examples_per_length = 100

    accuracy_medians = []
    accuracy_errs = [[],[]]

    for length in tqdm(max_out_lengths, desc="Testing."):
        accuracies = []
        for _ in range(examples_per_length):
            with torch.no_grad():
                inp, res = batch_generator(length, batch_size)
                output = model(inp.to(device))
                flat_output = torch.flatten(output, end_dim=1)
                flat_res = torch.flatten(res).to(device)
                preds = torch.argmax(flat_output, dim=-1)
                acc = (preds == flat_res).sum() / len(flat_res)
                accuracies.append(acc.item())
        median = np.median(accuracies)
        lower_error = median - np.quantile(accuracies, 0.1)
        upper_error = np.quantile(accuracies, 0.9) - median
        accuracy_medians.append(median)
        accuracy_errs[0].append(lower_error)
        accuracy_errs[1].append(upper_error)

    plot_test_errplot(max_out_lengths, accuracy_medians, accuracy_errs)