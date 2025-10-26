import torch
import numpy as np
import torch.nn as nn

def loss_batch(model, loss_func, xb, yb, opt=None):
    # Forward pass
    xb_out = model(xb)  # logits, shape (batch, num_classes)
    loss = loss_func(xb_out, yb.float())  # yb must be float for BCEWithLogitsLoss

    # Backprop
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Compute accuracy (elementwise match)
    preds = (torch.sigmoid(xb_out) > 0.5).int()
    correct = (preds == yb.int()).sum().item()
    acc = correct / yb.numel()

    return loss.item(), acc, len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    model.train()
    losses, accs, ns = [], [], []
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        loss, acc, n = loss_batch(model, loss_func, xb, yb, opt=opt)
        losses.append(loss)
        accs.append(acc * n)
        ns.append(n)
    train_loss = np.average(losses, weights=ns)
    train_acc = np.sum(accs) / np.sum(ns)
    return train_loss, train_acc


def val_step(model, val_dl, loss_func, device):
    model.eval()
    with torch.no_grad():
        losses, accs, ns = [], [], []
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, acc, n = loss_batch(model, loss_func, xb, yb)
            losses.append(loss)
            accs.append(acc * n)
            ns.append(n)
    val_loss = np.average(losses, weights=ns)
    val_acc = np.sum(accs) / np.sum(ns)
    return val_loss, val_acc


def fit(epochs, model, loss_func, opt, train_dl, val_dl, device):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_step(model, train_dl, loss_func, device, opt)
        val_loss, val_acc = val_step(model, val_dl, loss_func, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} | "
              f"train loss: {train_loss:.3f} acc: {train_acc:.3f} | "
              f"val loss: {val_loss:.3f} acc: {val_acc:.3f}")

    return history


def run_model(train_dl, val_dl, model, device, lr=1e-3, epochs=50, opt=None):
    optimizer = opt or torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()
    history = fit(epochs, model, loss_func, optimizer, train_dl, val_dl, device)
    return history
