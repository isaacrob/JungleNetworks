import torch
import progressbar
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def correct_count(pred, target):
    _, pred_class = torch.max(pred, 1)
    return torch.sum(pred_class == target).float()

def generate_text_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def train_one_epoch_text(model, data, crit, optimizer, batch_size = 256, is_classifier = True):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_text_batch)

    model.train()

    n_samples = 0.0
    n_correct = 0.0
    epoch_loss = 0.0

    for data, offsets, target in progressbar.progressbar(dataloader):
        data = data.to(device)
        target = target.to(device)
        offsets = offsets.to(device)

        print(data.shape)
        optimizer.zero_grad()
        pred = model(data, offsets)
        print(target.shape)
        print(pred.shape)
        loss = crit(pred, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if is_classifier:
            n_samples += data.shape[0]
            n_correct += correct_count(pred, target).item()

    epoch_loss /= len(dataloader)
    
    if is_classifier:
        acc = n_correct / n_samples
        return epoch_loss, acc
    else:
        return epoch_loss

def train_one_epoch(model, dataloader, crit, optimizer, is_classifier = True):
    model.train()

    n_samples = 0.0
    n_correct = 0.0
    epoch_loss = 0.0

    for data, target in progressbar.progressbar(dataloader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = crit(pred, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if is_classifier:
            n_samples += data.shape[0]
            n_correct += correct_count(pred, target).item()

    epoch_loss /= len(dataloader)
    
    if is_classifier:
        acc = n_correct / n_samples
        return epoch_loss, acc
    else:
        return epoch_loss

def evaluate(model, dataloader, crit, is_classifier = True):
    model.eval()

    n_samples = 0.0
    n_correct = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            total_loss += crit(pred, target).item()
            if is_classifier:
                n_samples += data.shape[0]
                n_correct += correct_count(pred, target).item()

    total_loss /= len(dataloader)

    if is_classifier:
        acc = n_correct / n_samples
        return total_loss, acc
    else:
        return total_loss