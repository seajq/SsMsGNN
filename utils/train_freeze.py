import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN!")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf!")
def get_metrics(y_pred, y_true):
    y_pred = y_pred.argmax(1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    return macro_precision, macro_recall, macro_f1


def val(model, dataloader, device):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    all_loss = 0
    test_logits = []
    test_label = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device).long()
        with torch.no_grad():
            logits = model(x, pretrain=False)
            loss = loss_func(logits, y)
            all_loss += loss.item()
            test_logits.append(logits)
            test_label.append(y)

    test_logits = torch.cat(test_logits, dim=0)
    test_label = torch.cat(test_label, dim=0)
    avg_loss = all_loss / len(dataloader.dataset)

    macro_precision, macro_recall, macro_f1 = get_metrics(test_logits, test_label)
    return avg_loss, macro_precision, macro_recall, macro_f1


def train(model, optim, args, pretrain_loader, finetune_loader, valid_loader):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    Rebuild_Func = nn.MSELoss()
    Classify_Func = nn.CrossEntropyLoss()

    optimizer = optim

    train_loss_list = []
    val_loss_list = []
    device = args.device
    early_stop_win = 10
    stop_improve_count = 0

    for phase in ['pretrain', 'fine_tune']:

        min_loss = 1e+8

        if phase == 'pretrain':
            for param in model.classifier.parameters():
                param.requires_grad = False

            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True

        if phase == 'fine_tune':
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True

        for i_epoch in range(1, args.pre_epoch + 1) if phase == 'pretrain' else range(1, args.tune_epoch + 1):
            all_loss = 0
            model.train()

            for x, y in pretrain_loader if phase == 'pretrain' else finetune_loader:

                x, y = x.to(device), y.to(device).long() if args.task == 'clf' else y.to(device)
                optimizer.zero_grad()
                # check_nan_inf(x, "Input")
                if phase == 'pretrain':
                    pred = model(x, pretrain=True)
                    gt = x[:, -1, :].to(device)
                    loss = Rebuild_Func(pred, gt)
                else:
                    logits = model(x, pretrain=False)
                    loss = Classify_Func(logits, y)
                # check_nan_inf(loss, "Loss before backward")
                loss.backward()
                optimizer.step()
                all_loss += loss.item()

            if phase == 'pretrain':
                train_loss_list.append(all_loss / len(pretrain_loader.dataset))
                print(f'[Pretrain Epoch {i_epoch}/{args.pre_epoch}] Pretrain Loss: {all_loss / len(pretrain_loader.dataset):.8f}')
            elif phase == 'fine_tune':
                train_loss_list.append(all_loss / len(finetune_loader.dataset))
                print(f'[Fine-tune Epoch {i_epoch}/{args.tune_epoch}] Fine-tune Loss: {all_loss / len(finetune_loader.dataset):.8f}')
            if valid_loader is not None:
                val_loss, macro_precision, macro_recall, macro_f1 = val(model, valid_loader, device)
                val_loss_list.append(val_loss)

            if all_loss < min_loss:
                torch.save(model, args.save)
                min_loss = all_loss

    return train_loss_list, val_loss_list
