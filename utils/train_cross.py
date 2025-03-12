import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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


def train(model, optim, args, pretrain_loader, finetune_loader, valid_loader=None):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    Rebuild_Func = nn.MSELoss()
    Classify_Func = nn.CrossEntropyLoss()
    optimizer = optim

    rebuild_list = []
    classify_list = []
    val_list = []

    device = args.device
    min_loss = float('inf')
    penalty = float(1)
    max_f1 = 0
    for i_epoch in range(1, args.epoch + 1):
        model.train()
        Rebulid_Loss = 0
        Classify_Loss = 0

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.rb_lr

        for x, y in pretrain_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x, pretrain=True)
            gt = x[:, -1, :].to(device)
            loss = Rebuild_Func(pred, gt)

            loss.backward()
            optimizer.step()

            Rebulid_Loss += loss.item()

        rebuild_list.append(Rebulid_Loss / len(pretrain_loader.dataset))
        # print(f'[Reconstruct Epoch {i_epoch}/{args.epoch}] Rebuild Loss: {Rebulid_Loss / len(pretrain_loader):.8f}')
        # if i_epoch != 1:
        #     val_loss, macro_precision, macro_recall, macro_f1 = val(model, valid_loader, device)
        #     _f1 = macro_f1 - bef_f1
        #     if _f1 < 0:
        #         return np.abs(_f1) + 1
        #     else:
        #         return 1 / (_f1 + 1e-6)

        for param_group in optimizer.param_groups:
            param_group['lr'] = optimizer.defaults['lr']

        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()

            logits = model(x, pretrain=False)
            loss = Classify_Func(logits, y)
            # loss = penalty * Classify_Func(logits, y)

            loss.backward()
            optimizer.step()

            Classify_Loss += loss.item()

        classify_list.append(Classify_Loss / len(finetune_loader.dataset))

        if valid_loader is not None:
            val_loss, macro_precision, macro_recall, macro_f1 = val(model, valid_loader, device)
            val_list.append(val_loss)
            # bef_f1 = macro_f1
            print(
                f'[Classify Epoch {i_epoch}/{args.epoch}] | Train Loss:{Classify_Loss / len(finetune_loader.dataset):.8f} | Val Loss:{val_loss:.8f} | Val Precision: {macro_precision:.4f} | Val Recall: {macro_recall:.4f} | Val F1-Score: {macro_f1:.4f}')
            if val_loss < min_loss or macro_f1 > max_f1:
                torch.save(model, args.save)
                min_loss = val_loss
                max_f1 = macro_f1
        else:
            val_loss, macro_precision, macro_recall, macro_f1 = val(model, finetune_loader, device)
            if Classify_Loss < min_loss:
                torch.save(model, args.save)
                min_loss = Classify_Loss
            print(
                f'[Classify Epoch {i_epoch}/{args.epoch}] Train Loss: {Classify_Loss / len(finetune_loader.dataset):.8f} | Train Loss:{val_loss:.8f} | Train Precision: {macro_precision:.4f} | Train Recall: {macro_recall:.4f} | Train F1-Score: {macro_f1:.4f}')


    return np.array(classify_list), np.array(val_list)
