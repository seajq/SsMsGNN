import torch, argparse, os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support


def get_metrics(y_pred, y_true):
    y_pred = y_pred.argmax(1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()

    return macro_precision, macro_recall, macro_f1


def test(model, dataloader, device, save):
    model.eval()
    test_pred = []
    test_ground = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device).long()

        with torch.no_grad():
            logits = model(x, pretrain=False)

            test_pred.append(logits)
            test_ground.append(y)

    test_pred = torch.cat(test_pred, dim=0)
    test_ground = torch.cat(test_ground, dim=0)

    pred = test_pred.argmax(1).cpu().numpy()
    ground = test_ground.cpu().numpy()

    cm = metrics.confusion_matrix(ground, pred)
    Total_acc = cm.diagonal().sum() / cm.sum()
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    # report = classification_report(ground, pred, digits=4)
    # print(report)

    macro_precision, macro_recall, macro_f1 = get_metrics(test_pred, test_ground)
    os.rename(save, os.path.join(os.path.dirname(save), f"best{int(macro_f1*1000)}.pt"))

    print(f'| Test Precision: {macro_precision:.4f} | Test Recall: {macro_recall:.4f} | Test F1-Score: {macro_f1:.4f}')
    print(f'{macro_precision:.4f} {macro_recall:.4f} {macro_f1:.4f}')

    png_save = f"./respic/Res{macro_f1}.png"

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracy_per_class)), accuracy_per_class, color='skyblue')
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(accuracy_per_class)), [f'{i}' for i in range(len(accuracy_per_class))])
    plt.savefig(png_save)

    return macro_precision, macro_recall, macro_f1
