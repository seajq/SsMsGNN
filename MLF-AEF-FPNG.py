import argparse, datetime, os
import torch.optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from utils.AEDataset import AEDataset
from utils.train_freeze import train
from utils.test_freeze import test
from model.CAE import FreezeAE, FPNEncoder, FPNDecoder, FPNClassifier


def get_save_path(save_path):
    if not os.path.isdir(save_path):
        dir_path = os.path.dirname(save_path)
    else:
        dir_path = save_path

    now = datetime.datetime.now()
    datestr = now.strftime('%m|%d-%H:%M:%S')

    paths = f'{dir_path}/best_{datestr}.pt'

    dirname = os.path.dirname(paths)
    os.makedirs(dirname, exist_ok=True)

    return paths


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, help='location of the data file', default='../Data/MF24')
parser.add_argument('--save', type=str, help='path to save the final model', default='save/MF24/MLF-AEF-FPNG')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--test_ratio', type=float, default=0.1)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--labeled_ratio', type=float, default=0.01)
parser.add_argument('--unlabeled_ratio', type=float, default=0.1)
parser.add_argument('--scale', type=int, default=1)

parser.add_argument('--gcn_depth', type=int, help='graph convolution depth', default=2)
parser.add_argument('--num_nodes', type=int, help='number of nodes/variables', default=24)
parser.add_argument('--dropout', type=float, help='dropout rate', default=0.3)
parser.add_argument('--subgraph_size', type=int, help='k', default=7)
parser.add_argument('--node_dim', type=int, help='dim of nodes', default=40)
parser.add_argument('--in_channels', type=int, help='inputs dimension', default=1)
parser.add_argument('--conv_channels', type=int, help='convolution channels', default=16)
parser.add_argument('--end_channels', type=int, help='end channels', default=64)
parser.add_argument('--out_channels', type=int, help='output channels', default=1)
parser.add_argument('--seq_length', type=int, help='input sequence length', default=120)
parser.add_argument('--stride', type=int, help='stride', default=1)

parser.add_argument('--kernel_set', nargs='+', type=int, default=[7, 6, 3])

parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
parser.add_argument('--decay', type=float, help='weight decay rate', default=1e-5)
parser.add_argument('--propalpha', type=float, help='prop alpha', default=0.05)
parser.add_argument('--alpha', type=float, help='prop alpha', default=3)
parser.add_argument('--pre_epoch', type=int, help='', default=30)
parser.add_argument('--tune_epoch', type=int, help='', default=70)
parser.add_argument('--interval', type=int, help='', default=2)

parser.add_argument('--task', type=str, help='step size for scheduler', default='clf')
parser.add_argument('--class_num', type=int, help='step size for scheduler', default=7)
parser.add_argument('--random_seed', type=int, help='random_seed', default=42)
parser.add_argument('--device', type=str, help='', default='cuda:1')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)
args.save = get_save_path(args.save)


#  python MLF-AEF-FPNG.py --valid_ratio 0.3 --device cuda:1
#  python MLF-AEF-FPNG.py --valid_ratio 0.5 --device cuda:2

def run(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    LabeledData = AEDataset(args.data, 'train_labeled', args.train_ratio, args.test_ratio, args.valid_ratio, args.labeled_ratio, args.unlabeled_ratio, args.seq_length, args.stride, args.scale)
    UnLabeledData = AEDataset(args.data, 'train_unlabeled', args.train_ratio, args.test_ratio, args.valid_ratio, args.labeled_ratio, args.unlabeled_ratio, args.seq_length, args.stride, args.scale)
    ValidData = AEDataset(args.data, 'valid', args.train_ratio, args.test_ratio, args.valid_ratio, args.labeled_ratio, args.unlabeled_ratio, args.seq_length, args.stride, args.scale)
    testData = AEDataset(args.data, 'test', args.train_ratio, args.test_ratio, args.valid_ratio, args.labeled_ratio, args.unlabeled_ratio, args.seq_length, args.stride, args.scale)

    pretrain_loader = DataLoader(ConcatDataset([LabeledData, UnLabeledData]), batch_size=args.batch_size, shuffle=True)
    finetune_loader = DataLoader(LabeledData, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(ValidData, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testData, batch_size=args.batch_size, shuffle=True)

    model = FreezeAE(Encoder=FPNEncoder(gcn_depth=args.gcn_depth, num_nodes=args.num_nodes, device=args.device, kernel_set=args.kernel_set, dropout=args.dropout, subgraph_size=args.subgraph_size,
                                        node_dim=args.node_dim, seq_length=args.seq_length, in_channels=args.in_channels, propalpha=args.propalpha, alpha=args.alpha, class_num=args.class_num,
                                        conv_channels=args.conv_channels, end_channels=args.end_channels, out_channels=1, ),
                     Decoder=FPNDecoder(gcn_depth=args.gcn_depth, num_nodes=args.num_nodes, device=args.device, kernel_set=args.kernel_set, dropout=args.dropout, subgraph_size=args.subgraph_size,
                                        node_dim=args.node_dim, seq_length=args.seq_length, in_channels=args.in_channels, propalpha=args.propalpha, alpha=args.alpha, class_num=args.class_num,
                                        conv_channels=args.conv_channels, end_channels=args.end_channels, out_channels=1, ),
                     Classifier=FPNClassifier(gcn_depth=args.gcn_depth, num_nodes=args.num_nodes, device=args.device, kernel_set=args.kernel_set, dropout=args.dropout, subgraph_size=args.subgraph_size,
                                              node_dim=args.node_dim, seq_length=args.seq_length, in_channels=args.in_channels, propalpha=args.propalpha, alpha=args.alpha, class_num=args.class_num,
                                              conv_channels=args.conv_channels, end_channels=args.end_channels, out_channels=1, )).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # Namespace
    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    train(model, optim, args, pretrain_loader, finetune_loader)

    best_model = torch.load(args.save).to(args.device)
    test(best_model, test_loader, args.device, args.save)


run(args)
