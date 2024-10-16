import torch
import torch.nn.functional as F
from sklearn.metrics import *
from torch_geometric.loader import DataLoader

from model.AmyGNN import AMYGNN


def model_test(test_loader):
    model.eval()
    loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for test_data in test_loader:
        output = model(test_data.x.float().to(device),
                       test_data.edge_index.to(device),
                       test_data.edge_attr.float().to(device),
                       test_data.batch.to(device)
                       )
        classifier = output.argmax(dim=1)
        loss_test = F.nll_loss(output, test_data.y.to(device))
        acc_test = accuracy_score(classifier.to('cpu'), test_data.y)
        cm = confusion_matrix(test_data.y, classifier.to('cpu'))
        for i in range(len(classifier)):
            prob = output[i][1].cpu().detach().numpy()
            y_score.append(prob)
        f1 = f1_score(test_data.y, classifier.to('cpu'))
        fpr, tpr, thresholds = roc_curve(test_data.y, y_score, pos_label=1)
        fprl.append([f for f in fpr])
        tprl.append([f for f in tpr])
        auroc = auc(fpr, tpr)
        y_score.clear()
        mcc = matthews_corrcoef(test_data.y, classifier.to('cpu'))
        loss += loss_test
    losstest = loss / len(test_loader)
    print("Test set results:",
          "loss= {:.4f}".format(losstest),
          "accuracy= {:.4f}".format(acc_test),
          "f1-score= {:.4f}".format(f1),
          "auroc= {:.4f}".format(auroc),
          "mcc= {:.4f}".format(mcc),
          cm
          )
    return fprl, tprl, auroc, acc_test


if __name__ == '__main__':
    # 模型测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # afterpre_peptide_file = r'./data\after_process.csv'
    # pre_aaindex_file = r"./data\AAindex\aaindex1.txt"
    # afterpre_aaindex_file = r'./data\Amyloid_Database\AAIndex_data.xlsx'
    # peptide_pdb_file = r"./data\Amyloid_Database\PDB_Data"
    # AADist_file = r"./data\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt"

    test_dataset = torch.load('./data/processed_dataset/test_dataset.pkl')
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    # train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = AMYGNN(input_channels=534, hidden_channels=64)  # 创建模型实例
    model = torch.load('./trained_models/Threelayers_model.pt')

    model.eval()
    model.to(device)
    y_score = []
    fprl, tprl = [], []
    fpr, tpr, auroc, acc = model_test(test_data_loader)
