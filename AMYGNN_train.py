import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import  random_split
from dataset.generate_peptide_graph import peptide_to_graph
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score
from model.AmyGNN import AMYGNN
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import *

# AmyGNN总模型
read_peptide_file = r"./data\Amyloid_Database\CPAD2.0_Data\aggregating_peptides.xlsx"
afterpre_peptide_file = r'./data\after_process.csv'
pre_aaindex_file = r"./data\AAindex\aaindex1.txt"
afterpre_aaindex_file = r'./data\Amyloid_Database\AAIndex_data.xlsx'
peptide_pdb_file = r"./data\Amyloid_Database\PDB_Data"
AADist_file = r"./data\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt"

dataset = peptide_to_graph(pre_aaindex_file,afterpre_aaindex_file,afterpre_peptide_file,AADist_file)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data,val_data,test_data = random_split(dataset,[train_size,val_size,test_size])
train_loader = DataLoader(train_data,batch_size = 128,shuffle=True)
val_loader = DataLoader(val_data,batch_size = 64,shuffle = True)
test_data_loader = DataLoader(test_data,batch_size = test_size,shuffle=True)
# print(train_data[0],val_data[0],test_data[0])
# print(len(train_loader))
# print(len(train_data))

#定义模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AMYGNN(dataset[0].num_node_features,hidden_channels = 256).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)

# viz = Visdom()
# viz.line([0.],[0.], win="train loss", opts=dict(title='train_loss'))
lossl,loss_l = [],[]
y_score,fprl,tprl = [],[],[]

#训练网络
def train():
    model.train()
    log_dir = "./Logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(300):
        loss_train,train_correct = 0,0
        optimizer.zero_grad()
        batch_ind=0
        for data in train_loader:
            batch_ind += 1
            out = model(data.x.to(device),data.edge_index.to(device),data.edge_attr.to(device),data.batch.to(device))
            probs = out.argmax(dim = 1)
            loss = F.nll_loss(out,data.y.to(device))
            # loss = torch.nn.CrossEntropyLoss()(out.to('cpu'), data.y)
            train_correct += int((probs == data.y.to(device)).sum())
            # train_acc = accuracy_score(probs.to('cpu'),data.y)
            writer.add_scalar('Train/Loss', loss_train, epoch * len(train_loader) + batch_ind)
            writer.add_scalar('Train/Accuracy', train_correct / len(train_loader), epoch * len(train_loader) + batch_ind)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        loss_train = loss_train / len(train_loader)
        train_acc = train_correct / len(train_data)

        lossl.append(loss_train)
        # viz.line([loss_train],[epoch],win="train loss",update = 'append')

        # 在验证集上进行评估
        model.eval()
        correct,loss_all = 0,0
        with torch.no_grad():
            # optimizer.zero_grad()
            for val_data in val_loader:
                pred = model(val_data.x.to(device),val_data.edge_index.to(device),val_data.edge_attr.to(device),val_data.batch.to(device))
                probs = pred.argmax(dim = 1)
                val_loss = F.nll_loss(pred, val_data.y.to(device))
                # val_loss = torch.nn.CrossEntropyLoss()(pred.to('cpu'),val_data.y)
                correct += int((probs == val_data.y.to(device)).sum())
                val_acc = accuracy_score(val_data.y,probs.cpu().detach().numpy())
                # val_loss.backward()
                # optimizer.step()
                loss_all += val_loss.item()
            loss_all = loss_all / len(val_loader)
            writer.add_scalar('Val/Loss', loss_all, epoch * len(train_loader) + batch_ind)
            writer.add_scalar('Val/Accuracy', val_acc, epoch * len(train_loader) + batch_ind)

        if epoch == 299:
            torch.save(model.state_dict(), './trained_models/model_nosmo_1.pt')
        # print(loss_l)
        # val_acc = correct / len

        # model.eval()
        # correct = 0
        # for data in test_loader:
        #     out = model(data.x.to(device),data.edge_index.to(device),data.edge_attr.to(device),data.batch.to(device))
        #     probs = out.argmax(dim = 1)
        #     # print(data.y,probs)
        #     correct += int((probs.to('cpu') == data.y).sum())
        #     acc = accuracy_score(probs.to('cpu'),data.y)
        #     f1 = f1_score(data.y,probs.to('cpu'))
        #     mcc = matthews_corrcoef(data.y,probs.to('cpu'))

        #在单独的数据集上进行验证

        print('Epoch: {:03d}, Train Loss: {:.4f},Val Loss: {:.4f},Train Accuracy : {:.4f},Val Accuracy: {:.4f}'.format(epoch, loss_train, loss_all, train_acc, val_acc))
        # if epoch % 5 == 0:
        # torch.save(model,'D:\python代码练习\\new_model\\'  + str(epoch) + '_' + str(round(train_acc,4)) + '_' + str(round(val_acc,4)) + '.pt')
        # if acc >= 0.83 and mcc >= 0.66 and val_acc > 0.9:
        #     torch.save(model, 'D:\python代码练习\Model\\paper' + '_' + str(epoch) + '_' + str(round(train_acc,5)) + '_' + str(round(val_acc,5)) + '_' + str(round(acc,5)) + '_' + str(round(mcc,5)) + '.pt')
        # if acc_test >= 0.9 and mcc_test >= 0.9:
        #     torch.save(model, 'D:\python代码练习\Model\\paper' + '_' + str(epoch) + '_' + str(round(acc_test,5)) + '_' + str(round(mcc_test,5)) + '.pt')
        # print('Epoch: {:03d}, Train Loss: {:.4f},Val Loss: {:.4f},Train Accuracy : {:.4f},Val Accuracy: {:.4f},Test Acuuracy:{:.4f},Test MCC:{:.4f},Test F1score:{:.4f}'.format(epoch,loss_train,loss_all,train_acc, val_acc,acc,mcc,f1))
        writer.close()

def model_test(test_loader):
    model.eval()
    loss = 0
    for test_data in test_loader:
        output = model(test_data.x.to(device),test_data.edge_index.to(device),test_data.edge_attr.to(device),test_data.batch.to(device))
        classifier = output.argmax(dim = 1)
        loss_test = F.nll_loss(output,test_data.y.to(device))
        acc_test = accuracy_score(classifier.to('cpu'),test_data.y)
        cm = confusion_matrix(test_data.y, classifier.to('cpu'))
        for i in range(len(classifier)):
            prob = output[i][1].cpu().detach().numpy()
            y_score.append(prob)
        f1 = f1_score(test_data.y,classifier.to('cpu'))
        fpr, tpr, thresholds = roc_curve(test_data.y, y_score, pos_label=1)
        fprl.append([f for f in fpr])
        tprl.append([f for f in tpr])
        auroc = auc(fpr,tpr)
        y_score.clear()
        mcc = matthews_corrcoef(test_data.y,classifier.to('cpu'))
        loss += loss_test
    losstest = loss / len(test_loader)
    print("Test set results:",
              "loss= {:.4f}".format(losstest),
              "accuracy= {:.4f}".format(acc_test),
              "f1-score= {:.4f}".format(f1),
              "auroc= {:.4f}".format(auroc),
              "mcc= {:.4f}".format(mcc),
              cm,
    )
    return fprl,tprl,auroc,acc_test

# train()
torch.cuda.empty_cache()
train()

fpr, tpr, auroc, acc = model_test(test_data_loader)

