import numpy as np
from Bio.PDB import *
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from model.AmyGNN import AMYGNN
from sklearn.metrics import *
import os

target_dict = {'Amyloid' : 1,'Non-amyloid' : 0}
def Elu_dist(coord_1,coord_2):
    return np.sqrt(sum((coord_1 - coord_2) ** 2))

def Cosine_dist(coords_1,coords_2):
    cosine_dist = np.dot(coords_1, coords_2) / (np.sqrt(sum((coords_1) ** 2)) * np.sqrt(sum((coords_2) ** 2)))
    return cosine_dist

def protein_to_coord(pdb_file_path,protein_file,dist_file):
    """从多肽得结构数据中提取CA原子得坐标"""
    proteinname_list = list(pd.read_excel(protein_file,header = [0])['Protein Name'])
    protein_position = pd.read_excel(protein_file,header = [0])['Position']
    sequence = pd.read_excel(protein_file,header = [0])['Sequence']
    coord_list,coord,elu_list,cosine_list,edge_source,edge_target = [],[],[],[],[],[]
    coord_list_dict = {}

    pdb_parser = PDBParser(QUIET=True)
    for j in range(0,len(proteinname_list)):
        protein_name = proteinname_list[j]
        position = protein_position[j]
        print(position)
        pdb_structure = pdb_parser.get_structure(protein_name,pdb_file_path + '\\' + protein_name + '.pdb')
        residues = list(pdb_structure.get_residues())
        # print(len(residues))
        if type(position) != float:
            start = int(position.split('_')[0])
            end = int(position.split('_')[1])
            if (protein_name == '8b3a' or protein_name == '6c51'):
                for i in range(start, end + 1):
                    atoms = residues[i].get_atoms()
                    for atom in atoms:
                        if atom.get_fullname().strip(' ') == 'CA':
                            atom_coord = atom.get_coord()
                            coord_list.append(atom_coord)
            else:
                for i in range(start - 1,end):
                    atoms = residues[i].get_atoms()
                    for atom in atoms:
                        if atom.get_fullname().strip(' ') == 'CA':
                            atom_coord = atom.get_coord()
                            coord_list.append(atom_coord)
        else:
            pass
        with open(dist_file,'a+') as f:
            for m in range(0,len(coord_list)):
                for n in range(0,len(coord_list)):
                    elu_list.append(Elu_dist(coord_list[m],coord_list[n]))
                    cosine_list.append(Cosine_dist(coord_list[m],coord_list[n]))
                    if Elu_dist(coord_list[m],coord_list[n]) != 0.0 and Elu_dist(coord_list[m],coord_list[n]) <= 5.0:
                        edge_source.append(m)
                        edge_target.append(n)
            f.write(protein_name + 'Edge_Source:' + str(edge_source) + '\n')
            f.write(protein_name + 'Edge_Target:' + str(edge_target) + '\n')

            f.write(protein_name + 'Elu_dist:' + str(elu_list) + '\n')
            f.write(protein_name + 'Cosine_Dist:' + str(cosine_list) + '\n')
            f.write(protein_name + 'CA_Coord:' + str(coord_list) + '\n')

        f.close()

        coord_list_dict = dict([(protein_name + '_' + 'elu',elu_list),(protein_name + '_' + 'cosine',cosine_list)])

        # df = pd.DataFrame({'PDB_ID':pdb_id,'Elu_Dist' : elu_list,'Cosine_Dist' : cosine_list})
        # df.to_csv(file_path,index = True,sep = ',')
        print(protein_name, len(coord_list), [k for k in coord_list_dict.keys()],
              len([v for v in coord_list_dict.values()][0]))

        coord_list.clear()
        elu_list.clear()
        cosine_list.clear()
        edge_source.clear()
        edge_target.clear()

def extract_node_feature(file):
    """从处理之后的AAIndex数据中提取到我们要使用的特征"""
    feature_data = pd.read_excel(file,header = [0])
    feature_data.set_index("ID", inplace=True)
    feature_data.drop('Unnamed: 0', axis=1, inplace=True)

    with open(r"D:\python代码练习\iFeature\data\AAindex.txt") as file:
        data = file.readlines()[1:]
        feature_list = []
        for feature_name in data:
            feature_list.append(feature_data.loc[feature_name.strip('\n').split('\t')[0]])
        file.close()
    return feature_list

def get_coord(file,protein_file):
    protein_data = pd.read_excel(protein_file,header = [0])
    cl,coord_list = [],[]
    with open(file,'r') as f:
        records = f.readlines()
        Coord = records[4::5]
        f.close()

    for i in range(len(Coord)):
        coord = Coord[i].split('CA_Coord:')[1].strip('[\n')[:-1].split('),')
        # print(len(coord))
        for j in range(len(coord)):
            c = coord[j].strip(' ').split(', dtype=')[0].strip('array(').strip('[]').strip(' ').split(',')
            for f in c:
                cl.append(f.strip(' '))
        coord_list.append([l for l in cl])
        cl.clear()

    return coord_list

def generate_edge(file,protein_file):
    peptide_data = pd.read_excel(protein_file,header = [0])
    distance,elu_dist,elu_distance,dpc_list,edge_source,edge_target,edge_attr = [],[],[],[],[],[],[]
    edge_index = [[],[]]
    edge_index_save,edge_attr_save = [],[]

    Dpc = pd.read_csv(r"E:\paper_datasets\CaseStudy\dpc1.csv", header=[0],
                      delimiter=',')
    Dpc.rename(columns={'Unnamed: 0': 'Entry'}, inplace=True)

    with open(file,'r') as f:
        records = f.readlines()
        Edge_Source = records[0::5]
        Edge_Target = records[1::5]
        Elu_dist = records[2::5]
        f.close()
    for i in range(len(peptide_data['Protein Name'])):
        sequence = peptide_data['Sequence']
        edge_source = Edge_Source[i].split('Edge_Source:')[1].strip('[').strip('\n')[:-1].split(',')
        edge_target = Edge_Target[i].split('Edge_Target:')[1].strip('[').strip('\n')[:-1].split(',')
        elu_dist = Elu_dist[i].split('Elu_dist:')[1].strip('[').strip('\n')[:-1].split(',')
        if edge_source != ['']:
            for e1 in edge_source:
                e1 = int(e1.strip(' '))
                edge_index[0].append(e1)
        if edge_target != ['']:
            for e2 in edge_target:
                e2 = int(e2.strip(' '))
                edge_index[1].append(e2)
        edge_index_tensor = torch.tensor(edge_index,dtype = torch.long)
        for j in range(len(edge_index[0])):
            aa1 = sequence[i][edge_index[0][j]]
            aa2 = sequence[i][edge_index[1][j]]
            index_name = 'DPC_' + aa1 + aa2
            dpc_list.append(Dpc[index_name][i])
            # print(str(i) + '_' + str(j) + ':_' + aa1 + aa2)

        edge_attr.append(dpc_list)
        edge_attr_np = np.array(edge_attr).transpose()
        edge_attr_tensor = torch.tensor(edge_attr_np, dtype=torch.float32)
        # print(edge_attr_np)
        # edge_attr.append(elu_distance)
        edge_attr[0].clear()
        edge_attr.clear()
        edge_index[0].clear()
        edge_index[1].clear()
        distance.clear()
        elu_distance.clear()
        # edge_index.clear()
        # print(peptide_data['Entry'][i],edge_index_tensor,edge_index_tensor.shape, edge_attr_tensor.shape)
        edge_index_save.append(edge_index_tensor)
        edge_attr_save.append(edge_attr_tensor)

    return edge_index_save,edge_attr_save

def get_class(peptide_data):
    class_info = []
    for i in range(0,len(peptide_data['Protein Name'])):
        classification = peptide_data['Classification'][i]
        class_info.append(target_dict[classification])
        y_label = torch.tensor(class_info,dtype = torch.float)
    return y_label

def protein_to_graph(protein_file,dist_file,feature_file):
    """构建图数据"""
    peptide_data = pd.read_excel(protein_file,header = [0])
    coord_list = get_coord(dist_file,protein_file)
    edge_index_list = generate_edge(dist_file,protein_file)[0]
    edge_attr_list = generate_edge(dist_file,protein_file)[1]

    amino_list,fd,feature = [],[],[]
    peptide_graph_dataset = []
    feature_list = extract_node_feature(feature_file)
    y_label = get_class(peptide_data)

    for i in range(0,len(peptide_data['Protein Name'])):
        sequence = peptide_data['Sequence']
        # index = peptide_data['Entry']
        for s in sequence[i].strip(' '):
            amino_list.append(s)
            for f in feature_list:
                feature.append(f[s.strip(' ')])
        length = len(sequence[i])
        coord = coord_list[i]
        # print(len(coord),len(sequence[i]))
        for j in range(0,length):
            fd.append(list(feature[len(feature_list) * j : len(feature_list) * (j + 1)]))
            fd[j].append(coord[0 + j * 3])
            fd[j].append(coord[1 + j * 3])
            fd[j].append(coord[2 + j * 3])

        # fd_np = np.array(fd)
        # print(peptide_data['Entry'][i],fd_np.shape)
        # print(type(fd))
        x = torch.tensor(np.array(fd,dtype = float),dtype = torch.float)
        # x = torch.where(torch.isnan(x).any(), torch.full_like(x, 0), x)
        # print(x[0])

        edge_index_torch = to_undirected(edge_index_list[i])

        y = torch.tensor(np.array(y_label[i]),dtype = torch.long)
        # print(torch.isnan(y).any())
        peptide_graph = Data(x = x,edge_index = edge_index_torch,edge_attr = edge_attr_list[i],y = y)
        peptide_graph_dataset.append(peptide_graph)
        amino_list.clear()
        feature.clear()
        fd.clear()

    return peptide_graph_dataset

pdb_file = r'E:\paper_datasets\CaseStudy'
case_file = r'E:\paper_datasets\CaseStudy\case_data.xlsx'
distance_file = r'E:\paper_datasets\CaseStudy\Distance_new.txt'
feature_file = r'E:\paper_datasets\Amyloid_Database\AAIndex_data.xlsx'

case_dataset = protein_to_graph(case_file,distance_file,feature_file)
case_loader = DataLoader(case_dataset,batch_size = 8,shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(r'AMYGNN_model.pt').to(device)

def test(loader):
    # model_list = os.listdir(path)
    # for i in range(len(model_list)):
    #     model_name = model_list[i]
    #     model = torch.load(path + '\\' + model_name).to(device)
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x.to(device),data.edge_index.to(device),data.edge_attr.to(device),data.batch.to(device))
        probs = out.argmax(dim = 1)

        correct += int((probs.to('cpu') == data.y).sum())
        acc = accuracy_score(probs.to('cpu'),data.y)
        f1 = f1_score(data.y,probs.to('cpu'))
        mcc = matthews_corrcoef(data.y,probs.to('cpu'))
        cm = confusion_matrix(data.y,probs.to('cpu'))
        print("Case study results:",
          "accuracy= {:.4f}".format(acc),
          "f1-score= {:.4f}".format(f1),
          "mcc= {:.4f}".format(mcc),
         "confusion matrix:",cm
          )
    # return accuracy_score(data.y,probs),precision_score(data.y,probs),recall_score(data.y,probs),confusion_matrix(data.y,probs)
test(case_loader)
