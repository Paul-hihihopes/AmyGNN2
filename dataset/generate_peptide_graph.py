import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected

from common.Preprocess_AAIndexdata import aaindex_to_csv,extract_node_feature,get_coord
from common.extract_edge_feature import generate_edge
from common.Preprocess_peptide_data import get_peptide_data

target_dict = {'Amyloid' : 1,'Non-amyloid' : 0}

#AAindex:获取氨基酸的特征信息
#CPAD2:

# def get_class(peptide_data):
#     class_info = []
#     for i in range(0,len(peptide_data['Entry'])):
#         classification = peptide_data['Classification'][i]
#         class_info.append(target_dict[classification.capitalize()])
#         y_label = torch.tensor(class_info,dtype = torch.float)
#     return y_label

#获取数据集

def get_class(peptide_data):
    class_info = []
    for i in range(0,len(peptide_data['Entry'])):
        classification = peptide_data['Label'][i]
        class_info.append(classification)
        y_label = torch.tensor(class_info,dtype = torch.float)
    return y_label

def peptide_to_graph(path1,path2,peptide_path,dist_path):
    """构建图数据，获取数据集"""
    #读取肽链的基本信息，包括长度，位置等
    peptide_data = get_peptide_data(peptide_path)

    #coord_list：肽链中原子的三维坐标
    coord_list = get_coord(dist_path,peptide_path)
    #获得每个肽链的连边数据与边的特征
    edge_index_list,edge_attr_list = generate_edge(dist_path,peptide_path)

    #fd:结点特征矩阵 37
    amino_list,fd,feature = [],[],[]
    peptide_graph_dataset = []
    #feature_df：氨基酸残基的特征矩阵
    #feature_list:提取的氨基酸特征
    feature_df = aaindex_to_csv(path1,path2)
    feature_list = extract_node_feature(feature_df)
    y_label = get_class(peptide_data)

    for i in range(0,len(peptide_data['Entry'])):
        sequence = peptide_data['Peptide']
        # index = peptide_data['Entry']
        for s in sequence[i].strip(' '):
            amino_list.append(s)
            for f in feature_list:
                feature.append(f[s.strip(' ')])
        length = len(sequence[i])
        coord = coord_list[i]
        for j in range(0,length):
            #对于每个残基，将特征与三维坐标合并
            fd.append(list(feature[len(feature_list) * j : len(feature_list) * (j + 1)]))
            fd[j].append(coord[0 + j * 3])
            fd[j].append(coord[1 + j * 3])
            fd[j].append(coord[2 + j * 3])

            # fd[j].append([on for on in onehot_encodings[i][j]])

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
        # print(peptide_data['Entry'][i],classification,class_info,y_label,peptide_graph)
        # print(amino_list,len(amino_list),peptide_graph,len(edge_index))
        amino_list.clear()
        feature.clear()
        fd.clear()

    return peptide_graph_dataset

if __name__ == "__main__":
    pre_peptide_file = r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides.xlsx"
    after_peptide_file = r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides_dropnan.xlsx"
    pre_file = r"E:\paper_datasets\AAindex\aaindex1.txt"
    after_feature_file = r'E:\paper_datasets\Amyloid_Database\AAIndex_data.xlsx'
    peptide_pdb_file =  r"E:\paper_datasets\Amyloid_Database\PDB_Data"
    # AADist_file = r"E:\paper_datasets\Amyloid_Database\Feature\aggregating_peptide_AADIST_feature.txt"
    AADist_file = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt"
    # AADist_file = r'E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature_6.txt'
    # AADist_file = r'E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature_7.txt'
    # AADist_file = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature_8.txt"
    # AADist_file = r'E:\paper_datasets\CaseStudy\distance.txt'

    file = r'E:\paper_datasets\after_SMOTE.csv'

    # aaindex_data = aaindex_to_csv(pre_file,after_feature_file)
    # feature_df = extract_feature(aaindex_data)
    # print(generate_edge(AADist_file,after_peptide_file))
    # print(get_coord(AADist_file,after_peptide_file))
    # generate_edge(AADist_file,after_peptide_file)
    x_Data = peptide_to_graph(pre_file,after_feature_file,file,AADist_file)
    # print(x_Data)
    # print(x_Data[0:10])
    # print(x_Data[0].edge_index)
    # print(x_Data[0].edge_attr)
    # for i in range(0,6):
    #     print(x_Data[i].edge_index)
    #     pep_graph = to_networkx(x_Data[i])
    #     print(x_Data[i])
    #     nx.draw(pep_graph,cmap = plt.get_cmap('Set3'))
    #     plt.show()
