import numpy as np
import pandas as pd
import torch
from common.Preprocess_peptide_data import get_peptide_data


#根据文件生成图的边
def generate_edge(file,peptide_path):
    peptide_data = get_peptide_data(peptide_path)
    Dpc = pd.read_csv(r"./data/Amyloid_Database/Feature/aggregating_peptides_DPC_feature.csv", header=[0], delimiter =',')
    Dpc.rename(columns={'Unnamed: 0': 'Entry'}, inplace=True)
    #初始化向量
    distance,elu_dist,elu_distance,dpc_list,edge_source,edge_target,edge_attr = [],[],[],[],[],[],[]
    edge_index = [[],[]]
    edge_index_save,edge_attr_save = [],[]

    with open(file,'r') as f:
        records = f.readlines()
        #读取边文件里特定行数据，获得边的数据集合
        Edge_Source = records[0::5]
        Edge_Target = records[1::5]
        Elu_dist = records[2::5]
        f.close()
    for i in range(len(peptide_data['Entry'])):
        sequence = peptide_data['Peptide']
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
        #elu_dist: 图的邻接矩阵
        if elu_dist != ['']:
            for dist in elu_dist:
                if float(dist) != 0.0:
                    distance.append(float(dist.strip('')))
        # for dis in distance:
        #     if dis < 5.0 and dis != 0.0:
        #         elu_distance.append(dis)
        # edge_attr.append(elu_distance)

        #aa1 aa2:形成肽键的氨基酸
        for j in range(len(edge_index[0])):
            aa1 = sequence[i][(np.round(edge_index[0][j]))]
            aa2 = sequence[i][(np.round(edge_index[1][j]))]
            index_name = 'DPC_' + aa1 + aa2
            dpc_list.append(Dpc[index_name][i])
        edge_attr.append(dpc_list)
        edge_attr_np = np.array(edge_attr).transpose((1,0))

        # print(peptide_data['Entry'][i],edge_index[0],edge_index[1],np.array(edge_index).shape,edge_attr_np.shape)

        edge_attr_tensor = torch.tensor(edge_attr_np, dtype=torch.float32)
        # print(edge_attr_np)
        # print(edge_attr_tensor.shape)
        edge_attr[0].clear()
        # edge_attr[1].clear()
        edge_attr.clear()
        edge_index[0].clear()
        edge_index[1].clear()
        distance.clear()
        elu_distance.clear()
        # edge_index.clear()
        # print(peptide_data['Entry'][i],edge_index_tensor,edge_index_tensor.shape, edge_attr_tensor.shape)
        edge_index_save.append(edge_index_tensor)
        edge_attr_save.append(edge_attr_tensor)
        # print(peptide_data['Entry'][i],edge_index_save[i])
    return edge_index_save,edge_attr_save

# generate_edge(r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt",r'E:\paper_datasets\after_SMOTE.csv')