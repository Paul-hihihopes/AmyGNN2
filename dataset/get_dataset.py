import pandas as pd
import numpy as np
import torch
from Bio.PDB import PDBParser,PDBIO
import requests
from common.Preprocess_AAIndexdata import extract_node_feature
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.data import Data
from pathlib import Path

def Elu_dist(coord_1,coord_2):
    return np.sqrt(sum((coord_1 - coord_2) ** 2))

def Cosine_dist(coords_1,coords_2):
    cosine_dist = np.dot(coords_1, coords_2) / (np.sqrt(sum((coords_1) ** 2)) * np.sqrt(sum((coords_2) ** 2)))
    return cosine_dist

def generate_dataset(peptide_path, save_path, buffer_path, shortest_length):
    peptide_data = pd.read_csv(peptide_path,header = [0],delimiter = ',',encoding = 'utf-8')
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    graph_dataset = []

    sequence = peptide_data['Peptide']
    y_labels = peptide_data['Label']

    feature_df = pd.read_excel('../data\\Amyloid_Database\\AAIndex_data.xlsx',header = [0])
    feature_list = extract_node_feature(feature_df)

    Dpc = pd.read_csv(r"../data/Amyloid_Database/Feature/aggregating_peptides_DPC_feature.csv", header=[0],
                      delimiter=',')
    Dpc.rename(columns={'Unnamed: 0': 'Entry'}, inplace=True)

    for i in range(0,len(peptide_data['Peptide'])):
        pdb_parser = PDBParser(QUIET=True)
        pdb_path = Path(f"{buffer_path}/PDB_{sequence[i]}.pdb")

        if not pdb_path.is_file():
            response = requests.post(url, data=sequence[i])

            if response.status_code == 200:
                # 将响应内容保存为 pdb 文件
                with open("../temp/temp.pdb", "w") as file:
                    file.write(response.text)
                print("PDB 文件已保存为 temp")

            structure = pdb_parser.get_structure("init", "../temp/temp.pdb")

            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_path.as_posix())

        # uniprot_id = peptide_data['Uniprot ID'][i]
        # pdb_structure = pdb_parser.get_structure(uniprot_id,f"../data\\Amyloid_Database\\PDB_Data\\PDB_{sequence}.pdb")
        pdb_structure = pdb_parser.get_structure(i, pdb_path)

        residues = list(pdb_structure.get_residues())
        coord_list = []
        elu_list = []
        cosine_list = []
        edge_source, edge_target = [], []
        fd , dpc_list=[], []
        amino_list, feature = [], []

        # 获取肽链的开始位置和结束位置
        # 获取每一个氨基酸残基每一个Cα原子的三维坐标

        # if type(pep_position) != float:
        #     # 获取肽链的开始位置和结束位置
        #     start = int(pep_position.split('-')[0])
        #     end = int(pep_position.split('-')[1])
        #     for k in range(start-1, end):
        #         atoms = residues[k].get_atoms()
        #         for atom in atoms:
        #             if atom.get_fullname().strip(' ') == 'CA':
        #                 atom_coord = atom.get_coord()
        #                 coord_list.append(atom_coord)

        for k in range(0, len(residues)):
            atoms = residues[k].get_atoms()
            for atom in atoms:
                if atom.get_fullname().strip(' ') == 'CA':
                    atom_coord = atom.get_coord()
                    coord_list.append(atom_coord)

        for m in range(0, len(coord_list)):
            for n in range(0, len(coord_list)):
                # Elu_dist:三维坐标之间的距离
                elu_list.append(Elu_dist(coord_list[m], coord_list[n]))
                cosine_list.append(Cosine_dist(coord_list[m], coord_list[n]))
                # 最大的边长为8A
                if Elu_dist(coord_list[m], coord_list[n]) != 0.0 and Elu_dist(coord_list[m], coord_list[n]) <= shortest_length:
                    edge_source.append(m)
                    edge_target.append(n)

        for s in sequence[i].strip(' '):
            amino_list.append(s)
            for f in feature_list:
                feature.append(f[s.strip(' ')])

        for j in range(0,len(sequence[i])):
            fd.append(list(feature[len(feature_list) * j: len(feature_list) * (j + 1)]))
            fd[j].append(coord_list[j][0])
            fd[j].append(coord_list[j][1])
            fd[j].append(coord_list[j][2])

        for j in range(len(edge_source)):
            aa1 = sequence[i][(np.round(edge_source[j]))]
            aa2 = sequence[i][(np.round(edge_target[j]))]
            index_name = 'DPC_' + aa1 + aa2
            dpc_list.append(Dpc[index_name][i])

        eat = torch.tensor(dpc_list)
        x = torch.tensor(np.array(fd, dtype=float), dtype=torch.float)
        edge_index = torch.stack([torch.tensor(edge_source),torch.tensor(edge_target)])
        edge_index_torch = to_undirected(edge_index)
        y = torch.tensor(np.array(y_labels[i]), dtype=torch.long)

        peptide_graph = Data(x=x, edge_index=edge_index_torch, edge_attr=eat, y=y)
        graph_dataset.append(peptide_graph)
        print(f'当前进度:{i}')

    torch.save(graph_dataset, save_path)
    return graph_dataset

if __name__ == '__main__':

    graph_dataset0 = generate_dataset('../data/test_raw.csv','../data/processed_dataset/test_dataset.pkl','../temp/trainset_buffer/',5)
    print(graph_dataset0)
    print(len(graph_dataset0))

    # graph_read = torch.load('../data/processed_dataset/train_dataset.pkl')
    # print(graph_read)
