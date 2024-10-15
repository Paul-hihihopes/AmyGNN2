import numpy as np
import pandas as pd
import re

AA_name_list = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS']
AA_pattern = re.compile(r"I{1}\s*[\w*\/\w*\s*]*")
Number_pattern = re.compile(r'\s*[1-9]\d*\.\d*|0\.\d*[1-9]\d*\s*[1-9]\d*\.\d*|0\.\d*[1-9]\d*')
ID_pattern = re.compile('H{1}\s{1}\w+\d+')
Describe_pattern = re.compile(r"D+\s{1}[\w*-?\s*;\*\d*%*]*")

A_list, R_list, N_list, D_list, C_list, Q_list, E_list, G_list, H_list, I_list, L_list, K_list, M_list, F_list, P_list, S_list, T_list, W_list, Y_list, V_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
ID_list = []
Describe_list = []

def aaindex_to_csv(pre_path,after_path):
    """将原始的AAIndex文件进行处理，提取出H、T列和各个氨基酸对应的数据"""
    with open(pre_path, 'r') as file:
        records = file.read().split('\n')
        for i in range(0, len(records)):
            AA_pattern_flag = re.match(AA_pattern, records[i])
            Number_pattern_flag = re.match(Number_pattern, records[i])
            ID_pattern_flag = re.match(ID_pattern, records[i])
            Describe_parttern_flag = re.match(Describe_pattern, records[i])

            if ID_pattern_flag:
                ID_list.append(ID_pattern_flag.group().split('H ')[1])

            if Describe_parttern_flag:
                Describe_list.append(Describe_parttern_flag.group().split('D ')[1])

            if AA_pattern_flag:
                aa_index1_list = records[i + 1].split()
                aa_index2_list = records[i + 2].split()
                A_list.append(aa_index1_list[0])
                R_list.append(aa_index1_list[1])
                N_list.append(aa_index1_list[2])
                D_list.append(aa_index1_list[3])
                C_list.append(aa_index1_list[4])
                Q_list.append(aa_index1_list[5])
                E_list.append(aa_index1_list[6])
                G_list.append(aa_index1_list[7])
                H_list.append(aa_index1_list[8])
                I_list.append(aa_index1_list[9])
                L_list.append(aa_index2_list[0])
                K_list.append(aa_index2_list[1])
                M_list.append(aa_index2_list[2])
                F_list.append(aa_index2_list[3])
                P_list.append(aa_index2_list[4])
                S_list.append(aa_index2_list[5])
                T_list.append(aa_index2_list[6])
                W_list.append(aa_index2_list[7])
                Y_list.append(aa_index2_list[8])
                V_list.append(aa_index2_list[9])
        file.close()
    data_output = {'ID': [], 'Describe': [],
                   'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
                   'L': [], 'K': [], 'M': [], 'F': [], 'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': []
                   }
    data_output['ID'] = ID_list
    data_output["Describe"] = Describe_list
    data_output["A"] = A_list
    data_output['R'] = R_list
    data_output['N'] = N_list
    data_output['D'] = D_list
    data_output['C'] = C_list
    data_output['Q'] = Q_list
    data_output['E'] = E_list
    data_output['G'] = G_list
    data_output['H'] = H_list
    data_output['I'] = I_list
    data_output['L'] = L_list
    data_output['K'] = K_list
    data_output['M'] = M_list
    data_output['F'] = F_list
    data_output['P'] = P_list
    data_output['S'] = S_list
    data_output['T'] = T_list
    data_output['W'] = W_list
    data_output['Y'] = Y_list
    data_output['V'] = V_list

    #data = pd.DataFrame(data_output).to_excel(after_path, index=True)
    feature_data = pd.read_excel(after_path,header = [0])
    return feature_data

def extract_node_feature(dataframe):
    """从处理之后的AAIndex数据中提取到我们要使用的特征"""
    # feature_list = []
    feature_data = dataframe
    feature_data.set_index("ID", inplace=True)
    feature_data.drop('Unnamed: 0', axis=1, inplace=True)

    # for feature_name in feature_name_list:
    #     feature_list.append(feature_data.loc[feature_name])

    with open(r"./data/tests/transponse_aaindex.txt", encoding='utf-16') as file:
        data = file.readlines()[1:]
        feature_list = []
        for feature_name in data:
            feature_list.append(feature_data.loc[feature_name.strip('\n').split('\t')[0]])
        file.close()
    # print('Extract:',feature_list)
    # for feature_ser in feature_list:
    #     for amino in amino_list:
    #         feature.append(feature_ser[amino])
    return feature_list

def get_coord(file,peptide_path):
    # peptide_data = get_peptide_data(peptide_path)
    coord_list,cl,cd = [],[],[]

    with open(file,'r') as f:
        records = f.readlines()
        Coord = records[4::5]
        f.close()

    for i in range(len(Coord)):
        coord = Coord[i].split('CA_Coord:')[1].strip('[\n')[:-1].split('),')
        #     print(len(coord))
        for j in range(len(coord)):
            c = coord[j].strip(' ').split(', dtype=')[0].strip('array(').strip('[]').strip(' ').split(',')
            for f in c:
                cl.append(f.strip(' '))
        coord_list.append([l for l in cl])
        cl.clear()
        # coord_list.append(cd)
        #         # print(peptide_data['Entry'][i],len(coord_list[i]))
    return coord_list

