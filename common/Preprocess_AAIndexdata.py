import numpy as np
import pandas as pd
import re

def extract_node_feature(dataframe):
    """从处理之后的AAIndex数据中提取到我们要使用的特征"""
    # feature_list = []
    feature_data = dataframe
    feature_data.set_index("ID", inplace=True)
    feature_data.drop('Unnamed: 0', axis=1, inplace=True)

    # for feature_name in feature_name_list:
    #     feature_list.append(feature_data.loc[feature_name])

    with open(r"../data/tests/transponse_aaindex.txt", encoding='utf-16') as file:
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

# def get_coord(file,peptide_path):
#     # peptide_data = get_peptide_data(peptide_path)
#     coord_list,cl,cd = [],[],[]
#
#     with open(file,'r') as f:
#         records = f.readlines()
#         Coord = records[4::5]
#         f.close()
#
#     for i in range(len(Coord)):
#         coord = Coord[i].split('CA_Coord:')[1].strip('[\n')[:-1].split('),')
#         #     print(len(coord))
#         for j in range(len(coord)):
#             c = coord[j].strip(' ').split(', dtype=')[0].strip('array(').strip('[]').strip(' ').split(',')
#             for f in c:
#                 cl.append(f.strip(' '))
#         coord_list.append([l for l in cl])
#         cl.clear()
#         # coord_list.append(cd)
#         #         # print(peptide_data['Entry'][i],len(coord_list[i]))
#     return coord_list