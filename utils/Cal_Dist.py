import numpy as np
from Bio.PDB import *
# from generate_peptide_graph import get_peptide_data
from common.Preprocess_peptide_data import preprocess_peptide_data,get_peptide_data

# file_path = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt"
# file_path = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature_6.txt"
# file_path = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature_7.txt"
file_path = r"../data/Amyloid_Database/Feature/new_aggregating_peptide_AADIST_feature_11.txt"

def Elu_dist(coord_1,coord_2):
    return np.sqrt(sum((coord_1 - coord_2) ** 2))

def Cosine_dist(coords_1,coords_2):
    cosine_dist = np.dot(coords_1, coords_2) / (np.sqrt(sum((coords_1) ** 2)) * np.sqrt(sum((coords_2) ** 2)))
    return cosine_dist

def peptide_to_coord(pdb_file_path,peptide_file):
    """从多肽得结构数据中提取CA原子得坐标"""
    uniprot_id_list = list(get_peptide_data(peptide_file)['Uniprot ID'])
    pdb_id_list = list(get_peptide_data(peptide_file)['Entry'])
    peptide_position = get_peptide_data(peptide_file)['Position']
    no_pdb_list = ['Q9NYQ7','Q8IZT6','synthetic','Q13315','Q8WZ42','P03036','Q8NEZ4','E9P9G2','Q8TCU4',
                   'P98160','P24043','P98161','P25391','P10636-2','Q9UKN7','P10636-8','P19137','P04275',
                   'P01607']

    #coord_list:三维坐标列表
    coord_list,coord,elu_list,cosine_list,edge_source,edge_target = [],[],[],[],[],[]
    # coord_list_dict = {}

    pdb_parser = PDBParser(QUIET=True)
    for j in range(0,len(uniprot_id_list)):
        uniprot_id = uniprot_id_list[j]
        pdb_id = pdb_id_list[j]
        position = peptide_position[j]
        if uniprot_id not in no_pdb_list:
            #获取pdb结构
            pdb_structure = pdb_parser.get_structure(uniprot_id,pdb_file_path + '\\PDB' +uniprot_id + '.pdb')
            #获取pdb文件中的残基序列
            residues = list(pdb_structure.get_residues())

            if type(position) != float:
                #获取肽链的开始位置和结束位置
                start = int(position.split('-')[0])
                end = int(position.split('-')[1])
                #获取每一个氨基酸残基每一个Cα原子的三维坐标
                for i in range(start - 1 ,end):
                    atoms = residues[i].get_atoms()
                    for atom in atoms:
                        if atom.get_fullname().strip(' ') == 'CA':
                            atom_coord = atom.get_coord()
                            coord_list.append(atom_coord)
            else:
                pass
        with open(file_path,'a+') as f:
            for m in range(0,len(coord_list)):
                for n in range(0,len(coord_list)):
                    #Elu_dist:三维坐标之间的距离
                    elu_list.append(Elu_dist(coord_list[m],coord_list[n]))
                    cosine_list.append(Cosine_dist(coord_list[m],coord_list[n]))
                    #最大的边长为8A
                    if Elu_dist(coord_list[m],coord_list[n]) != 0.0 and Elu_dist(coord_list[m],coord_list[n]) <= 8.0:
                        edge_source.append(m)
                        edge_target.append(n)
            f.write(pdb_id + 'Edge_Source:' + str(edge_source) + '\n')
            f.write(pdb_id + 'Edge_Target:' + str(edge_target) + '\n')

            f.write(pdb_id + 'Elu_dist:' + str(elu_list) + '\n')
            f.write(pdb_id + 'Cosine_Dist:' + str(cosine_list) + '\n')
            f.write(pdb_id + 'CA_Coord:' + str(coord_list) + '\n')

        f.close()

        coord_list_dict = dict([(pdb_id + '_' + 'elu',elu_list),(pdb_id + '_' + 'cosine',cosine_list)])

        # df = pd.DataFrame({'PDB_ID':pdb_id,'Elu_Dist' : elu_list,'Cosine_Dist' : cosine_list})
        # df.to_csv(file_path,index = True,sep = ',')
        print(pdb_id, len(coord_list), [k for k in coord_list_dict.keys()],
              len([v for v in coord_list_dict.values()][0]))

        coord_list.clear()
        elu_list.clear()
        cosine_list.clear()
        edge_source.clear()
        edge_target.clear()


read_file = r"../data\Amyloid_Database\CPAD2.0_Data\aggregating_peptides.xlsx"
save_file = r'../data\after_process.csv'
peptide_pdb_file =  r"../data\Amyloid_Database\PDB_Data"
peptide_to_coord(peptide_pdb_file,save_file)

