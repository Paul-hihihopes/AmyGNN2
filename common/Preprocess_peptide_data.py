import pandas as pd

def preprocess_peptide_data(read_path,save_path):
    """得到多肽数据"""
    no_pdb_list = ['Q9NYQ7', 'Q8IZT6', 'synthetic', 'Q13315', 'Q8WZ42', 'P03036', 'Q8NEZ4', 'E9P9G2', 'Q8TCU4',
                   'P98160', 'P24043', 'P98161', 'P25391', 'P10636-2', 'Q9UKN7', 'P10636-8', 'P19137', 'P04275',
                   'P01607']
    df = pd.read_excel(read_path, header=[0])
    for i in range(0, len(df['Entry'])):
        if (type(df['Uniprot ID'][i]) == float or (len(df['Peptide'][i].strip(' ')) != df['Length'][i])):
            df.drop([i], inplace=True)
        elif ((df['Position'][i] == 'scrambled') or (type(df['Position'][i]) == float)):
            df.drop([i],inplace = True)
        elif df['Uniprot ID'][i] in no_pdb_list:
            df.drop([i],inplace = True)
    df.to_excel(save_path, index=False, header=True)
    return save_path

def get_peptide_data(peptide_file):
    # peptide_data = pd.read_excel(peptide_file,header = [0])
    # peptide_data = peptide_data.fillna(0)
    peptide_data = pd.read_csv(peptide_file,header = [0],delimiter = ',',encoding = 'utf-8')
    return peptide_data
