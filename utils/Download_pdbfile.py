import urllib.request
from urllib import error
import time
import pandas as pd


def get_uniprotid(path):
    df = pd.read_excel(path,header = [0])
    return df['Uniprot ID']

def downloadpdb(uniprot_id):
    # "https://alphafold.ebi.ac.uk/files/AF-Q13148-F1-model_v4.pdb“
    url = "https://alphafold.ebi.ac.uk/files/" + 'AF-' + uniprot_id.strip(' ') + '-F1-model_v4.pdb'
    req = urllib.request.Request(url,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.41'})
    f = urllib.request.urlopen(req)
    fname = r"./data/Amyloid_Database\PDB" + uniprot_id + '.pdb'
    with open(fname,"wb") as g:
        g.write(f.read())

if __name__=="__main__":
    path = r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides_dropnan.xlsx"
    uniprotid = list(set(get_uniprotid(path)))
    try:
        not_exist_list = ['Q9NYQ7','Q8IZT6','synthetic','Q13315','Q8WZ42','P03036','Q8NEZ4','E9P9G2','Q8TCU4',
                          'P98160','P24043','P98161','P25391','P10636-2','Q9UKN7','P10636-8','P19137','P04275',
                          'P01607']
        for i in range(0,len(uniprotid)):
            uniprot_id = uniprotid[i]
            if uniprot_id not in not_exist_list:
                downloadpdb(uniprot_id)
                time.sleep(8)
            else:
                pass
    except error.HTTPError as e:
        print(uniprot_id)

# pl = PDBList().retrieve_pdb_file('https://alphafold.ebi.ac.uk/files/AF-P05067-F1-model_v4.pdb',pdir = r"E:\paper_datasets\Amyloid_Database\PDB")
# print(pl)

# data = pd.read_excel(r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides_dropnan.xlsx",header = [0])
# print(len(set(data['Uniprot ID'])))
# print(set(data['Uniprot ID']))
