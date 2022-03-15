prefix = '../data/'

def collect_data(inp='PP/index/INDEX_general_PP.2020'):
    import os
    import pandas as pd
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    from Bio.PDB import PDBParser

    from catboost import CatBoostRegressor

    from tqdm import tqdm
    import pickle
    
    data = {
        'pdb': [],
        'resolution': [],
        'year': [],
        'target': [],
        'value': [],
        'unit': [],
        'precision': []
    }

    with open(prefix + inp) as f:
        for line in tqdm(f.readlines()):
            if not line.startswith('#'):
                parts = line.strip().split()
                pdb = parts[0]
                resolution = 0 if parts[1] == 'NMR' else float(parts[1])
                year = int(parts[2])

                if '=' in parts[3]:
                    sep = '='
                elif '~' in parts[3]:
                    sep = '~'
                elif '<' in parts[3]:
                    sep = '<'
                elif '>' in parts[3]:
                    sep = '>'
                elif '<=' in parts[3]:
                    sep = '<='
                elif '=<' in parts[3]:
                    sep = '=<'
                elif '>=' in parts[3]:
                    sep = '>='
                elif '=>' in parts[3]:
                    sep = '=>'

                target, value_unit =  parts[3].split(sep)

                values = re.findall(r'\d+', value_unit)
                if len(values) == 1:
                    value = float(values[0])
                elif len(values) == 2:
                    value = float('.'.join(values))

                unit = re.sub(r'[^a-zA-Z]', '', value_unit)

                data['pdb'].append(pdb)
                data['resolution'].append(resolution)
                data['year'].append(year)
                data['target'].append(target)
                data['value'].append(value)
                data['unit'].append(unit)
                data['precision'].append(sep)

    data = pd.DataFrame(data)
    return data


def parse_chains(inp='PP', out='pdb2chains.pkl'):
    import os
    import pickle
    import warnings
    warnings.filterwarnings('ignore')

    from Bio.PDB import PDBParser
    from tqdm import tqdm

    pdb2chains = {}

    pdbs_path = prefix + inp

    parser = PDBParser()

    for fname in tqdm(os.listdir(pdbs_path)):
        if fname.endswith('ent.pdb'):
            pdb = fname.split('.')[0]
            path = os.path.join(pdbs_path, fname)
            struct = parser.get_structure('s', path)
            pdb2chains[pdb] = ','.join([c.id for c in struct.get_chains()])

    pickle.dump(pdb2chains, open(prefix + out, 'wb'))
    

def get_full(out='pdbbind_pp_summary.csv'):
    import pickle
    import numpy as np
    data = collect_data()
    parse_chains()
    
    pdb2chains = pickle.load(open(prefix + 'pdb2chains.pkl', 'rb'))
    data['chains'] = data['pdb'].map(pdb2chains)
    data['number_of_chains'] = data['chains'].str.split(',').str.len()
    unit2coeff = {
        'mM': 10e-3,
        'uM': 10e-6,
        'nM': 10e-9,
        'pM': 10e-12,
        'fM': 10e-15,
    }
    data['coefficient'] = data['unit'].map(unit2coeff)
    data['logM'] = np.log10(data['value'] * data['coefficient'])
    data.to_csv(prefix + out, index=False)
    return data

    
def prepare_input(data, out='synthetic.csv'):
    import os
    import pandas as pd
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    from Bio.PDB import PDBParser

    from catboost import CatBoostRegressor

    from tqdm import tqdm
    import pickle
    
    parser = PDBParser()
    kd_data = data[(data['target'] == 'Kd') & (data['number_of_chains'] == 2)]
    df = []
    for i in tqdm(range(len(kd_data))):
        row = kd_data.iloc[i].copy()
        name = row['pdb']
        path = prefix+f'PP/{name}.ent.pdb'
        structure = parser.get_structure(name, path)
        cur = []
        for chain in structure[0]:
            cnt = 0
            for residue in chain:
                for atom in residue:
                    cnt += 3
            cur.append(cnt)
        df.append(cur)
    df = pd.DataFrame(df)
    df = df.rename({0: 'first_protein_size', 1: 'second_protein_size'}, axis=1)
    df['target'] = list(kd_data['logM'])
    df.index = np.arange(1, len(df)+1)
    df.to_csv(prefix + out)
    return df

def train_test_div(df, train_size=0.8):
    train_sz = round(len(df) * train_size)
    X_train = df.iloc[:train_sz][['first_protein_size', 'second_protein_size']]
    y_train = df.iloc[:train_sz]['target']
    X_test = df.iloc[train_sz:][['first_protein_size', 'second_protein_size']]
    y_test = df.iloc[train_sz:]['target']
    return X_train, X_test, y_train, y_test