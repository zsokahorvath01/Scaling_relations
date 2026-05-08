import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from gdr3bcg.bcg import BolometryTable
from tqdm import tqdm
from joblib import Parallel, delayed

table = BolometryTable()

def bcg(row):
    """
    Deriving bolometric correction in G band using Monte-Carlo iteration.

    data: dataframe with columns [ 'T_eff', 'e_T_eff', 'log_g', 'e_log_g', '[Fe/H]', 'e_[Fe/H]', '[alpha/Fe]', 'e_[alpha/Fe]' ]
    """
    teff, eteff = row['T_eff'], row['e_T_eff']
    logg, elogg = row['log_g'], row['e_log_g']
    feh, efeh = row['[Fe/H]'], row['e_[Fe/H]']
    afe, eafe = row['[alpha/Fe]'], row['e_[alpha/Fe]']

    # Chossing from normal distribution for all the 4 parameters:
    norm_teff = np.random.normal(teff, eteff, 500)
    norm_logg = np.random.normal(logg, elogg, 500)
    norm_feh  = np.random.normal(feh, efeh, 500)
    norm_afe  = np.random.normal(afe, eafe, 500)

    # Calculating BC with the new parameters:
    bcg = np.array([
        table.computeBc([t, g, f, a])
        for t, g, f, a in zip(norm_teff, norm_logg, norm_feh, norm_afe)
    ])
    
    return bcg


def main():
    data = pd.read_csv('for_get_mass_v2.csv')
    dataiter = data.copy()

    # tqdm wrapper parallelhoz
    tqdm.pandas()

    rows = list(dataiter.iterrows())

    results = Parallel(n_jobs=4)(
        delayed(bcg)(row[1]) for row in tqdm(rows)
    )

    dataiter['bcg_results'] = results

    # mentés
    dataiter.to_pickle('bcg_results.pkl')


if __name__ == "__main__":
    main()
