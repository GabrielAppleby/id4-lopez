from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import umap
from joblib import Parallel, delayed
from rdkit import Chem
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

RANDOM_SEED: int = 42
N_JOBS: int = -1
OUTPUT_NAME: str = 'transformed_data.csv'
INPUT_NAME: str = 'excel_for_gabe.csv'
FIGURE_NAME_TEMPLATE: str = '{}_{}.html'

SMILE_KEY: str = 'smiles'
ATOMS_KEY: str = 'atoms'
FP_KEY: str = 'fingerprints'
IMAGE_KEY: str = 'img'
METRIC: str = 'jaccard'
CLUSTER: str = 'cluster'

PARAM_START: int = 2
PARAM_STOP: int = 101
PARAM_STEP: int = 5
PARAMS = list(range(PARAM_START, PARAM_STOP + PARAM_STEP, PARAM_STEP))

X_AXIS_TEMPLATE: str = 'x_{}'
Y_AXIS_TEMPLATE: str = 'y_{}'


def umap_reducer(metric, param):
    return umap.UMAP(n_neighbors=param, metric=metric, random_state=RANDOM_SEED)


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_NAME, index_col=False)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    atoms = []
    fingerprints = []
    for smile in df[SMILE_KEY].values:
        mol = Chem.MolFromSmiles(smile)
        atoms.append([x.GetSymbol() for x in mol.GetAtoms()])
        fingerprints.append(np.array([Chem.RDKFingerprint(mol)], dtype=bool))
    possible_atoms = set([atom for sublist in atoms for atom in sublist])

    df[ATOMS_KEY] = atoms
    for atom in possible_atoms:
        df[atom] = df[ATOMS_KEY].map(lambda x: x.count(atom))

    df[FP_KEY] = fingerprints
    df[IMAGE_KEY] = df[SMILE_KEY].map(
        lambda x: 'https://cactus.nci.nih.gov/chemical/structure/{}/image'.format(
            x.replace('#', "%23")))
    return df


def get_pairwise_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    stacked_fps = np.concatenate(df[FP_KEY])
    return pairwise_distances(stacked_fps, metric=METRIC)


def project_single(
        param: int,
        distances: np.ndarray) -> Tuple[np.ndarray, int]:
    return umap.UMAP(n_neighbors=param, metric='precomputed').fit_transform(distances), param


def project_data(distances: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    projections: Iterable[Tuple[np.ndarray, int]] = Parallel(n_jobs=-1)(
        delayed(project_single)(param, distances) for param in tqdm(PARAMS))
    for projection, param in projections:
        df[X_AXIS_TEMPLATE.format(param)] = projection[:, 0]
        df[Y_AXIS_TEMPLATE.format(param)] = projection[:, 1]
    return df


def cluster_data(distances: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    clusters = DBSCAN(metric='precomputed').fit_predict(distances)
    df[CLUSTER] = clusters
    return df


def main():
    df = load_raw_data()
    df = process_data(df)
    distances = get_pairwise_distance_matrix(df)
    df = project_data(distances, df)
    df = cluster_data(distances, df)
    df.to_csv(OUTPUT_NAME)


if __name__ == '__main__':
    main()
