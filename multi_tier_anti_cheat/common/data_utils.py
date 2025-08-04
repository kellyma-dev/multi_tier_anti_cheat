import pandas as pd
import arff

def read_arff(arff_file: str) -> pd.DataFrame:
    with open(arff_file, 'r') as f:
        dataset = arff.load(f)

    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

    return df