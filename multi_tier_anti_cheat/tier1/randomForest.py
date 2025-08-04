import os

from multi_tier_anti_cheat.common.data_utils import read_arff
from multi_tier_anti_cheat.common.csv_utils import combine_files

EX_SPC_PATH = "ExSPC/dataset"
REV_POV_PATH = "data/RevPOV/dataset/arff"


def test_read_arff(filename: str):
    current_dir = os.getcwd()
    print(current_dir)
    arff_file = os.path.join(current_dir, "../..", REV_POV_PATH, filename)
    print(arff_file)

    df = read_arff(arff_file)
    #common.combine_file()

    print(df.head())

# The Random Forest approach for cheat detection

test_read_arff('final_train_1.arff')

combine_files()

