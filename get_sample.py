from consts import DATADIR, NAME, SUBSET
from datasets import load_from_disk
if __name__ == "__main__":
    train_ds = load_from_disk(f'{DATADIR}/{NAME}-{SUBSET}-train')
    train_df = train_ds.to_pandas()
    print(train_df.describe())
