from consts import DATADIR, NAME, SUBSET
from datasets import load_dataset, Dataset, load_from_disk


def split(ds: Dataset, test_size:float=0.2, val_size: float=0.5) -> (Dataset, Dataset, Dataset):

    og_len = len(ds['Run1'])
    print(f' Len of original dataset: {og_len}')
    ds_train, ds_test = ds.train_test_split(test_size=test_size).values()
    ds_test, ds_val = ds_test.train_test_split(test_size=val_size).values()

    assert len(ds_train) == 0.8 * og_len
    assert len(ds_val) == 0.1 * og_len
    assert len(ds_test) == 0.1 * og_len

    print('split works')

    return ds_train, ds_val, ds_test

def save(ds: Dataset, dir, name, subset, split):
    ds.save_to_disk(f'{dir}/{name}-{subset}-{split}')


if __name__ == "__main__":
    ds = load_from_disk(f'{DATADIR}/{NAME}-{SUBSET}')

    train_ds, val_ds, test_ds = split(ds)
    save(train_ds, DATADIR, NAME, SUBSET, 'train')
    save(val_ds, DATADIR, NAME, SUBSET, 'val')
    save(test_ds, DATADIR, NAME, SUBSET, 'test')
