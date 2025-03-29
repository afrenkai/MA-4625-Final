from datasets import load_dataset
from consts import DATADIR, NAME, SUBSET


def get_dataset(name: str, subset: str, stream: bool = False):
    try:
        if stream:
            ds = load_dataset(name, subset, split='train', streaming = True)
            return ds
        ds = load_dataset(name, subset, split='train')
        return ds
    except Exception as e:
        print(e)


if __name__ == "__main__":
    ds = get_dataset(NAME, SUBSET)
    print(ds[1])
    ds.save_to_disk(f'{DATADIR}/{NAME}-{SUBSET}')