import pandas as pd
from sklearn.model_selection import train_test_split

def split(df: pd.DataFrame, test_size:float=0.2, val_size: float=0.5) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    og_len = len(df)
    print(f' Len of original dataset: {og_len}')
    df_train, df_test = train_test_split(df, test_size = test_size, random_state = 67)
    df_val, df_test = train_test_split(df_test, test_size = val_size, random_state = 67)
    
    assert len(df_train) == 0.8 * og_len
    assert len(df_val) == 0.1 * og_len
    assert len(df_test) == 0.1 * og_len

    print('split works')

    return df_train, df_val, df_test

def save(df: pd.DataFrame, split: str):
    df.to_csv(f'diamonds_{split}.csv')


if __name__ == "__main__":
    df = pd.read_csv('diamonds.csv')
    train_df, val_df, test_df = split(df)
    save(train_df, 'train')
    save(val_df, 'val')
    save(test_df, 'test')

