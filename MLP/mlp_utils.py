import pandas as pd
def get_data(split: 'str'):
    df = pd.read_csv(f'../diamonds_{split}.csv')
    df = df.rename(columns = {'Unnamed: 0': 'Index'})
    print(df.dtypes)
    x = df.drop(['Index', 'price', 'cut', 'color', 'clarity'], axis = 1)
    y = df['price']
    print(y.describe())
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    print(x_arr[1])
    print(y_arr[0])
    return x_arr, y_arr
if __name__ == "__main__":
    x_arr, y_arr = get_data('train')
