import pandas as pd
def get_data(split: 'str'):
    df = pd.read_csv(f'../diamonds_{split}.csv')
    df = df.rename(columns = {'Unnamed: 0': 'Index'})
    x = df.drop(['Index', 'price'], axis = 1)
    y = df['price']
    x_arr = x.to_numpy()
    y_arr = y.to_numpy()
    print(x_arr[1])
    print(y_arr[0])
    return x_arr, y_arr

