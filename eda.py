from consts import DATADIR, NAME, SUBSET
from datasets import load_from_disk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import itertools
from tqdm import tqdm
def eda(df: pd.DataFrame, func: str)-> None:
    graphical = ["scatterplot", "heatmap"]
    textual = ["corr", "head"]
    if not func:
        print('please define a function type')
        return None
    try:
        if func in textual:
            res = getattr(df, func)()
            print (res)
        elif func in graphical:
            if func == "scatterplot":
                ncols = df.select_dtypes(include=['number']).columns
                fig, axs = plt.subplots(len(ncols), len(ncols), figsize = (12,12))
                for (i, col1), (j, col2) in tqdm(itertools.product(enumerate(ncols), repeat = 2), total  = len(ncols) ** 2, desc = "generating pairplot the long way"):
                    if i == j:
                        sns.histplot(df[col1], ax=axs[i, j], kde=True)
                    else:
                        sns.scatterplot(data = df, x = col1, y = col2, ax = axs[i, j])
                plt.savefig(f'{func}.png')
            elif func == "heatmap":
                plt.figure(figsize = (12, 8))
                sns.heatmap(df.corr(), cmap = "mako", fmt=".2f")
                plt.savefig(f'{func}.png')
        else:
            print('function type either invalid or unsupported')
            return None
    except Exception as e:
        print(e)


if __name__ == "__main__":
    
    train_ds = load_from_disk(f'{DATADIR}/{NAME}-{SUBSET}-train')
    train_df = train_ds.to_pandas()

    parser = argparse.ArgumentParser(description="Perform EDA on a dataset.")
    
    parser.add_argument("--func", type=str, required=True, 
                        help="The EDA function to perform (head, corr, scatterplot, heatmap).")
    parser.add_argument("--fname", type=str, required=False, 
                        help="Filename to save graphical output (required for scatterplot and heatmap).")
    args = parser.parse_args()
    
    train_ds = load_from_disk(f'{DATADIR}/{NAME}-{SUBSET}-train')
    train_df = train_ds.to_pandas()


    eda(train_df, args.func)
