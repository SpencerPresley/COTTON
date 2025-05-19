import pandas as pd

def get_num_rows(filename):
    df = pd.read_csv(filename)
    return len(df)

if __name__ == "__main__":
    print(f"train.csv: {get_num_rows('train.csv')}")
    print(f"valid.csv: {get_num_rows('valid.csv')}")
    print(f"humaneval.csv: {get_num_rows('humaneval.csv')}")