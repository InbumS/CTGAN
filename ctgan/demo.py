# Demo
import pandas as pd

DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'

# 인구 조사 데이터를 읽어온다 (type: gzip)
def load_demo():
    return pd.read_csv(DEMO_URL, compression='gzip')
