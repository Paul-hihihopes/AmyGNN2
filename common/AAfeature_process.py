import pandas as pd

import pandas as pd

# 1. 读入文件
# 假设你的文件是 CSV 格式，文件名为 'data.csv'
df = pd.read_excel('../data/AAindex/aaindex1_data.xlsx')

# 2. 转置 DataFrame
df_transposed = df.T

# 3. 输出转置后的数据框
# 将转置后的结果输出到新的 CSV 文件
df_transposed.to_csv('../data/tests/transponse_aaindex.csv', header=False)