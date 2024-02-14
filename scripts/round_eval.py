import sys
import pandas as pd

# Used to create table in reports

input_csv = sys.argv[1]
df = pd.read_csv(input_csv)
#print(df)
columns_to_round = ['accuracy', 'precision', 'recall', 'f1_score', 'bal accuracy', 'f2_score']
df[columns_to_round] = df[columns_to_round].round(4)
#print(df)
df.to_csv('output.csv', index=False)
