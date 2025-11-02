# filter data to get only 2024 records

import pandas as pd
import sys

infile = sys.argv[1] if len(sys.argv) > 1 else "alex_steps_per_day.csv"
outfile = sys.argv[2] if len(sys.argv) > 2 else "alex_steps_per_day_2024.csv"

df = pd.read_csv(infile)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df_2024 = df[df["date"].dt.year == 2024]
df_2024.to_csv(outfile, index=False)

print(f" Saved {len(df_2024)} 2024-only rows to {outfile}")
