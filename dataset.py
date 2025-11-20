import pandas as pd

fname = "crime.csv"
out = "crime_filtered.csv"
chunksize = 100000
use_cols = ["Latitude","Longitude","Primary Type","Date","Year"]  # adjust

writer = None
for chunk in pd.read_csv(fname, usecols=use_cols, parse_dates=["Date"], chunksize=chunksize):
    # example filter: only 2018 onwards (if you have Year or Date)
    if "Year" in chunk.columns:
        filt = chunk[chunk["Year"] >= 2018]
    else:
        filt = chunk[chunk["Date"].dt.year >= 2018]

    if writer is None:
        filt.to_csv(out, index=False, mode="w")
        writer = True
    else:
        filt.to_csv(out, index=False, mode="a", header=False)
print("Saved filtered CSV to", out)
