import pandas as pd
import numpy as np
import os
import warnings

dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/"
filepath = "Processed_datasets/CSE-CIC-IDS2018-AWS/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
attr_types = {
              
            }
# get files
files = os.listdir(dirpath)

# get columns
headers_list = dict()
for f in files:
    headers = pd.read_csv(dirpath + f, sep=',', index_col=False, nrows=0,).columns.tolist()
    print(f"Filename: {f}")
    print(f"Number of fields: {len(headers)}")
    print(f"Fields: {headers}")
    headers_list.update({f : headers})

# get types
types_list = dict()
warnings_list =dict()
for f in files:
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Call code that triggers a warning.
        df = pd.read_csv(dirpath + f, sep=',', )

        # ignore any non-custom warnings that may be in the list
        w = filter(lambda i: issubclass(i.category, pd.errors.DtypeWarning), w)
        # get warning list and extract message
        lw = list(w)
        if len(lw)>0:
            warnings_list.update({f : lw[0].message})
    
        columns_types = dict(df.dtypes)
        print(f"Filename: {f}")
        print(f"Columns types: {columns_types}")
        types_list.update({f : columns_types})

# get number of records
record_numbers = dict()
for f in files:
    df = pd.read_csv(dirpath + f, sep=',', )
    numrows = df.shape[0]
    print(f"Filename: {f}")
    print(f"Number of rows: {numrows}")
    record_numbers.update({f : numrows})
    
# count the number of nunique and non-null values for each column
# dictionary {"column_name": (nunique, non-null)}
values_stats = dict()
for f in files:
    print(f"Filename: {f}")
    cols_stats = list()
    for column in headers_list.get(f):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
            df = pd.read_csv(dirpath + f, sep=',', usecols=[column])
            cols_stats.append({column: (df[column].nunique(), df[column].count())})
            
    print(cols_stats)    
    values_stats.update({f : cols_stats})
    
# aggregate stats
stats = dict()
nrows = list()
ncols = list()
fields = list()
fields_types = list()
cols_statistics = list()
type_warnings = list()
for f in files:
    nrows.append(record_numbers.get(f))
    ncols.append(len(headers_list.get(f)))
    fields.append(headers_list.get(f))
    fields_types.append(types_list.get(f))
    cols_statistics.append(values_stats.get(f))
    type_warnings.append(warnings_list.get(f))
    
stats = {"Filename" : files,
          "NRows" : nrows,
          "NColumns" : ncols,
          "Fields" : fields,
          "Fields-Types" : fields_types,
          "Stats (Unique, Non-null)" : cols_statistics,
          "Type_Warning" : type_warnings
        }

stats_df = pd.DataFrame.from_dict(stats, orient='columns')
stats_df.to_csv("ids2018_stats.txt", sep=',')
