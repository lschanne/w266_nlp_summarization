import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


ds, info = tfds.load('cnn_dailymail', split='train', with_info=True)

#tfds into dataframe
df = tfds.as_dataframe(ds, info)

#convert from bytes to string lists
document = list(df['article'].str.decode("utf-8"))
summary = list(df['highlights'].str.decode("utf-8"))
str_df = df.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
for col in str_df:
    df[col] = str_df[col]


#drop NA values    
df = df.dropna()
df.head()

#remove special characters and lowercase
df["article"] = df["article"].str.replace('[^A-Za-z\s]+', '')
df["highlights"] =df["highlights"].str.replace('[^A-Za-z\s]+', '')

df["article"] = df["article"].str.lower()
df["highlights"] = df["highlights"].str.lower()
df.head()

#save as csv
df.to_csv('cnn_dailymail.csv')