# This converts a matlab .mat file to a python pandas dataframe
import scipy.io
import pandas as pd


class ConvertMatlab:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.convert()
        self.df = self.remove_nan()

    def convert(self):
        # Data has keys in itself so its a nested dictionary
        mat = scipy.io.loadmat(self.file_path)
        data = mat["Data"]
        # Columns are  ['datenum', 'year', 'month', 'day', 'SIE']
        # They are also set as data types
        dataenum = data[0][0][0]
        year = data[0][0][1]
        month = data[0][0][2]
        day = data[0][0][3]
        SIE = data[0][0][4]

        df = pd.DataFrame(dataenum, columns=["dataenum"])
        df["year"] = year
        df["month"] = month
        df["day"] = day
        df["SIE"] = SIE

        print(df.head())

        return df

    def get_latest_sie_winter(self):
        # Get the latest winter sea ice extent
        df_march = self.df[self.df["month"] == 3]
        latest_year = df_march["year"].max()
        df_latest = df_march[df_march["year"] == latest_year]
        sie_winter = df_latest["SIE"].values[0]
        return sie_winter

    def get_latest_sie_summer(self):
        # Get the latest summer sea ice extent
        df_september = self.df[self.df["month"] == 9]
        latest_year = df_september["year"].max()
        df_latest = df_september[df_september["year"] == latest_year]
        sie_summer = df_latest["SIE"].values[0]
        return sie_summer

    def remove_nan(self):
        # Replae -999 with nan
        self.df = self.df.replace(-999, pd.NA)
        self.df = self.df.dropna()
        return self.df

    def get_df(self):
        return self.df
