import pandas as pd
import os


class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.read_csv()

    def read_csv(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError(f"File {self.file_path} not found")

    def remove_nan(self):
        self.df = self.df.dropna()
        return self.df

    def get_df(self):
        return self.df


class CO2Reader(CSVReader):
    def __init__(self, file_path, use_alternative=False):
        self.file_path = file_path
        self.use_alternative = use_alternative
        if self.use_alternative:
            self.df = self.read_csv_alternative()
        else:
            self.df = self.read_csv_standard()

    def read_csv_standard(self):
        """
        Reads the standard CO2 CSV with detailed columns.
        """
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            df = df[df["Entity"] == "World"]
            df["annual_co2_emissions_gt"] = (
                df["Annual CO₂ emissions"] / 1e9
            )  # Convert kg to Gt
            df["year"] = df["Year"]
            df = df[["year", "annual_co2_emissions_gt"]]
            return df
        else:
            raise FileNotFoundError(f"File {self.file_path} not found")

    def read_csv_alternative(self):
        """
        Reads the alternative CO2 CSV with limited columns.
        """
        if os.path.exists(self.file_path):
            df = pd.read_csv(self.file_path)
            df["annual_co2_emissions_gt"] = (
                df["fossil emissions excluding carbonation"] * 3.664
            )  # Convert Gt C to Gt CO2
            df["year"] = df["Year"]
            df = df[["year", "annual_co2_emissions_gt"]]
            return df
        else:
            raise FileNotFoundError(f"File {self.file_path} not found")

    def get_df(self):
        return self.df
