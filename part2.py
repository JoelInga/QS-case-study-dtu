import convert_matlab as convert_matlab_module
import noetz_stroeve as ns_module
import csv_reader as csv_reader_module
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm  # For progress bars in Monte Carlo simulations
import numpy as np


def calculate_linear_trend_co2(df):
    """
    Calculates the linear trend (slope) of CO2 emissions starting from 1978.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the CO2 emissions data.

    Returns:
    tuple: (slope, intercept) of the linear trend in Gt CO2/year² and Gt CO2 respectively.
    """
    # Ensure data is sorted by year
    df = df.sort_values(by="year")

    # Perform linear regression: emissions = slope * year + intercept
    # Using numpy's polyfit for accurate slope and intercept
    slope, intercept = np.polyfit(df["year"], df["annual_co2_emissions_gt"], 1)

    return slope, intercept


def monte_carlo_simulation(
    initial_sea_ice_area_mkm2,
    co2_emission_rate_gt,
    slope_gt_per_year,
    include_uncertainty,
    uncertainty_m2,
    threshold_mkm2,
    num_simulations=1000,
    use_linear_trend=True,
):
    """
    Performs Monte Carlo simulations to estimate the distribution of years
    when Arctic sea ice area falls below the threshold.

    Parameters:
    initial_sea_ice_area_mkm2 (float): Initial sea ice area in million km².
    co2_emission_rate_gt (float): Current annual CO2 emissions in Gt CO2/year.
    slope_gt_per_year (float): Slope of the linear trend in Gt CO2/year².
    include_uncertainty (bool): Whether to include uncertainties in calculations.
    uncertainty_m2 (float): Standard deviation for uncertainty in m².
    threshold_mkm2 (float): Sea ice area threshold in million km².
    num_simulations (int): Number of Monte Carlo iterations.

    Returns:
    list: List of estimated years when sea ice area falls below the threshold.
    """
    estimated_years = []

    for _ in tqdm(range(num_simulations), desc="Running Monte Carlo Simulations"):
        # Initialize the model for each simulation with logging disabled to reduce verbosity
        ice_loss_model = ns_module.ArcticSeaIceLossModel(
            initial_sea_ice_area=initial_sea_ice_area_mkm2,
            co2_emission_rate_per_year=co2_emission_rate_gt,
            initial_cumulative_emissions=0,
            co2_model="linear" if use_linear_trend else "constant",
            co2_increase_per_year=slope_gt_per_year,
            include_uncertainty=include_uncertainty,
            uncertainty_m2=uncertainty_m2,
            enable_logging=False,  # Disable logging for simulations
        )

        # Simulate until threshold is reached
        years_until_threshold = ice_loss_model.simulate_until_threshold(
            threshold=threshold_mkm2
        )

        # Estimate the year when the threshold is reached
        # Assuming the current year is the last year in the CO2 data
        # This will be updated outside the function
        estimated_years.append(years_until_threshold)

    return estimated_years


def main():
    # Optional: Set a random seed for reproducibility when using uncertainties
    random.seed(42)
    np.random.seed(42)

    # Define file paths
    sea_ice_file = os.path.join("data", "osisaf_nh_SIE_monthly.mat")
    co2_file = os.path.join("data", "annual-co2-emissions-per-country.csv")
    co2_alternative_file = os.path.join("data", "co2.csv")

    # Data op
    use_alternative = True
    # Parameters for the Monte Carlo simulation
    include_uncertainty = True
    uncertainty_m2 = 0.3  # Standard deviation for uncertainty in m²
    include_linear_trend = True
    # The source for co2.csv is https://globalcarbonbudget.org/carbonbudget2023/
    # Data is in "Historical Budget" section, "Fossil emissions excluding carbonation" column

    # Data source for annual-co2-emissions-per-country.csv is https://ourworldindata.org/co2-and-greenhouse-gas-emissions#explore-data-on-co2-and-greenhouse-gas-emissions
    # Data is in "Annual CO₂ emissions" column with "Entity" as "World"

    # Check if data files exist
    for file_path in [sea_ice_file, co2_file, co2_alternative_file]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

    # Convert the MATLAB file to a pandas dataframe
    matlab_converter = convert_matlab_module.ConvertMatlab(sea_ice_file)

    co2_reader = None
    if use_alternative:
        # Read in the CO2 data using the alternative method
        co2_reader = csv_reader_module.CO2Reader(
            co2_alternative_file, use_alternative=True
        )
    else:
        # Read in the CO2 data using the standard method
        co2_reader = csv_reader_module.CO2Reader(co2_file, use_alternative=False)

    co2_df = co2_reader.get_df()

    # Display the first few rows of CO2 data
    print("Initial CO2 Data:")
    print(co2_df.head())

    # Filter CO2 data for years >= 1978
    co2_df = co2_df[co2_df["year"] >= 1978].reset_index(drop=True)

    # Compute the cumulative sum of the CO2 emissions since 1978
    co2_sum_gt = co2_df["annual_co2_emissions_gt"].sum()  # in Gt

    # Create a new column for the cumulative sum
    co2_df["cumulative_sum_gt"] = co2_df["annual_co2_emissions_gt"].cumsum()

    # Print the cumulative sum
    print(
        f"\nThe cumulative sum of CO2 emissions since 1978 is {co2_sum_gt:.2f} gigatonnes"
    )

    # Plot the cumulative sum of CO2 emissions
    sns.set_theme()
    plt.figure(figsize=(12, 6))
    plt.plot(
        co2_df["year"], co2_df["cumulative_sum_gt"], label="Cumulative CO2 Emissions"
    )

    # Add scatter plot markers for each year
    plt.scatter(co2_df["year"], co2_df["cumulative_sum_gt"], color="red", s=10)

    plt.xlabel("Year")
    plt.ylabel("Cumulative CO2 Emissions (Gt)")
    plt.title("Cumulative CO2 Emissions Since 1978")
    plt.legend()

    # Ensure the plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join("plots", "cumulative_co2.png"))
    plt.close()
    print("Cumulative CO2 emissions plot saved as 'plots/cumulative_co2.png'.")

    # Calculate the linear trend (slope) of CO2 emissions from 1978
    slope_gt_per_year, intercept_gt = calculate_linear_trend_co2(co2_df)
    print("\nCalculated linear trend of CO2 emissions:")
    print(f"Slope: {slope_gt_per_year:.4f} Gt CO2/year²")
    print(f"Intercept: {intercept_gt:.2f} Gt CO2")

    # Get the current global emissions (latest year's emissions)
    current_emissions_gt = co2_df["annual_co2_emissions_gt"].iloc[-1]

    # Get the initial sea ice area (latest available data) in million km²
    initial_sea_ice_area_mkm2 = matlab_converter.get_latest_sie_summer() / 1e6

    print(f"\nCurrent emissions: {current_emissions_gt:.2f} Gt CO2/year")
    print(f"Initial sea ice area: {initial_sea_ice_area_mkm2:.2f} million km²")

    # Initialize parameters for the Monte Carlo simulation
    threshold_mkm2 = 1  # 1 million km²
    num_simulations = 10000  # Number of Monte Carlo iterations

    # Perform Monte Carlo simulation

    estimated_years = monte_carlo_simulation(
        initial_sea_ice_area_mkm2=initial_sea_ice_area_mkm2,
        co2_emission_rate_gt=current_emissions_gt,
        slope_gt_per_year=slope_gt_per_year,
        include_uncertainty=include_uncertainty,
        uncertainty_m2=uncertainty_m2,
        threshold_mkm2=threshold_mkm2,
        num_simulations=num_simulations,
        use_linear_trend=include_linear_trend,
    )

    # Get the last year from the CO2 data
    last_year = co2_df["year"].max()

    # Convert estimated_years to actual years
    actual_estimated_years = [last_year + years for years in estimated_years]

    # Convert estimated_years to a DataFrame
    results_df = pd.DataFrame(actual_estimated_years, columns=["estimated_year"])

    # Analyze the results
    print("\nMonte Carlo Simulation Results:")
    print(results_df["estimated_year"].describe())

    # Plot the distribution of estimated years
    plt.figure(figsize=(12, 6))
    sns.histplot(
        results_df["estimated_year"],
        bins=range(
            results_df["estimated_year"].min(), results_df["estimated_year"].max() + 1
        ),
        color="skyblue",
        discrete=True,
    )
    plt.xlabel("Estimated Year")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of Estimated Years When Sea Ice Falls Below {threshold_mkm2} Million km²"
    )
    plt.axvline(
        results_df["estimated_year"].median(),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {results_df['estimated_year'].median():.0f}",
    )
    plt.legend()

    name = "estimated_year_distribution"
    # Save the plot
    if include_linear_trend and include_uncertainty:
        name = "estimated_year_distribution_linear_trend_uncertainty"
    elif include_linear_trend:
        name = "estimated_year_distribution_linear_trend"
    elif include_uncertainty:
        name = "estimated_year_distribution_uncertainty"

    if use_alternative:
        name = name + "_globalcarbonbudget.png"
    else:
        name = name + "_ourworldindata.png"
    plt.savefig(os.path.join("plots", name))
    plt.close()
    print(
        "Estimated year distribution plot saved as 'plots/estimated_year_distribution.png'."
    )

    # Optional: Save the simulation results to a CSV file
    results_df.to_csv(os.path.join("plots", "monte_carlo_results.csv"), index=False)
    print("Monte Carlo simulation results saved as 'plots/monte_carlo_results.csv'.")


if __name__ == "__main__":
    main()
