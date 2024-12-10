import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from noetz_stroeve import ArcticSeaIceLossModel

# Constants
CO2_TO_METRIC_TON = 1e9  # 1 Gt = 1e9 metric tons
INITIAL_SEA_ICE_AREA_MKM2 = 5.57  # Initial sea ice area in million km²
CURRENT_EMISSIONS_GT = 37.15  # Current annual CO2 emissions in Gt CO2/year
POPULATION = 8.1e9  # 8.1 billion inhabitants
UNCERTAINTY_M2 = 0.3  # ±0.3 m² uncertainty
THRESHOLD_MKM2 = 1  # Threshold sea ice area in million km²
NUM_SIMULATIONS = 10000  # Number of Monte Carlo iterations
STARTING_YEAR = 2022  # Starting year for simulations

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)


def calculate_leftover_budget(
    loss_per_ton=3, threshold_mkm2=1, initial_sea_ice_mkm2=5.57
):
    """
    Calculate the leftover carbon budget before reaching the sea ice threshold.
    """
    total_loss_allowed_km2 = (initial_sea_ice_mkm2 - threshold_mkm2) * 1e6
    total_loss_allowed_m2 = total_loss_allowed_km2 * 1e6
    leftover_budget_tons = total_loss_allowed_m2 / loss_per_ton
    leftover_budget_gt = leftover_budget_tons / CO2_TO_METRIC_TON  # Convert to Gt
    return leftover_budget_gt


def calculate_per_capita_budget(leftover_budget_gt, population=8.1e9):
    """
    Calculate the per capita carbon budget.
    """
    leftover_budget_tons = leftover_budget_gt * CO2_TO_METRIC_TON
    budget_per_person_tons = leftover_budget_tons / population
    return budget_per_person_tons


def simulate_scenario(
    emission_rate_gt,
    slope_gt_per_year,
    include_uncertainty,
    uncertainty_m2,
    initial_sea_ice_mkm2,
    threshold_mkm2,
):
    """
    Simulate a single emission scenario and return the sea ice area history.
    The simulation stops if sea ice area starts increasing again or if the threshold is reached.

    Returns:
        years_history (list): List of years simulated.
        sea_ice_history (list): Corresponding sea ice areas.
        reached_threshold (bool): Whether the threshold was reached.
    """
    model = ArcticSeaIceLossModel(
        initial_sea_ice_area=initial_sea_ice_mkm2,
        co2_emission_rate_per_year=emission_rate_gt,
        co2_model="linear" if slope_gt_per_year != 0 else "constant",
        co2_increase_per_year=slope_gt_per_year,
        include_uncertainty=include_uncertainty,
        uncertainty_m2=uncertainty_m2,
        enable_logging=False,
    )
    sea_ice_history = []
    years_history = []
    reached_threshold = False
    current_emission_rate = emission_rate_gt

    for year in range(1000):  # Simulate up to 1000 years or until stopping condition
        current_sea_ice = model.current_sea_ice_area_km2 / 1e6  # Convert to million km²
        sea_ice_history.append(current_sea_ice)
        years_history.append(STARTING_YEAR + year)

        # Check if threshold is reached
        if current_sea_ice <= threshold_mkm2:
            reached_threshold = True
            break

        # Check for negative current yearly emissions
        if model.co2_emission_rate_per_year_tons < 0:
            break

        # Simulate the next year
        model.simulate_year()

        # Update the emission rate
        if slope_gt_per_year != 0:
            current_emission_rate += slope_gt_per_year
            if current_emission_rate < 0:
                current_emission_rate = 0  # Cap emission rate at zero
                # Update the model to stop decreasing emissions
                model.co2_emission_rate_per_year = current_emission_rate
                model.co2_increase_per_year = 0  # Set slope to zero

    return years_history, sea_ice_history, reached_threshold


def monte_carlo_simulation(
    emission_rate_gt,
    slope_gt_per_year,
    include_uncertainty,
    uncertainty_m2,
    initial_sea_ice_mkm2,
    threshold_mkm2,
    num_simulations=1000,
):
    """
    Perform Monte Carlo simulations and collect sea ice area histories.

    Returns:
        results (list of lists): Each sublist contains the sea ice area history for a simulation.
        thresholds_reached (list of bool): Whether each simulation reached the threshold.
    """
    results = []
    thresholds_reached = []
    for _ in tqdm(range(num_simulations), desc="Monte Carlo Simulations"):
        years, sea_ice, reached_threshold = simulate_scenario(
            emission_rate_gt=emission_rate_gt,
            slope_gt_per_year=slope_gt_per_year,
            include_uncertainty=include_uncertainty,
            uncertainty_m2=uncertainty_m2,
            initial_sea_ice_mkm2=initial_sea_ice_mkm2,
            threshold_mkm2=threshold_mkm2,
        )
        results.append(sea_ice)
        thresholds_reached.append(reached_threshold)
    return results, thresholds_reached


def plot_scenarios(
    scenarios,
    initial_sea_ice_mkm2,
    threshold_mkm2,
    include_uncertainty,
    uncertainty_m2,
    num_simulations=1000,
    plot_name="sea_ice_scenarios",
):
    """
    Plot sea ice area over time for different emission scenarios.
    For scenarios with uncertainty, plot mean and confidence bands.
    """
    plt.figure(figsize=(12, 8))
    max_year = STARTING_YEAR

    for name, (emission_rate, slope) in scenarios.items():
        if include_uncertainty:
            mc_results, thresholds_reached = monte_carlo_simulation(
                emission_rate_gt=emission_rate,
                slope_gt_per_year=slope,
                include_uncertainty=True,
                uncertainty_m2=uncertainty_m2,
                initial_sea_ice_mkm2=initial_sea_ice_mkm2,
                threshold_mkm2=threshold_mkm2,
                num_simulations=num_simulations,
            )
            # Determine the maximum length among all simulations
            max_length = max(len(sim) for sim in mc_results)
            # Initialize a 2D array with NaNs
            sea_ice_matrix = np.full((num_simulations, max_length), np.nan)
            for i, sim in enumerate(mc_results):
                sea_ice_matrix[i, : len(sim)] = sim

            # Compute statistics ignoring NaNs
            mean_sea_ice = np.nanmean(sea_ice_matrix, axis=0)
            lower_bound = np.nanpercentile(sea_ice_matrix, 2.5, axis=0)
            upper_bound = np.nanpercentile(sea_ice_matrix, 97.5, axis=0)
            years = np.arange(STARTING_YEAR, STARTING_YEAR + max_length)

            plt.plot(years, mean_sea_ice, label=f"{name} Mean")
            plt.fill_between(
                years, lower_bound, upper_bound, alpha=0.2, label=f"{name} 95% CI"
            )

            max_year = max(max_year, years[-1])

            # Annotate if some simulations did not reach the threshold
            if not all(thresholds_reached):
                plt.text(
                    years[-1],
                    mean_sea_ice[-1],
                    f"{name}: {sum(thresholds_reached)/num_simulations*100:.1f}% reached threshold",
                    fontsize=8,
                    verticalalignment="bottom",
                )
            elif all(thresholds_reached) and slope < 0:
                print(f"{name}: All simulations reached threshold")
                plt.text(
                    years[-1],
                    mean_sea_ice[-1],
                    f"{name}: All simulations reached threshold",
                    fontsize=8,
                    verticalalignment="bottom",
                )
        else:
            # Run a single simulation without uncertainty
            years_history, sea_ice_history, reached_threshold = simulate_scenario(
                emission_rate_gt=emission_rate,
                slope_gt_per_year=slope,
                include_uncertainty=False,
                uncertainty_m2=0,
                initial_sea_ice_mkm2=initial_sea_ice_mkm2,
                threshold_mkm2=threshold_mkm2,
            )

            plt.plot(years_history, sea_ice_history, label=name)
            max_year = max(max_year, years_history[-1])

            # Annotate if the threshold was not reached
            if not reached_threshold:
                plt.text(
                    years_history[-1],
                    sea_ice_history[-1],
                    f"{name}: Threshold not reached",
                    fontsize=8,
                    verticalalignment="bottom",
                )

    plt.axhline(
        y=threshold_mkm2, color="k", linestyle="--", label="Threshold Sea Ice Area"
    )
    plt.xlabel("Year")
    plt.ylabel("Arctic Sea Ice Area (million km²)")
    plt.title("Arctic Sea Ice Area Over Time Under Different Emission Scenarios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Set x-axis limit
    plt.xlim(left=STARTING_YEAR, right=max_year + 10)

    # Save the plot
    uncertainty_label = (
        "_with_uncertainty" if include_uncertainty else "_without_uncertainty"
    )
    plot_filename = f"plots/{plot_name}{uncertainty_label}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Emission scenarios plot saved as '{plot_filename}'.")


def plot_combined_scenarios(
    emission_scenarios,
    decreased_scenarios,
    initial_sea_ice_mkm2,
    threshold_mkm2,
    include_uncertainty,
    uncertainty_m2,
    num_simulations=1000,
):
    """
    Plot all emission scenarios (both standard and decreased) on a single plot with confidence bands if uncertainty is included.
    """
    plt.figure(figsize=(14, 10))
    # Combine standard and decreased scenarios into one dictionary
    all_scenarios = {
        **emission_scenarios,
        **{name: (rate, slope) for name, rate, slope in decreased_scenarios},
    }
    max_year = STARTING_YEAR

    for name, (emission_rate, slope) in all_scenarios.items():
        if include_uncertainty:
            mc_results, thresholds_reached = monte_carlo_simulation(
                emission_rate_gt=emission_rate,
                slope_gt_per_year=slope,
                include_uncertainty=True,
                uncertainty_m2=uncertainty_m2,
                initial_sea_ice_mkm2=initial_sea_ice_mkm2,
                threshold_mkm2=threshold_mkm2,
                num_simulations=num_simulations,
            )
            # Determine the maximum length among all simulations
            max_length = max(len(sim) for sim in mc_results)
            # Initialize a 2D array with NaNs
            sea_ice_matrix = np.full((num_simulations, max_length), np.nan)
            for i, sim in enumerate(mc_results):
                sea_ice_matrix[i, : len(sim)] = sim

            # Compute statistics ignoring NaNs
            mean_sea_ice = np.nanmean(sea_ice_matrix, axis=0)
            lower_bound = np.nanpercentile(sea_ice_matrix, 2.5, axis=0)
            upper_bound = np.nanpercentile(sea_ice_matrix, 97.5, axis=0)
            years = np.arange(STARTING_YEAR, STARTING_YEAR + max_length)

            plt.plot(years, mean_sea_ice, label=f"{name} Mean")
            plt.fill_between(
                years, lower_bound, upper_bound, alpha=0.2, label=f"{name} 95% CI"
            )

            max_year = max(max_year, years[-1])

            # Annotate if some simulations did not reach the threshold
            if not all(thresholds_reached):
                print(
                    f"{name}: {sum(thresholds_reached)/num_simulations*100:.1f}% reached threshold"
                )
                print(f"years, last sea ice: {years[-1]}, {mean_sea_ice[-1]}")
                plt.text(
                    years[-1],
                    mean_sea_ice[-1],
                    f"{name}: {sum(thresholds_reached)/num_simulations*100:.1f}% reached threshold",
                    fontsize=8,
                    verticalalignment="bottom",
                )
            elif all(thresholds_reached) and slope < 0:
                print(f"{name}: All simulations reached threshold")
                plt.text(
                    years[-1],
                    mean_sea_ice[-1],
                    f"{name}: All simulations reached threshold",
                    fontsize=8,
                    verticalalignment="bottom",
                )
        else:
            # Run a single simulation without uncertainty
            years_history, sea_ice_history, reached_threshold = simulate_scenario(
                emission_rate_gt=emission_rate,
                slope_gt_per_year=slope,
                include_uncertainty=False,
                uncertainty_m2=0,
                initial_sea_ice_mkm2=initial_sea_ice_mkm2,
                threshold_mkm2=threshold_mkm2,
            )

            plt.plot(years_history, sea_ice_history, label=name)
            max_year = max(max_year, years_history[-1])

            # Annotate if the threshold was not reached
            if not reached_threshold:
                plt.text(
                    years_history[-1],
                    sea_ice_history[-1],
                    f"{name}: Threshold not reached",
                    fontsize=8,
                    verticalalignment="bottom",
                )

    plt.axhline(
        y=threshold_mkm2, color="k", linestyle="--", label="Threshold Sea Ice Area"
    )
    plt.xlabel("Year")
    plt.ylabel("Arctic Sea Ice Area (million km²)")
    plt.title("Arctic Sea Ice Area Over Time Under All Emission Scenarios")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Set x-axis limit
    plt.xlim(left=STARTING_YEAR, right=max_year + 10)

    # Save the combined plot
    uncertainty_label = (
        "_with_uncertainty" if include_uncertainty else "_without_uncertainty"
    )
    plot_filename = f"plots/combined_all_scenarios{uncertainty_label}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Combined all scenarios plot saved as '{plot_filename}'.")


def main():
    # 1. Answer the two questions
    print("### Answering the Questions ###\n")

    # Question 1: Leftover carbon budget before reaching 1 million km² of sea ice
    leftover_budget_gt = calculate_leftover_budget(
        loss_per_ton=3,
        threshold_mkm2=THRESHOLD_MKM2,
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
    )
    print(f"1. Leftover Carbon Budget: {leftover_budget_gt:.2f} Gt CO₂")

    # Question 2: Per capita carbon budget
    budget_per_person_tons = calculate_per_capita_budget(
        leftover_budget_gt=leftover_budget_gt, population=POPULATION
    )
    print(
        f"2. Per Capita Carbon Budget: {budget_per_person_tons:.2f} metric tons CO₂ per person\n"
    )

    # 2. Analyze different emission scenarios
    print("### Analyzing Different Emission Scenarios ###\n")

    # Define standard emission scenarios
    emission_scenarios = {
        "Constant Emissions": (CURRENT_EMISSIONS_GT, 0),
        "Linear Increase (1 Gt/year²)": (CURRENT_EMISSIONS_GT, 1),
        "Linear Increase (2 Gt/year²)": (CURRENT_EMISSIONS_GT, 2),
        "Linear Increase (5 Gt/year²)": (CURRENT_EMISSIONS_GT, 5),
    }

    # Plot emission scenarios without uncertainty
    plot_scenarios(
        scenarios=emission_scenarios,
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
        threshold_mkm2=THRESHOLD_MKM2,
        include_uncertainty=False,
        uncertainty_m2=0,
        num_simulations=NUM_SIMULATIONS,
        plot_name="sea_ice_scenarios",
    )

    # Plot emission scenarios with uncertainty
    plot_scenarios(
        scenarios=emission_scenarios,
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
        threshold_mkm2=THRESHOLD_MKM2,
        include_uncertainty=True,
        uncertainty_m2=UNCERTAINTY_M2,
        num_simulations=NUM_SIMULATIONS,
        plot_name="sea_ice_scenarios",
    )

    # 3. Simulate decreased emissions scenarios with linear decrease
    print("\n### Simulating Decreased Emissions Scenarios (Linear Decrease) ###\n")

    # Define decreased emissions scenarios with linear decreases
    decrease_period_years = 60  # Number of years over which emissions decrease to 0

    decreased_scenarios = [
        (
            f"Linear Decrease to 0 in {decrease_period_years} years",
            CURRENT_EMISSIONS_GT,
            -CURRENT_EMISSIONS_GT / decrease_period_years,  # Negative slope
        ),
        (
            f"Linear Decrease to 0 in {decrease_period_years *1.2} years",
            CURRENT_EMISSIONS_GT,
            -CURRENT_EMISSIONS_GT / (decrease_period_years * 1.2),
        ),
        (
            f"Linear Decrease to 0 in {decrease_period_years * 1.4} years",
            CURRENT_EMISSIONS_GT,
            -CURRENT_EMISSIONS_GT / (decrease_period_years * 1.4),
        ),
    ]

    # Plot decreased emissions scenarios without uncertainty
    plot_scenarios(
        scenarios={name: (rate, slope) for name, rate, slope in decreased_scenarios},
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
        threshold_mkm2=THRESHOLD_MKM2,
        include_uncertainty=False,
        uncertainty_m2=0,
        num_simulations=NUM_SIMULATIONS,
        plot_name="decreased_emissions_scenarios",
    )

    # Plot decreased emissions scenarios with uncertainty
    plot_scenarios(
        scenarios={name: (rate, slope) for name, rate, slope in decreased_scenarios},
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
        threshold_mkm2=THRESHOLD_MKM2,
        include_uncertainty=True,
        uncertainty_m2=UNCERTAINTY_M2,
        num_simulations=NUM_SIMULATIONS,
        plot_name="decreased_emissions_scenarios",
    )

    # 4. Combined Plot of All Scenarios
    print("\n### Creating Combined Plot of All Scenarios ###\n")

    # Combine standard and decreased emission scenarios into one plot
    plot_combined_scenarios(
        emission_scenarios=emission_scenarios,
        decreased_scenarios=decreased_scenarios,
        initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
        threshold_mkm2=THRESHOLD_MKM2,
        include_uncertainty=True,
        uncertainty_m2=UNCERTAINTY_M2,
        num_simulations=NUM_SIMULATIONS,
    )


if __name__ == "__main__":
    main()
