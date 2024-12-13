import os
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from noetz_stroeve import ArcticSeaIceLossModel

# Constants
CO2_TO_METRIC_TON = 1e9  # 1 Gt = 1e9 metric tons
INITIAL_SEA_ICE_AREA_MKM2 = 5.57  # Initial sea ice area in million km²
CURRENT_EMISSIONS_GT = 37.15  # Current annual CO2 emissions in Gt CO2/year
POPULATION = 8.1e9  # 8.1 billion inhabitants
UNCERTAINTY_M2 = 0.3  # ±0.3 m² uncertainty in loss_per_ton
THRESHOLD_MKM2 = 1  # Threshold sea ice area in million km²
NUM_SIMULATIONS = 10000  # Number of Monte Carlo simulations
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
        stopping_years (list of int): The year each simulation stopped.
    """
    results = []
    thresholds_reached = []
    stopping_years = []
    for sim in tqdm(range(num_simulations), desc="Monte Carlo Simulations"):
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
        stopping_years.append(years[-1])  # Last simulated year
    return results, thresholds_reached, stopping_years


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
    Plot sea ice area over time and the number of running simulations for different emission scenarios.
    """
    # Create a figure with two subplots: top for sea ice area, bottom for running simulations
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    max_year = STARTING_YEAR
    color_map = plt.get_cmap("tab10")  # Choose a color map
    scenario_colors = {}  # To store colors for each scenario

    for idx, (name, (emission_rate, slope)) in enumerate(scenarios.items()):
        color = color_map(idx % 10)  # Assign a color from the color map
        scenario_colors[name] = color  # Store the color for later use

        if include_uncertainty:
            mc_results, thresholds_reached, stopping_years = monte_carlo_simulation(
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

            # Plot mean and confidence interval on ax1
            ax1.plot(years, mean_sea_ice, label=f"{name} Mean", color=color)
            ax1.fill_between(
                years,
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=color,
                label=f"{name} 95% CI",
            )

            # Annotate percentage of simulations reaching the threshold
            pct_reached = sum(thresholds_reached) / num_simulations * 100
            ax1.text(
                years[-1],
                mean_sea_ice[-1],
                f"{name}: {pct_reached:.1f}% reached threshold",
                fontsize=8,
                verticalalignment="bottom",
                color=color,
            )

            max_year = max(max_year, years[-1])

            # Process stopping years to count running simulations
            stopping_years_sorted = np.sort(stopping_years)
            running_counts = []
            year_range = np.arange(STARTING_YEAR, years[-1] + 1)
            for year in year_range:
                # Number of simulations that stop after the current year
                running = np.sum(stopping_years_sorted >= year)
                running_counts.append(running)

            # Plot running simulations on ax2
            ax2.plot(
                year_range,
                running_counts,
                label=f"{name} Running Simulations",
                color=color,
            )
            # Increase ax2 y-axis limit by 20% for better visualization, make sure lower limit is 0
            ax2.set_ylim(top=max(running_counts) * 1.4, bottom=0)

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

            ax1.plot(years_history, sea_ice_history, label=name, color=color)
            max_year = max(max_year, years_history[-1])

            # Annotate if the threshold was not reached
            if not reached_threshold:
                ax1.text(
                    years_history[-1],
                    sea_ice_history[-1],
                    f"{name}: Threshold not reached",
                    fontsize=8,
                    verticalalignment="bottom",
                    color=color,
                )

    # Configure ax1 (Sea Ice Area)
    ax1.axhline(
        y=threshold_mkm2, color="k", linestyle="--", label="Threshold Sea Ice Area"
    )
    ax1.set_ylabel("Arctic Sea Ice Area (million km²)")
    ax1.set_title("Arctic Sea Ice Area Over Time Under Different Emission Scenarios")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Configure ax2 (Running Simulations)
    if include_uncertainty:
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Running Simulations")
        ax2.set_title("Number of Running Simulations Over Time")
        ax2.legend(loc="upper right")
        ax2.grid(True)
    else:
        ax2.axis("off")

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
    Also plots the number of running simulations below.
    """
    # Create a figure with two subplots: top for sea ice area, bottom for running simulations

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 14), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Combine standard and decreased scenarios into one dictionary
    all_scenarios = {
        **emission_scenarios,
        **{name: (rate, slope) for name, rate, slope in decreased_scenarios},
    }
    max_year = STARTING_YEAR
    color_map = plt.get_cmap("tab10")  # Choose a color map
    scenario_colors = {}  # To store colors for each scenario

    for idx, (name, (emission_rate, slope)) in enumerate(all_scenarios.items()):
        color = color_map(idx % 10)  # Assign a color from the color map
        scenario_colors[name] = color  # Store the color for later use

        if include_uncertainty:
            mc_results, thresholds_reached, stopping_years = monte_carlo_simulation(
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

            # Plot mean and confidence interval on ax1
            ax1.plot(years, mean_sea_ice, label=f"{name} Mean", color=color)
            ax1.fill_between(
                years,
                lower_bound,
                upper_bound,
                alpha=0.2,
                color=color,
                label=f"{name} 95% CI",
            )

            # Annotate percentage of simulations reaching the threshold
            pct_reached = sum(thresholds_reached) / num_simulations * 100
            ax1.text(
                years[-1],
                mean_sea_ice[-1],
                f"{name}: {pct_reached:.1f}% reached threshold",
                fontsize=8,
                verticalalignment="bottom",
                color=color,
            )

            max_year = max(max_year, years[-1])

            # Process stopping years to count running simulations
            stopping_years_sorted = np.sort(stopping_years)
            running_counts = []
            year_range = np.arange(STARTING_YEAR, years[-1] + 1)
            for year in year_range:
                # Number of simulations that stop after the current year
                running = np.sum(stopping_years_sorted >= year)
                running_counts.append(running)

            # Plot running simulations on ax2
            ax2.plot(
                year_range,
                running_counts,
                label=f"{name} Running Simulations",
                color=color,
            )
            # Increase ax2 y-axis limit by 200 for better visualization, make sure lower limit is 0
            ax2.set_ylim(top=max(running_counts) * 1.4, bottom=0)

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

            ax1.plot(years_history, sea_ice_history, label=name, color=color)
            max_year = max(max_year, years_history[-1])

            # Annotate if the threshold was not reached
            if not reached_threshold:
                ax1.text(
                    years_history[-1],
                    sea_ice_history[-1],
                    f"{name}: Threshold not reached",
                    fontsize=8,
                    verticalalignment="bottom",
                    color=color,
                )

    # Configure ax1 (Sea Ice Area)
    ax1.axhline(
        y=threshold_mkm2, color="k", linestyle="--", label="Threshold Sea Ice Area"
    )
    ax1.set_ylabel("Arctic Sea Ice Area (million km²)")
    ax1.set_title("Arctic Sea Ice Area Over Time Under All Emission Scenarios")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True)
    # Configure ax2 (Running Simulations)
    if include_uncertainty:
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Running Simulations")
        ax2.set_title("Number of Running Simulations Over Time")
        ax2.legend(loc="upper right")
        ax2.grid(True)
    else:
        ax2.axis("off")
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


def calculate_budget_with_uncertainty(
    num_simulations=10000,
    loss_per_ton_mean=3,
    loss_per_ton_std=0.3,
    threshold_mkm2=1,
    initial_sea_ice_mkm2=5.57,
    population=8.1e9,
):
    """
    Calculate the total and per capita carbon budget with uncertainty using Monte Carlo simulations.

    Returns:
        budget_gt_samples (numpy.ndarray): Array of total carbon budget samples in Gt CO2.
        budget_per_capita_tons_samples (numpy.ndarray): Array of per capita budget samples in metric tons CO2 per person.
    """
    # Sample loss_per_ton from a normal distribution
    loss_per_ton_samples = np.random.normal(
        loc=loss_per_ton_mean, scale=loss_per_ton_std, size=num_simulations
    )

    # Calculate leftover budget for each sample
    total_loss_allowed_km2 = (initial_sea_ice_mkm2 - threshold_mkm2) * 1e6
    total_loss_allowed_m2 = total_loss_allowed_km2 * 1e6
    leftover_budget_tons = total_loss_allowed_m2 / loss_per_ton_samples
    leftover_budget_gt = leftover_budget_tons / CO2_TO_METRIC_TON  # Convert to Gt

    # Calculate per capita budget for each sample
    leftover_budget_tons = leftover_budget_gt * CO2_TO_METRIC_TON
    budget_per_person_tons = leftover_budget_tons / population

    return leftover_budget_gt, budget_per_person_tons


def plot_budget_uncertainty(
    budget_gt_samples,
    budget_per_capita_tons_samples,
    plot_filename="plots/carbon_budget_with_uncertainty.png",
):
    """
    Plot the distribution of total and per capita carbon budgets.

    Args:
        budget_gt_samples (numpy.ndarray): Array of total carbon budget samples in Gt CO2.
        budget_per_capita_tons_samples (numpy.ndarray): Array of per capita budget samples in metric tons CO2 per person.
        plot_filename (str): Path to save the plot.
    """
    plt.figure(figsize=(14, 6))

    # Plot for Total Carbon Budget
    plt.subplot(1, 2, 1)
    plt.hist(budget_gt_samples, bins=50, color="skyblue", edgecolor="black")
    plt.axvline(
        np.percentile(budget_gt_samples, 2.5),
        color="red",
        linestyle="dashed",
        linewidth=1,
    )
    plt.axvline(
        np.percentile(budget_gt_samples, 97.5),
        color="red",
        linestyle="dashed",
        linewidth=1,
    )
    # Add mean and median lines
    plt.axvline(
        np.mean(budget_gt_samples), color="green", linestyle="dashed", linewidth=1
    )
    plt.axvline(
        np.median(budget_gt_samples), color="purple", linestyle="dashed", linewidth=1
    )
    # Add labels and title
    plt.title("Total Carbon Budget Before Reaching Threshold")
    plt.xlabel("Carbon Budget (Gt CO₂)")
    plt.ylabel("Frequency")
    plt.legend(["95% CI", "Mean", "Median"])
    plt.grid(True)

    # Plot for Per Capita Carbon Budget
    plt.subplot(1, 2, 2)
    plt.hist(
        budget_per_capita_tons_samples, bins=50, color="lightgreen", edgecolor="black"
    )
    plt.axvline(
        np.percentile(budget_per_capita_tons_samples, 2.5),
        color="red",
        linestyle="dashed",
        linewidth=1,
    )
    plt.axvline(
        np.percentile(budget_per_capita_tons_samples, 97.5),
        color="red",
        linestyle="dashed",
        linewidth=1,
    )
    # Add mean and median lines
    plt.axvline(
        np.mean(budget_per_capita_tons_samples),
        color="green",
        linestyle="dashed",
        linewidth=1,
    )
    plt.axvline(
        np.median(budget_per_capita_tons_samples),
        color="purple",
        linestyle="dashed",
        linewidth=1,
    )

    plt.title("Per Capita Carbon Budget Before Reaching Threshold")
    plt.xlabel("Carbon Budget (metric tons CO₂ per person)")
    plt.ylabel("Frequency")
    plt.legend(["95% CI", "Mean", "Median"])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Carbon budget uncertainty plot saved as '{plot_filename}'.")


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

    # 2. Calculate Carbon Budget with Uncertainty
    print("### Calculating Carbon Budget with Uncertainty ###\n")

    # Perform Monte Carlo simulation for budget calculations
    budget_gt_samples, budget_per_capita_tons_samples = (
        calculate_budget_with_uncertainty(
            num_simulations=NUM_SIMULATIONS,
            loss_per_ton_mean=3,
            loss_per_ton_std=UNCERTAINTY_M2,
            threshold_mkm2=THRESHOLD_MKM2,
            initial_sea_ice_mkm2=INITIAL_SEA_ICE_AREA_MKM2,
            population=POPULATION,
        )
    )

    # Calculate statistics
    budget_gt_mean = np.mean(budget_gt_samples)
    budget_gt_median = np.median(budget_gt_samples)
    budget_gt_2_5 = np.percentile(budget_gt_samples, 2.5)
    budget_gt_97_5 = np.percentile(budget_gt_samples, 97.5)

    budget_per_capita_mean = np.mean(budget_per_capita_tons_samples)
    budget_per_capita_median = np.median(budget_per_capita_tons_samples)
    budget_per_capita_2_5 = np.percentile(budget_per_capita_tons_samples, 2.5)
    budget_per_capita_97_5 = np.percentile(budget_per_capita_tons_samples, 97.5)

    print("### Total Carbon Budget ###")
    print(f"Mean: {budget_gt_mean:.2f} Gt CO₂")
    print(f"Median: {budget_gt_median:.2f} Gt CO₂")
    print(
        f"95% Confidence Interval: [{budget_gt_2_5:.2f}, {budget_gt_97_5:.2f}] Gt CO₂\n"
    )

    print("### Per Capita Carbon Budget ###")
    print(f"Mean: {budget_per_capita_mean:.4f} metric tons CO₂ per person")
    print(f"Median: {budget_per_capita_median:.4f} metric tons CO₂ per person")
    print(
        f"95% Confidence Interval: [{budget_per_capita_2_5:.4f}, {budget_per_capita_97_5:.4f}] metric tons CO₂ per person\n"
    )

    # Plot the budget distributions
    plot_budget_uncertainty(
        budget_gt_samples=budget_gt_samples,
        budget_per_capita_tons_samples=budget_per_capita_tons_samples,
        plot_filename="plots/carbon_budget_with_uncertainty.png",
    )

    # 3. Analyze different emission scenarios
    print("\n### Analyzing Different Emission Scenarios ###\n")

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

    # 4. Simulate decreased emissions scenarios with linear decrease
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
            f"Linear Decrease to 0 in {int(decrease_period_years *1.2)} years",
            CURRENT_EMISSIONS_GT,
            -CURRENT_EMISSIONS_GT / (decrease_period_years * 1.2),
        ),
        (
            f"Linear Decrease to 0 in {int(decrease_period_years * 1.4)} years",
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

    # 5. Combined Plot of All Scenarios
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
