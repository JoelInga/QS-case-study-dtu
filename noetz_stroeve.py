import random
import logging


class ArcticSeaIceLossModel:
    def __init__(
        self,
        initial_sea_ice_area,
        co2_emission_rate_per_year,
        initial_cumulative_emissions=0,
        co2_model="constant",  # 'constant' or 'linear'
        co2_increase_per_year=0,  # Only used if co2_model is 'linear'
        include_uncertainty=False,  # Boolean flag to include uncertainties
        uncertainty_m2=0.3,  # Uncertainty in m²
        enable_logging=False,  # Flag to enable or disable logging
        log_level=logging.INFO,  # Logging level
    ):
        """
        Initializes the model.

        Parameters:
        initial_sea_ice_area (float): Initial Arctic September sea-ice area in million km².
        co2_emission_rate_per_year (float): Annual global CO2 emissions in Gt CO2.
        initial_cumulative_emissions (float): Cumulative CO2 emissions up to the base year in tons.
        co2_model (str): 'constant' or 'linear' CO2 emission model.
        co2_increase_per_year (float): Annual increase in CO2 emission rate (Gt CO2/year), used if co2_model is 'linear'.
        include_uncertainty (bool): Whether to include uncertainties in sea-ice loss calculations.
        uncertainty_m2 (float): The magnitude of uncertainty in m² (± value).
        enable_logging (bool): Whether to enable logging.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Configure logger
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(handler)
            self.logger.setLevel(log_level)
        else:
            # If logging is disabled, add a NullHandler to suppress logs
            self.logger.addHandler(logging.NullHandler())

        self.initial_sea_ice_area_km2 = initial_sea_ice_area * 1e6  # Convert to km²
        self.co2_emission_rate_per_year_tons = (
            co2_emission_rate_per_year * 1e9
        )  # Convert Gt to tons
        self.annual_loss_per_ton_co2_m2 = (
            3  # Base m² loss per ton of CO2 (from the model)
        )
        self.current_sea_ice_area_km2 = self.initial_sea_ice_area_km2
        self.logger.info(
            f"Initial sea ice area: {self.initial_sea_ice_area_km2/1e6:.2f} million km²"
        )
        self.cumulative_emissions_tons = (
            initial_cumulative_emissions  # Cumulative emissions in tons
        )
        self.co2_model = co2_model.lower()
        if self.co2_model not in ["constant", "linear"]:
            raise ValueError("co2_model must be either 'constant' or 'linear'.")
        self.co2_increase_per_year_tons = (
            co2_increase_per_year * 1e9
        )  # Convert Gt to tons
        self.include_uncertainty = include_uncertainty
        self.uncertainty_m2 = uncertainty_m2

    def simulate_year(self):
        """
        Simulates one year of CO2 emissions and sea-ice loss.
        """
        # Update CO2 emission rate if using linear model
        if self.co2_model == "linear" and hasattr(self, "co2_increase_per_year_tons"):
            self.co2_emission_rate_per_year_tons += self.co2_increase_per_year_tons
            self.logger.info(
                f"Updated yearly CO2 emissions rate: {self.co2_emission_rate_per_year_tons/1e9:.2f} Gt CO2/year"
            )

        # Increase cumulative emissions
        self.cumulative_emissions_tons += self.co2_emission_rate_per_year_tons
        self.logger.info(
            f"Yearly emissions: {self.co2_emission_rate_per_year_tons/1e9:.2f} Gt CO2"
        )

        # Determine annual loss per ton CO2 with uncertainty if enabled
        if self.include_uncertainty:
            # Sample annual loss per ton from normal distribution
            loss_per_ton = random.gauss(
                self.annual_loss_per_ton_co2_m2, self.uncertainty_m2
            )
            # Ensure loss_per_ton is non-negative
            loss_per_ton = max(loss_per_ton, 0)
            self.logger.info(
                f"Annual loss per ton CO2 (with uncertainty): {loss_per_ton:.2f} m²/ton"
            )
        else:
            loss_per_ton = self.annual_loss_per_ton_co2_m2
            self.logger.info(
                f"Annual loss per ton CO2 (constant): {loss_per_ton:.2f} m²/ton"
            )

        # Calculate total loss in m² based on cumulative emissions
        total_loss_m2 = self.cumulative_emissions_tons * loss_per_ton
        self.logger.info(f"Total loss: {total_loss_m2:.2f} m²")

        # Convert total loss to km²
        total_loss_km2 = total_loss_m2 / 1e6

        # Update current sea ice area
        self.current_sea_ice_area_km2 = max(
            self.initial_sea_ice_area_km2 - total_loss_km2, 0
        )
        self.logger.info(
            f"Current sea ice area: {self.current_sea_ice_area_km2/1e6:.2f} million km²"
        )
        self.logger.info(
            f"Cumulative emissions: {self.cumulative_emissions_tons/1e9:.2f} Gt CO2"
        )

    def simulate_until_threshold(self, threshold=1):
        """
        Simulates the model until the sea-ice area falls below a certain threshold.

        Parameters:
        threshold (float): Threshold sea-ice area in million km².

        Returns:
        int: The number of years it takes to reach the threshold.
        """
        years = 0
        threshold_km2 = threshold * 1e6  # Convert to km²
        self.logger.info(
            f"Starting simulation until sea ice area falls below {threshold} million km²."
        )
        while self.current_sea_ice_area_km2 > threshold_km2:
            self.simulate_year()
            years += 1
            # Prevent infinite loop by setting a reasonable maximum
            if years > 1000:
                self.logger.warning("Threshold not reached within 1000 years.")
                break
        self.logger.info(f"Reached threshold in {years} years.")
        return years

    def estimate_sea_ice_area(self, years):
        """
        Estimates the sea-ice area after a given number of years.

        Parameters:
        years (int): The number of years to simulate.

        Returns:
        float: Estimated sea-ice area in million km².
        """
        self.logger.info(f"Estimating sea ice area after {years} years.")
        for _ in range(years):
            self.simulate_year()
        estimated_area = (
            self.current_sea_ice_area_km2 / 1e6
        )  # Convert back to million km²
        self.logger.info(
            f"Estimated sea ice area after {years} years: {estimated_area:.2f} million km²"
        )
        return estimated_area
