# QS Case Study DTU

## Overview

**QS Case Study DTU** models the loss of Arctic sea ice in response to global CO₂ emissions. It processes historical CO₂ emission data and sea ice extent measurements to simulate and predict when Arctic sea ice may fall below critical thresholds using Monte Carlo simulations amd the Noetzli-Stroeve model.

## Project Structure

```
QS-case-study-dtu/
├── .git
├── README.md
├── convert_matlab.py       # Converts MATLAB .mat files to pandas DataFrames
├── csv_reader.py           # Reads and processes CO₂ emission CSV data
├── noetz_stroeve.py        # Implements the Arctic Sea Ice Loss Model
├── part2.py                # Main script to perform simulations and generate plots
└── plots                   # Directory to store generated plots
```

## Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/QS-case-study-dtu.git
   cd QS-case-study-dtu
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas scipy seaborn matplotlib numpy tqdm
   ```

3. **Prepare Data Files**

   Place the required data files in a `data` directory:
   - `osisaf_nh_SIE_monthly.mat` – Monthly sea ice extent data.
   - `annual-co2-emissions-per-country.csv` – Standard CO₂ emissions data.
   - `co2.csv` – Alternative CO₂ emissions data.

4. **Run the Simulation**
   ```bash
   python part2.py
   ```

5. **View Outputs**

   - **Plots**: Check the `plots` folder for generated visualizations.
   - **Results**: Simulation results are saved as `monte_carlo_results.csv` in the `plots` directory.

## Notes

- **Data Sources**:
  - Sea Ice: [OSISAF Sea Ice Index](https://osisaf.met.no/)
  - CO₂ Emissions:
    - [Our World in Data](https://ourworldindata.org/co2-and-greenhouse-gas-emissions)
    - [Global Carbon Budget 2023](https://globalcarbonbudget.org/carbonbudget2023/)
