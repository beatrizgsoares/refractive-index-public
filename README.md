# refractive-index-public
Data and code regarding the envelope method paper

# Structure

    refractive-index-public/
    └── Data/
        ├── R/                  # Reflection measurements
        │   ├── W0000.CSV       # No coating
        │   └── W{N>=1}.CSV     # With coating; several measurements over time
        │
        └── T/                  # Transmission measurements
            ├── W0000.CSV       # No coating
            └── W{N>=1}.CSV     # With coating; several measurements over time

    └── functions.py          # Python file with all necessary functions
    └── n_experimental.ipynb  # Analysis of the experimental data
    └── simulations.ipynb     # Numerical simulations
    └── theory.ipynb          # Deduction of the mathematical formulas

    
