# refractive-index-public
Data and code regarding the envelope method paper

# Structure

    refractive-index-public/
    └── Data/
        ├── PAZO_nk_TL+G.txt    # Ellipsometer Measurements
        │
        ├── R{n}/               # Reflection measurements (n=1: Sample A; n=2: Sample B)
        │   ├── W0000.CSV       # No coating
        │   └── W{N>=1}.CSV     # With coating; several measurements over time
        │
        └── T{n}/               # Transmission measurements (n=1: Sample A; n=2: Sample B)
            ├── W0000.CSV       # No coating
            └── W{N>=1}.CSV     # With coating; several measurements over time

    └── functions.py          # Python file with all necessary functions
    └── n_experimental.ipynb  # Analysis of the experimental data
    └── simulations.ipynb     # Numerical simulations
    └── theory.ipynb          # Deduction of the mathematical formulas

    
