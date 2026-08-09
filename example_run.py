"""
Optera Supply Chain Framework - Minimal Quickstart Example

This script demonstrates how to import and run Optera in 1 line or modularly.
"""

import optera

def main():
    print("Executing Optera Integrated Supply Chain Engine...")
    
    # Run the integrated 5-layer pipeline
    report = optera.run(
        data_path="./demand_forecasting.csv",
        output_dir=".",
        total_budget=500000.0,
        num_simulations=1000,
        horizon_days=90
    )
    
    print("Optera pipeline completed! Report saved to:", report)

if __name__ == "__main__":
    main()
