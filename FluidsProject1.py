
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


# Data from Figure E7.4a
pi1_E4a = [0.0195, 0.0175, 0.0155, 0.0132, 0.0113, 0.0101, 0.00939, 0.00893]
pi2_E4a = [4.01e3, 6.68e3, 9.97e3, 2.00e4, 3.81e4, 5.8e4, 8.0e4, 9.85e4]

# Convert lb/ft³ to slugs/ft³: divide by 32.2
fluid_properties = {
    "Gasoline": {
        "SI": {"density": 680, "viscosity": 3.1e-4},  # kg/m³, Pa·s
        "BG": {"density": 1.32, "viscosity": 6.5e-6}  # slugs/ft³, lb·s/ft²
    },
    "SAE 30 Oil": {
        "SI": {"density": 912, "viscosity": 3.8-1},  # kg/m³, Pa·s
        "BG": {"density": 1.77, "viscosity": 8e-3}  # slugs/ft³, lb·s/ft²
    },
    "Ethyl Alcohol": {
        "SI": {"density": 789, "viscosity": 1.19e-3},  # kg/m³, Pa·s
        "BG": {"density": 1.53, "viscosity": 2.49e-5}  # slugs/ft³, lb·s/ft²
    }
}

# Unit systems
unit_systems = {
    "SI": {
        "length": "meters",
        "velocity": "m/s",
        "pressure": "Pa",
        "density": "kg/m³",
        "viscosity": "Pa·s"
    },
    "BG": {
        "length": "feet",
        "velocity": "ft/s",
        "pressure": "lbf/ft²",
        "density": "slugs/ft³",
        "viscosity": "lb·s/ft²"
    }
}



def power_law(x, a, b):
    """Power law function for curve fitting"""
    return a * x**b

def calculate_reynolds(velocity, diameter, density, viscosity):
    """Calculate Reynolds number"""
    return (density * velocity * diameter) / viscosity

def blasius_equation(Re, a=0.1505, n=0.2458):
    """Blasius equation for friction factor"""
    return a * Re**(-n)

def calculate_pressure_drop(velocity, diameter, length, density, viscosity):
    """Calculate pressure drop using Blasius equation"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    
    if Re < 2100:
        st.warning("Warning: Flow is laminar! Results may not be accurate.")

    
    
    # Calculate pressure drop
    dp = (((blasius_equation(Re)) * density * velocity ** 2) / diameter) * 50
    
    return dp





def main():
    st.title("Pipe Flow Analysis Tool")
    st.write("Based on Blasius formula for turbulent flow in smooth pipes")
    
   # Add Hugging Face API token to session state if not present
    if 'api_token' not in st.session_state:
        st.session_state.api_token = ''
    
    
    # Add Hugging Face API token input to sidebar
    with st.sidebar:
        
        
                # Unit system selection
        unit_system = st.radio("Select Unit System", ["SI", "BG"])
        
        # Fluid selection
        fluid_type = st.radio("Fluid Selection", ["Preset Fluid", "Custom Fluid"])
        
        if fluid_type == "Preset Fluid":
            fluid = st.selectbox("Select Fluid", list(fluid_properties.keys()))
            density = fluid_properties[fluid][unit_system]["density"]
            viscosity = fluid_properties[fluid][unit_system]["viscosity"]
            
            # Display fluid properties
            st.write(f"Density: {density:.6f} {unit_systems[unit_system]['density']}")
            st.write(f"Viscosity: {viscosity:.6e} {unit_systems[unit_system]['viscosity']}")
        else:
            # Custom fluid properties with default values
            if unit_system == "SI":
                default_density = 1000.0  # water in kg/m³
                default_viscosity = 0.001  # water in Pa·s
            else:
                default_density = 1.94  # water in slugs/ft³ (62.4/32.2)
                default_viscosity = 2.09e-5  # water in lb·s/ft²
            
            density = st.number_input(
                f"Fluid Density ({unit_systems[unit_system]['density']})",
                min_value=0.1,
                value=default_density,
                format="%.6f"
            )
            viscosity = st.number_input(
                f"Fluid Viscosity ({unit_systems[unit_system]['viscosity']})",
                min_value=1e-6,
                value=default_viscosity,
                format="%.2e"
            )
        
        # Pipe parameters
        diameter = st.number_input(
            f"Pipe Diameter ({unit_systems[unit_system]['length']})",
            min_value=0.001,
            value=4.0,
            format="%.4e"
        )
        
        
        # Flow velocity
        velocity = st.number_input(
            f"Flow Velocity ({unit_systems[unit_system]['velocity']})",
            min_value=0.1,
            value=2.0
        )



    # Calculate pressure drop
    dp = calculate_pressure_drop(velocity, diameter, length, density, viscosity)
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    
    # Display results
    st.header("Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Reynolds Number", f"{Re:,.2f}")
    
    with col2:
        st.metric(
            f"Pressure Drop ({unit_systems[unit_system]['pressure']})", 
            f"{dp:.5f}"
        )
    
    # Create plots
    st.header("Analysis Plots")
    
    tab1, tab2 = st.tabs(["Figure E7.4a Data", "Blasius Equation"])
    
    with tab1:
        # Perform curve fit on Figure E7.4a data
        popt, pcov = curve_fit(power_law, pi2_E4a, pi1_E4a)
        
        # Generate points for the fitted curve
        Re_fit = np.logspace(np.log10(min(pi2_E4a)), np.log10(max(pi2_E4a)), 100)
        pi1_fit = power_law(Re_fit, *popt)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(Re_fit, pi1_fit, 'k-', label=f'Curve Fit (y = {popt[0]:.4f}x^{popt[1]:.4f})')
        ax1.plot(pi2_E4a, pi1_E4a, 'b*', label='Experimental Data')
        ax1.plot(Re, blasius_equation(Re), 'ro', label='Current Operating Point') 
        ax1.set_xlabel('Reynolds Number (Re)')
        ax1.set_ylabel('Π₁ (D∆p/ρV²)')
        ax1.grid(True, which="both", ls="-")
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax1.loglog(Re_fit, pi1_fit, 'k-', label=f'Curve Fit (y = {popt[0]:.4f}x^{popt[1]:.4f})')
        ax1.loglog(pi2_E4a, pi1_E4a, 'b*', label='Experimental Data')
        ax1.loglog(Re, blasius_equation(Re), 'ro', label='Current Operating Point') 
        ax1.set_xlabel('Reynolds Number (Re)')
        ax1.set_ylabel('Π₁ (D∆p/ρV²)')
        ax1.grid(True, which="both", ls="-")
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        
        
        st.write(f"""
        Curve fit equation: Π₁ = {popt[0]:.4f} × Re^({popt[1]:.4f})
        
        This compares well with the Blasius formula coefficient of 0.1582 and exponent of -0.25
        """)
    
    with tab2:
        Re = np.logspace(3.5, 5, 100)
        f = 4 * blasius_equation(Re)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(Re, f, 'b-', label='Blasius equation')
        
        # Mark current operating point
        current_f = 4 * blasius_equation(Re)
        ax2.plot(Re, current_f, 'ro', label='Current Operating Point')
        
        ax2.set_xlabel('Reynolds Number (Re)')
        ax2.set_ylabel('Friction Factor (f)')
        ax2.grid(True, which="both", ls="-")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig2)

    # Additional information
    st.header("Additional Information")
    st.write(f"""
    Current fluid properties in {unit_system} units:
    - Density: {density:.6f} {unit_systems[unit_system]['density']}
    - Viscosity: {viscosity:.6e} {unit_systems[unit_system]['viscosity']}
   
    
    Notes:
    - The Blasius equation is valid for Reynolds numbers between 4,000 and 100,000
    - The pipe is assumed to be smooth-walled
    - Flow is assumed to be fully developed
    """)

if __name__ == "__main__":
    main()
