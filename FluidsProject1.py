
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import requests
import json

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
def query_huggingface(prompt, api_token):
    """
    Query the Llama model through Hugging Face's API
    
    Args:
        prompt (str): The input prompt for the model
        api_token (str): Hugging Face API token
        
    Returns:
        str: Model response or error message
    """
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Request parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 1000,  # Increased max length
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "top_k": 50,
            "return_full_text": False  # Only return the generated response
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Debug logging
        st.write("Debug - Raw API Response:", response.text)
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                elif "text" in result[0]:
                    return result[0]["text"]
            return str(result[0])
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            elif "text" in result:
                return result["text"]
        
        # If we get here, try to return any text content we can find
        return str(result)
            
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {str(e)}"
    except json.JSONDecodeError:
        return f"Error: Failed to parse API response - Raw response: {response.text}"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"

def create_fluid_mechanics_prompt(user_question, context=""):
    """
    Create a well-formatted prompt for fluid mechanics questions
    
    Args:
        user_question (str): The user's question
        context (str): Additional context about the system
        
    Returns:
        str: Formatted prompt
    """
    prompt = f"""Based on the following context and question, provide a detailed technical explanation:

Context:
{context}

Question: {user_question}

Provide a clear, technically accurate response that explains the concepts, includes relevant calculations if needed, and relates to the given context where appropriate."""
    
    return prompt


def power_law(x, a, b):
    """Power law function for curve fitting"""
    return a * x**b

def calculate_reynolds(velocity, diameter, density, viscosity):
    """Calculate Reynolds number"""
    return (density * velocity * diameter) / viscosity

def blasius_equation(Re, a=0.1582, n=0.25):
    """Blasius equation for friction factor"""
    return a * Re**(-n)

def calculate_pressure_drop(velocity, diameter, length, density, viscosity):
    """Calculate pressure drop using Blasius equation"""
    Re = calculate_reynolds(velocity, diameter, density, viscosity)
    
    if Re < 2100:
        st.warning("Warning: Flow is laminar! Results may not be accurate.")
    
    # Calculate friction factor using Blasius equation
    f = 4 * blasius_equation(Re)
    
    # Calculate pressure drop
    dp = f * (length/diameter) * density * (velocity**2) / 2
    
    return dp





def main():
    st.title("Pipe Flow Analysis Tool")
    st.write("Based on Blasius formula for turbulent flow in smooth pipes")
    
   # Add Hugging Face API token to session state if not present
    if 'api_token' not in st.session_state:
        st.session_state.api_token = ''
    
    
    # Add Hugging Face API token input to sidebar
    with st.sidebar:
        st.header("API Configuration")
        api_token = st.text_input(
            "Enter Hugging Face API Token",
            value=st.session_state.api_token,
            type="password",
            help="Get your API token from https://huggingface.co/settings/tokens"
        )
        if api_token:
            st.session_state.api_token = api_token
            
        # Add a test connection button
        if st.button("Test API Connection"):
            with st.spinner("Testing connection..."):
                test_response = query_huggingface("Test connection", api_token)
                if "Error" not in test_response:
                    st.success("Connection successful!")
                else:
                    st.error(f"Connection failed: {test_response}")
        st.header("Input Parameters")
        
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
            min_value=0.1,
            value=4.0
        )
        
        length = st.number_input(
            f"Pipe Length ({unit_systems[unit_system]['length']})",
            min_value=1.0,
            value=50.0
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
            f"{dp:.2f}"
        )
    
    # Create plots
    st.header("Analysis Plots")
    
    tab1, tab2, tab3 = st.tabs(["Figure E7.4a Data", "Blasius Equation", "LLM Assistant"])
    
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

    with tab3:
        st.header("Advanced Fluid Mechanics Assistant")
        st.info("""This assistant uses the Llama 1B model to answer questions about fluid mechanics. 
                It has access to your current system parameters and can provide insights about your specific setup.""")
        
        if not st.session_state.api_token:
            st.warning("Please enter your Hugging Face API token in the sidebar to use the Advanced Assistant.")
            st.info("Get a free API token at https://huggingface.co/settings/tokens")
            st.markdown("""
            Setup instructions:
            1. Create a free account at [Hugging Face](https://huggingface.co)
            2. Go to Settings > Access Tokens
            3. Create a new token with 'read' access
            4. Paste the token in the sidebar
            """)
        else:
            # Create context information about the current system
            context = f"""
Current System Parameters:
- Reynolds Number: {calculate_reynolds(velocity, diameter, density, viscosity):.2f}
- Flow Velocity: {velocity} {unit_systems[unit_system]['velocity']}
- Pipe Diameter: {diameter} {unit_systems[unit_system]['length']}
- Pipe Length: {length} {unit_systems[unit_system]['length']}
- Fluid Density: {density} {unit_systems[unit_system]['density']}
- Fluid Viscosity: {viscosity} {unit_systems[unit_system]['viscosity']}
- Calculated Pressure Drop: {dp:.2f} {unit_systems[unit_system]['pressure']}
- Flow Regime: {"Turbulent" if calculate_reynolds(velocity, diameter, density, viscosity) > 4000 else "Transitional" if calculate_reynolds(velocity, diameter, density, viscosity) > 2300 else "Laminar"}
"""
            
            # Create columns for different preset questions
            st.write("### Quick Questions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Analyze Flow Regime"):
                    user_query = "Based on the Reynolds number and other parameters, what are the key characteristics of the current flow regime? What implications does this have for the system behavior?"
                    st.session_state.last_query = user_query
                    
            with col2:
                if st.button("📊 Optimization Tips"):
                    user_query = "How could we optimize this pipe system to reduce pressure drop while maintaining the same flow rate? What are the key parameters to adjust?"
                    st.session_state.last_query = user_query
                    
            with col3:
                if st.button("🛠️ Practical Considerations"):
                    user_query = "What practical considerations and potential issues should be considered for this pipe system based on the current parameters?"
                    st.session_state.last_query = user_query

            # Custom question input
            st.write("### Custom Question")
            user_query = st.text_area(
                "Ask any fluid mechanics question:",
                value=st.session_state.get('last_query', ''),
                height=100,
                help="Ask about flow characteristics, system optimization, practical applications, or any other fluid mechanics topic"
            )

            # Process the question when submitted
            if user_query:
                with st.spinner("Analyzing your question..."):
                    # Create the prompt and query the model
                    prompt = create_fluid_mechanics_prompt(user_query, context)
                    
                    # Debug - Show the prompt
                    st.write("Debug - Prompt sent to API:", prompt)
                    
                    response = query_huggingface(prompt, st.session_state.api_token)
                    
                    if "Error" not in response:
                        st.markdown("### Analysis:")
                        # Remove the prompt from the response if it's included
                        cleaned_response = response.replace(prompt, "").strip()
                        if cleaned_response:
                            st.markdown(cleaned_response)
                        else:
                            st.error("The model didn't generate a meaningful response. Please try again or rephrase your question.")
                    else:
                        st.error(response)
                        st.info("""Troubleshooting Tips:
                        1. Verify your API token is valid
                        2. Check your internet connection
                        3. Try refreshing the page
                        4. Start with a simpler question
                        5. Check the debug output above""")

if __name__ == "__main__":
    main()