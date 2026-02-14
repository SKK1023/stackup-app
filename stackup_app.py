import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Dimension:
    """
    Represents a single linear dimension in the stackup.
    
    Attributes:
        name: Description of the part/dimension.
        nominal: The target value.
        tol_plus: Positive tolerance (upper deviation).
        tol_minus: Negative tolerance (lower deviation).
        is_subtractive: True if this dimension reduces the gap (e.g., part thickness),
                        False if it increases the gap (e.g., housing cavity).
    """
    name: str
    nominal: float
    tol_plus: float
    tol_minus: float
    is_subtractive: bool = False

    @property
    def effective_nominal(self) -> float:
        return -self.nominal if self.is_subtractive else self.nominal

class ToleranceStackup:
    def __init__(self, name: str):
        self.name = name
        self.dimensions: List[Dimension] = []

    def add_dimension(self, name: str, nominal: float, tol_plus: float, tol_minus: float, is_subtractive: bool = False):
        """Adds a dimension to the stackup."""
        dim = Dimension(name, nominal, tol_plus, tol_minus, is_subtractive)
        self.dimensions.append(dim)

    def calculate_nominal_gap(self) -> float:
        """Calculates the theoretical nominal gap."""
        return abs(sum(d.effective_nominal for d in self.dimensions))

    def calculate_worst_case(self):
        """
        Calculates Worst Case (WC) limits.
        Max Gap = Nominal + Sum(Positive Tolerances)
        Min Gap = Nominal - Sum(Negative Tolerances)
        """
        nominal_gap = self.calculate_nominal_gap()
        
        total_tol_plus = sum(d.tol_plus for d in self.dimensions)
        total_tol_minus = sum(d.tol_minus for d in self.dimensions)

        max_gap = nominal_gap + total_tol_plus
        min_gap = nominal_gap - total_tol_minus
        
        return {
            "method": "Worst Case",
            "nominal": nominal_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "tol_plus": total_tol_plus,
            "tol_minus": total_tol_minus
        }

    def calculate_rss(self, sigma_level: float = 3.0):
        """
        Calculates Root Sum Squares (RSS) limits.
        Statistical combination of tolerances.
        
        Args:
            sigma_level: The sigma level of the input tolerances. 
                         Standard industry practice assumes input tolerances are 3-sigma.
        """
        nominal_gap = self.calculate_nominal_gap()
        
        # RSS aligned with user's simple summation logic
        rss_var_plus = sum(d.tol_plus**2 for d in self.dimensions)
        rss_var_minus = sum(d.tol_minus**2 for d in self.dimensions)

        rss_tol_plus = np.sqrt(rss_var_plus)
        rss_tol_minus = np.sqrt(rss_var_minus)
        
        return {
            "method": "RSS",
            "nominal": nominal_gap,
            "min_gap": nominal_gap - rss_tol_minus,
            "max_gap": nominal_gap + rss_tol_plus,
            "sigma_level": sigma_level
        }

    def create_figure(self):
        """
        Generates a matplotlib figure resembling CAD tolerance analysis plots.
        """
        wc = self.calculate_worst_case()
        rss = self.calculate_rss()
        
        mean = rss['nominal']
        rss_tol_span = rss['max_gap'] - rss['min_gap']
        wc_tol_span = wc['max_gap'] - wc['min_gap']
        
        rss_min = rss['min_gap']
        rss_max = rss['max_gap']
        wc_min = wc['min_gap']
        wc_max = wc['max_gap']

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate X-axis data points for the curve
        # We span wide enough to see the Worst Case limits
        span = max(wc_tol_span, rss_tol_span) * 0.8
        if span == 0: span = 1.0
        x_axis = np.linspace(mean - span, mean + span, 1000)
        
        # 1. Plot RSS Normal Distribution Curve
        # For asymmetric, we approximate a normal curve centered on the mean of the RSS limits
        # or just use the average sigma.
        std_dev = (rss_tol_span / 2.0) / 3.0 if rss_tol_span > 0 else 0.001
        y_axis = norm.pdf(x_axis, mean, std_dev)
        
        # Plot filled curve
        ax.plot(x_axis, y_axis, color='#4472C4', linewidth=2, label='Distribution')
        ax.fill_between(x_axis, y_axis, color='#4472C4', alpha=0.3)

        # 2. Plot Limits
        # Nominal
        ax.axvline(mean, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Nominal')
        
        # RSS Limits (Green)
        ax.axvline(rss_min, color='#548235', linestyle='--', linewidth=2, label='RSS Limits')
        ax.axvline(rss_max, color='#548235', linestyle='--', linewidth=2)

        # Worst Case Limits (Red)
        ax.axvline(wc_min, color='#C00000', linestyle=':', linewidth=2, label='Worst Case')
        ax.axvline(wc_max, color='#C00000', linestyle=':', linewidth=2)

        # Annotations
        y_max = max(y_axis)
        def add_label(x, color, text, y_pos):
            ax.annotate(f'{text}\n{x:.3f}', xy=(x, y_pos), xytext=(0, 10), 
                        textcoords='offset points', ha='center', color=color,
                        fontweight='bold', fontsize=9)

        add_label(mean, 'black', 'Nominal', y_max * 0.95)
        add_label(rss_min, '#548235', 'RSS Min', y_max * 0.5)
        add_label(rss_max, '#548235', 'RSS Max', y_max * 0.5)
        add_label(wc_min, '#C00000', 'WC Min', y_max * 0.2)
        add_label(wc_max, '#C00000', 'WC Max', y_max * 0.2)

        # Annotations and Styling
        ax.set_title("Stackup Distribution", fontsize=16)
        ax.set_xlabel("Gap Dimension (mm)", fontsize=12)
        
        # Clean up axes
        ax.get_yaxis().set_visible(False) # Hide Y axis (density)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="Stackup Calculator", layout="wide")
    st.title("Tolerance Stackup Calculator")

    # Initialize Session State
    if 'stackup' not in st.session_state:
        st.session_state.stackup = ToleranceStackup("Assembly Stackup")

    # Fix for AttributeError: Reset state if old data format is detected
    if st.session_state.stackup.dimensions:
        if not hasattr(st.session_state.stackup.dimensions[0], 'tol_plus'):
            st.session_state.stackup = ToleranceStackup("Assembly Stackup")
            st.rerun()

    if 'calculate_pressed' not in st.session_state:
        st.session_state.calculate_pressed = False

    if 'waiting_for_tolerance' not in st.session_state:
        st.session_state.waiting_for_tolerance = False

    # --- Helper Callback ---
    def add_dim_callback(mode, is_sub):
        # mode: 'nom' or 'tol'
        key = f"{mode}_input"
        val = st.session_state.get(key)
        val = val if val is not None else 0.0
        
        count = len(st.session_state.stackup.dimensions) + 1
        direction_str = "Neg" if is_sub else "Pos"
        
        if mode == 'nom':
            name = f"Dim {count}"
            # Add new dimension with 0 tolerances
            st.session_state.stackup.add_dimension(name, val, 0.0, 0.0, is_subtractive=is_sub)
            st.session_state.waiting_for_tolerance = True
        else:
            # Update the LAST added dimension
            if st.session_state.stackup.dimensions:
                last_dim = st.session_state.stackup.dimensions[-1]
                if not is_sub: # Positive Button -> Tol Plus
                    last_dim.tol_plus = val
                else: # Negative Button -> Tol Minus
                    last_dim.tol_minus = val
            st.session_state.waiting_for_tolerance = False
            
        st.session_state[key] = None
        st.session_state.calculate_pressed = False

    # --- Sidebar: Add Dimensions ---
    with st.sidebar:
        st.header("Add Dimension")

        # --- Nominal Section ---
        st.subheader("Nominal")
        # Disable Nominal input if waiting for tolerance
        nom_disabled = st.session_state.waiting_for_tolerance
        st.number_input("Nominal Value", value=None, step=0.1, key="nom_input", disabled=nom_disabled)
        
        c1, c2 = st.columns(2)
        c1.button("Positive (+)", key="btn_nom_fwd", on_click=add_dim_callback, args=("nom", False), disabled=nom_disabled)
        c2.button("Negative (-)", key="btn_nom_bwd", on_click=add_dim_callback, args=("nom", True), disabled=nom_disabled)

        # --- Tolerance Section ---
        st.subheader("Tolerance")
        # Disable Tolerance input if there are no dimensions to edit
        tol_disabled = len(st.session_state.stackup.dimensions) == 0
        st.number_input("Tolerance Value", value=None, step=0.01, min_value=0.0, key="tol_input", disabled=tol_disabled)
        
        c3, c4 = st.columns(2)
        c3.button("Positive (+)", key="btn_tol_fwd", on_click=add_dim_callback, args=("tol", False), disabled=tol_disabled)
        c4.button("Negative (-)", key="btn_tol_bwd", on_click=add_dim_callback, args=("tol", True), disabled=tol_disabled)

        st.divider()

        # --- Actions ---
        if st.button("Calculate", type="primary", use_container_width=True):
            st.session_state.calculate_pressed = True
        
        if st.button("Clear All", use_container_width=True):
            st.session_state.previous_stackup = st.session_state.stackup
            st.session_state.previous_calculate_pressed = st.session_state.calculate_pressed
            st.session_state.stackup = ToleranceStackup("Assembly Stackup")
            st.session_state.calculate_pressed = False
            st.session_state.waiting_for_tolerance = False
            st.rerun()

        if st.session_state.get("previous_stackup") is not None:
            if st.button("Previous Value (Undo Clear)", use_container_width=True):
                st.session_state.stackup = st.session_state.previous_stackup
                st.session_state.calculate_pressed = st.session_state.get("previous_calculate_pressed", False)
                st.session_state.waiting_for_tolerance = False
                st.rerun()

    # --- Main Area ---
    stack = st.session_state.stackup

    if not stack.dimensions:
        st.info("👈 Please add dimensions using the sidebar to begin analysis.")
        return

    # 1. Display Dimensions Table (Editable)
    st.subheader("Stackup Dimensions")
    dim_data = []
    for d in stack.dimensions:
        dim_data.append({
            "ID": d.name,
            "Type": "Negative (-)" if d.is_subtractive else "Positive (+)",
            "Nominal": d.nominal,
            "Positive Tolerance": d.tol_plus,
            "Negative Tolerance": d.tol_minus,
            "Effective Value": d.effective_nominal
        })
    
    df = pd.DataFrame(dim_data)
    # Ensure column order matches request: Nominal -> Positive Tolerance -> Negative Tolerance
    cols = ["ID", "Type", "Nominal", "Positive Tolerance", "Negative Tolerance", "Effective Value"]
    df = df[cols]
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")

    # Sync edits back to stackup object
    # We reconstruct the dimensions list from the edited dataframe
    new_dimensions = []
    for index, row in edited_df.iterrows():
        is_sub = True if row["Type"] == "Negative (-)" else False
        new_dimensions.append(Dimension(
            name=row["ID"],
            nominal=float(row["Nominal"]),
            tol_plus=float(row["Positive Tolerance"]),
            tol_minus=float(row["Negative Tolerance"]),
            is_subtractive=is_sub
        ))
    
    # Update session state if changes detected (simple check)
    if len(new_dimensions) != len(stack.dimensions) or \
       any(d1.nominal != d2.nominal or d1.tol_plus != d2.tol_plus or d1.tol_minus != d2.tol_minus \
           for d1, d2 in zip(stack.dimensions, new_dimensions)):
        stack.dimensions = new_dimensions
        st.rerun()

    if st.session_state.calculate_pressed:
        # 2. Calculations
        wc = stack.calculate_worst_case()
        rss = stack.calculate_rss()

        # Calculate separate totals for Forward and Backward
        fwd_dims = [d for d in stack.dimensions if not d.is_subtractive]
        bwd_dims = [d for d in stack.dimensions if d.is_subtractive]

        total_fwd_nom = sum(d.nominal for d in fwd_dims)
        total_bwd_nom = sum(d.nominal for d in bwd_dims)

        # Calculate Nominal Gap
        nominal_gap_value = abs(total_fwd_nom - total_bwd_nom)

        # Max/Min Gap comes directly from Worst Case calculation now
        max_gap = wc['max_gap']
        min_gap = wc['min_gap']
        
        # Derived total tolerances for display
        total_plus_tolerance = wc['tol_plus']
        total_minus_tolerance = wc['tol_minus']

        # 3. Metrics Display
        st.subheader("Analysis Results")

        # Row 1: Breakdown
        c1, c2 = st.columns(2)
        c1.info(f"**Nominal Breakdown**\n\nPos Dims: {total_fwd_nom:.3f}\n\nNeg Dims: {total_bwd_nom:.3f}")
        c2.warning(f"**Tolerance Breakdown**\n\nTotal Pos Tol (+): {total_plus_tolerance:.3f}\n\nTotal Neg Tol (-): {total_minus_tolerance:.3f}")

        # Row 2: Final Results
        col1, col2, col3 = st.columns(3)
        col1.metric("Nominal Gap", f"{nominal_gap_value:.3f} mm")
        col2.metric("Max Gap", f"{max_gap:.3f} mm", delta=f"+{total_plus_tolerance:.3f}")
        col3.metric("Min Gap", f"{min_gap:.3f} mm", delta=f"-{total_minus_tolerance:.3f}")

        # Display Worst Case and RSS
        st.write(f"Worst Case Min Gap: {wc['min_gap']:.3f}")
        st.write(f"Worst Case Max Gap: {wc['max_gap']:.3f}")
        st.write(f"RSS Min Gap: {rss['min_gap']:.3f}")
        st.write(f"RSS Max Gap: {rss['max_gap']:.3f}")
        
        # 4. Plot
        st.subheader("Gap Distribution Graph")
        fig = stack.create_figure()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
