import streamlit as st
import pickle
import pandas as pd
from utils import get_recommendations_gensim, get_recommendations_surprise
from PIL import Image
import requests
from io import BytesIO
import re
from bs4 import BeautifulSoup
import sys
import traceback
import os
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Make the sidebar wider with custom CSS
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 300px;
            max-width: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_segmentation_data():
    """Load all segmentation data from pickle files"""
    try:
        results = {}
        
        # Load RFM data
        with open('data/rfm_segments.pkl', 'rb') as f:
            results['rfm'] = pickle.load(f)
        
        # Load KMeans results
        with open('data/kmeans_segments.pkl', 'rb') as f:
            results['kmeans'] = pickle.load(f)
        
        # Load Hierarchical results
        with open('data/hierarchical_segments.pkl', 'rb') as f:
            results['hierarchical'] = pickle.load(f)
        
            
        return results
    except Exception as e:
        st.error(f"Error loading segmentation data: {str(e)}")
        st.error("Please check if all pickle files exist in the data/ directory")
        return None

def load_all_data():
    """Load all segmentation data AND custom rules"""
    results = {}
    # Load base segmentation data
    try:
        # Ensure data directory exists - useful if running first time
        os.makedirs('data', exist_ok=True) 
        
        with open('data/rfm_segments.pkl', 'rb') as f:
            results['rfm'] = pickle.load(f)
        with open('data/kmeans_segments.pkl', 'rb') as f:
            results['kmeans'] = pickle.load(f)
        with open('data/hierarchical_segments.pkl', 'rb') as f:
            results['hierarchical'] = pickle.load(f)
    except FileNotFoundError as e:
         st.error(f"Error loading base segmentation file: {e}. Please ensure 'rfm_segments.pkl', 'kmeans_segments.pkl', and 'hierarchical_segments.pkl' exist in the 'data' directory.")
         return None # Cannot proceed without base data
    except Exception as e:
        st.error(f"Error loading base segmentation data: {str(e)}")
        return None # Return None on other critical errors

    # Load custom manual rules if they exist
    rules_filepath = 'data/custom_manual_rules.pkl'
    results['custom_manual_rules'] = None # Default to None
    if os.path.exists(rules_filepath):
        try:
            with open(rules_filepath, 'rb') as f:
                loaded_rules = pickle.load(f)
                # Basic validation
                if isinstance(loaded_rules, list):
                     results['custom_manual_rules'] = loaded_rules
                     # Use sidebar for less intrusive feedback
                     # st.sidebar.success("Loaded saved custom rules.") 
                else:
                     st.sidebar.warning(f"Custom rules file ({rules_filepath}) invalid format. Using defaults.")
                     
        except Exception as e:
            st.sidebar.warning(f"Could not load custom rules: {str(e)}. Using defaults.")
            # Keep results['custom_manual_rules'] as None
            
    return results

def Get_ManualSegmentation(rfm_df):
    """
    Creates a UI for manual RFM segmentation, applies it to the current rfm_df, 
    AND saves the defined rules for later use.
    
    Parameters:
    rfm_df (pandas.DataFrame): DataFrame with RFM scores
    
    Returns:
    dict: Dictionary containing segmented dataframe and metrics for the *current* run.
    """
    st.subheader("Define Your Own RFM Segments")
    st.write("Use the inputs below to create customer segments based on RFM score thresholds.")
    st.write("_These rules can be saved and used later for assigning segments to new customers._")

    # Default segment names and settings (Can be used as initial suggestions)
    default_segments = [
        {"name": "Champions", "r_value": 4, "f_value": 4, "m_value": 4, "description": "High value, frequent buyers who purchased recently"},
        {"name": "Loyal Customers", "r_value": 3, "f_value": 3, "m_value": 3, "description": "Regular spenders who purchase often"},
        {"name": "Potential Loyalists", "r_value": 2, "f_value": 2, "m_value": 2, "description": "Recent customers with moderate spending"},
        {"name": "At Risk Customers", "r_value": 1, "f_value": 1, "m_value": 1, "description": "Past loyal customers who haven't purchased recently"}
    ]
    
    num_segments = st.slider("Number of Segments to Define", min_value=1, max_value=10, value=len(default_segments), key="num_manual_segments")
    
    segments_container = st.container()
    
    # List to store the rule definitions {name, r_min, f_min, m_min}
    defined_rules = [] 
    
    with segments_container:
        for i in range(num_segments):
            st.markdown(f"---")
            st.markdown(f"#### Segment {i+1} Definition")
            
            # Use default name if available, otherwise generate one
            default_name = default_segments[i]["name"] if i < len(default_segments) else f"Segment {i+1}"
            r_default = default_segments[i].get("r_value", int(rfm_df['R'].quantile(0.5))) if i < len(default_segments) else int(rfm_df['R'].quantile(0.5))
            f_default = default_segments[i].get("f_value", int(rfm_df['F'].quantile(0.5))) if i < len(default_segments) else int(rfm_df['F'].quantile(0.5))
            m_default = default_segments[i].get("m_value", int(rfm_df['M'].quantile(0.5))) if i < len(default_segments) else int(rfm_df['M'].quantile(0.5))
            
            # Display description if available
            if i < len(default_segments) and "description" in default_segments[i]:
                 st.markdown(f"**Suggestion:** *{default_segments[i]['description']}*")

            seg_name = st.text_input(f"Segment Name", default_name, key=f"rule_name_{i}")
            
            # Use columns for RFM threshold inputs
            col1, col2, col3 = st.columns(3)
            with col1:
                r_min = st.number_input(f"Minimum R Score", min_value=int(rfm_df['R'].min()), max_value=int(rfm_df['R'].max()), value=r_default, key=f"rule_r_min_{i}")
            with col2:
                f_min = st.number_input(f"Minimum F Score", min_value=int(rfm_df['F'].min()), max_value=int(rfm_df['F'].max()), value=f_default, key=f"rule_f_min_{i}")
            with col3:
                m_min = st.number_input(f"Minimum M Score", min_value=int(rfm_df['M'].min()), max_value=int(rfm_df['M'].max()), value=m_default, key=f"rule_m_min_{i}")

            # Store the defined rule
            defined_rules.append({
                "name": seg_name,
                "r_min": r_min,
                "f_min": f_min,
                "m_min": m_min
            })

    st.markdown("---")
    # Add button to apply segmentation AND save the rules
    results = None
    col_apply, col_save_info = st.columns([1, 3]) # Layout button
    
    with col_apply:
        apply_and_save = st.button("Apply Segmentation & Save Rules", key="apply_save_rules_btn")
        
    if apply_and_save:
        if len(defined_rules) != num_segments:
            st.error(f"Mismatch in rule definitions. Expected {num_segments}, found {len(defined_rules)}. Please refresh.")
        else:
            # --- Apply rules to the current DataFrame (for display) ---
            custom_seg_df = rfm_df.copy()
            custom_seg_df['Custom_Segment'] = 'Other Customers' # Default category
            
            # Apply rules in the defined priority order
            for rule in defined_rules:
                mask = (
                    (custom_seg_df['R'] >= rule['r_min']) & 
                    (custom_seg_df['F'] >= rule['f_min']) & 
                    (custom_seg_df['M'] >= rule['m_min']) &
                    (custom_seg_df['Custom_Segment'] == 'Other Customers') # Only assign if not already assigned by a higher priority rule
                )
                custom_seg_df.loc[mask, 'Custom_Segment'] = rule['name']
            
            # Add the 'Other Customers' name explicitly to the list of segments if it exists
            final_segment_names = [rule['name'] for rule in defined_rules]
            if 'Other Customers' in custom_seg_df['Custom_Segment'].unique():
                 if 'Other Customers' not in final_segment_names:
                      # Technically, 'Other Customers' isn't a rule, but needed for assignment logic later
                      # We save the main rules, the assignment function will handle the default
                      pass 


            # --- Save the defined rules to a file ---
            rules_filepath = 'data/custom_manual_rules.pkl'
            try:
                # Ensure data directory exists
                os.makedirs(os.path.dirname(rules_filepath), exist_ok=True)
                
                with open(rules_filepath, 'wb') as f:
                    pickle.dump(defined_rules, f)
                st.success(f"Custom segmentation rules saved successfully to {rules_filepath}!")
            except Exception as e:
                st.error(f"Error saving custom rules: {str(e)}")
                # Continue with displaying results even if saving failed

            # --- Display results of applying to current data ---
            st.subheader("Custom Segmentation Results (Applied to Current Data)")
            st.write(f"Created {len(custom_seg_df['Custom_Segment'].unique())} segments based on saved rules.")
            st.dataframe(custom_seg_df.head(10))
            
            seg_counts = custom_seg_df['Custom_Segment'].value_counts()
            total_customers = len(custom_seg_df)
            
            try: # Plotting can sometimes fail with weird data
                fig, ax = plt.subplots(figsize=(6,4))
                colors = ['#FF4B4B', '#1E88E5', '#4CAF50', '#FFC107', '#9C27B0', '#FF9800', '#607D8B', '#03A9F4', '#E91E63', '#8BC34A']
                labels = [f"{segment} ({count})" for segment, count in seg_counts.items()]
                
                ax.pie(seg_counts, labels=labels, autopct='%1.1f%%', 
                       startangle=90, colors=colors[:len(seg_counts)])
                ax.axis('equal') 
                plt.title(f'Custom Segment Distribution (Total: {total_customers} customers)')
                st.pyplot(fig)
            except Exception as plot_e:
                st.warning(f"Could not display pie chart: {plot_e}")

            # --- Calculate and Show metrics ---
            st.subheader("Segment Summary (Based on Rules Applied)")
            try:
                 segment_metrics = custom_seg_df.groupby('Custom_Segment').agg(
                     Mean_R=('R', 'mean'),
                     Mean_F=('F', 'mean'),
                     Mean_M=('M', 'mean'),
                     Count=('Member_number', 'count') # Assuming 'Member_number' exists
                 ).reset_index() # Use reset_index to make Custom_Segment a column

                 # Calculate Percentage
                 segment_metrics['Percentage'] = (segment_metrics['Count'] / total_customers * 100).round(2)
                 
                 # Prepare for display (e.g., format percentage)
                 segment_metrics_display = segment_metrics.copy()
                 segment_metrics_display['Percentage'] = segment_metrics_display['Percentage'].astype(str) + '%'

                 # Optional: Add a total row if desired (calculated from original rfm_df or custom_seg_df)
                 # ... (logic to add total row similar to before) ...

                 st.dataframe(segment_metrics_display)

                 # Prepare results dictionary (maybe less critical if rules are saved separately)
                 results = {
                     "segmented_df": custom_seg_df, # The df segmented with the custom rules
                     "segment_metrics": segment_metrics, # Metrics from this run
                     "segment_counts": seg_counts,
                     "fig": fig if 'fig' in locals() else None # Pass fig if created
                 }
            except Exception as metric_e:
                 st.warning(f"Could not calculate segment summary: {metric_e}")

    return results # Return results dict (might be None if button not clicked)

def show_segments():
    # Add a header with a nice icon
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #FF4B4B;'>Customer Segmentations</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the segmentation data
    data = load_segmentation_data()
    if data is None:
        st.error("Could not load segmentation data. Please check the data files.")
        return
    
    st.markdown("""
    <div style='margin-top: 20px; padding: 20px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;'>
        <h3>📊 Segmentations Results</h3>
        <p>The demonstration is using the master segments that are predefined using transaction data from a retail store.</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Add tabs for different segmentation methods
    tab1, tab2, tab3 = st.tabs(["RFM Analysis", "K-Means Clusters", "Hierarchical Clusters"])
    
    with tab1:
        st.subheader("RFM Segmentation")
        
        if 'rfm' in data:
            rfm_df = data['rfm']['rfm_df']
            st.dataframe(rfm_df.head())
            
            # Use the function for manual segmentation
            Get_ManualSegmentation(rfm_df)
    
    with tab2:
        st.subheader("K-Means Clustering Results")
        if 'kmeans' in data:
            kmeans_df = data['kmeans']['segmented_df']
            st.dataframe(kmeans_df.head())
            
            # Show cluster distribution
            st.subheader("Cluster Distribution")
            cluster_counts = kmeans_df['KMeans_Segment'].value_counts()
            st.bar_chart(cluster_counts)
    
    with tab3:
        st.subheader("Hierarchical Clustering Results")
        if 'hierarchical' in data:
            hierarchical_df = data['hierarchical']['segmented_df']
            st.dataframe(hierarchical_df.head())
            
            # Show segment distribution
            st.subheader("Segment Distribution")
            segment_counts = hierarchical_df['Hierarchical_Segment'].value_counts()
            st.bar_chart(segment_counts)


def show_mastersegments():
    # Add a header with a nice icon
    master_data = {}
    master_segment = pd.DataFrame()
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #FF4B4B;'>⛃ Master Customer Segments</h1>
    </div>
    """, unsafe_allow_html=True)

    
    # with col1:
        # st.markdown("""
        # <div style='margin-top: 20px;'>
        #     <h3>📊 Segment Overview</h3>
        #     <p>The demonstration is using the master segment that are predefined using transaction data from a retail store.</p>
        # </div>
        # """, unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top: 20px; padding: 20px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;'>
        <h3>📊 Segment Overview</h3>
        <p>The demonstration is using the master segment that are predefined using transaction data from a retail store.</p>

    </div>
    """, unsafe_allow_html=True)
    
    


    
    # Add a button to apply filters
    dataloaded = False
    if st.button("Load Master Segments"):
        master_data = load_segmentation_data()
        master_segment = master_data['rfm']['rfm_df']
        dataloaded = True

        st.info("Data Loaded!")
        
        st.write(master_segment.head(10))

        

    # with col2:
    st.markdown("""
    <div style='margin-top: 20px; padding: 20px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;'>
        <h3>📈 Master Segment Statistics</h3>
    </div>
    """, unsafe_allow_html=True)

    if dataloaded == True:
    
        # Add some example metrics
        Remin = master_segment['Recency'].min()
        Remax = master_segment['Recency'].max()

        Frmin = master_segment['Frequency'].min()
        Frmax = master_segment['Frequency'].max()

        Momin = master_segment['Monetary'].min()
        Momax = master_segment['Monetary'].max()



        st.metric(label="Total Customers", value=master_segment['Member_number'].count())

        st.metric(label="R", value=f'{Remin} - {Remax}')
        st.metric(label="F", value=f'{Frmin} - {Frmax}')
        st.metric(label="M", value=f'{Momin} - {Momax}')
        
        # Add a small chart placeholder
        st.markdown("### Distributions")
            
        # Create three equal columns for the layout
        col1, col2, col3 = st.columns(3)
        
        # Draw distributions of RFM values when data is loaded
        if not master_segment.empty:
            # Set consistent figure size for all plots
            fig_width, fig_height = 8, 6
            
            with col1:
                # Recency Distribution
                fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                ax1.hist(master_segment['Recency'], bins=20, color='#FF4B4B')
                ax1.set_title('Recency Distribution')
                ax1.set_xlabel('Recency')
                ax1.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig1)
                
            with col2:
                # Frequency Distribution
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                ax2.hist(master_segment['Frequency'], bins=20, color='#1E88E5')
                ax2.set_title('Frequency Distribution')
                ax2.set_xlabel('Frequency')
                ax2.set_ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig2)
            
            with col3:
                # Monetary Distribution
                fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
                ax3.hist(master_segment['Monetary'], bins=20, color='#4CAF50')
                ax3.set_title('Monetary Distribution')
                ax3.set_xlabel('Monetary')
                ax3.set_ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig3)

def show_SegmentAssignment():
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #FF4B4B;'>🏷️  Segment Assignment</h1>
    </div>
    """, unsafe_allow_html=True)
    
    data = load_all_data() 

    # --- DEBUG PRINTS START ---
    # print("--- Debugging Data Load ---")
    # if data:
    #     print("Keys in loaded data:", data.keys())
    #     if 'kmeans' in data:
    #         print("Keys in data['kmeans']:", data['kmeans'].keys())
    #     else:
    #         print("Key 'kmeans' not found in loaded data.")
    #     if 'hierarchical' in data:
    #         print("Keys in data['hierarchical']:", data['hierarchical'].keys())
    #     else:
    #         print("Key 'hierarchical' not found in loaded data.")
    # else:
    #     print("Loaded data is None.")
    # print("--- Debugging Data Load END ---")
    # --- DEBUG PRINTS END ---

    if data is None: return 
    if 'rfm' not in data or 'rfm_df' not in data['rfm']:
         st.error("Essential RFM base data could not be loaded. Cannot proceed.")
         return

    # st.sidebar.expander("Manual Segmentation rules (Debug Info)").write(data.get('custom_manual_rules', 'No custom rules loaded')) # Optional Debug

    st.markdown("""
    <div style='margin-top: 20px; padding: 20px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px;'>
        <h3>📌 Assign New Customers to Segments</h3>
        <p>Select a method (Manual Rules, KMeans, or Hierarchical) and provide customer RFM values to assign them to the corresponding segments.</p> 
    </div>
    """, unsafe_allow_html=True)

    # --- Add Method Selection ---
    assignment_method = st.selectbox(
        "Select Assignment Method:",
        options=["Manual Rules", "KMeans", "Hierarchical Clustering"],
        key="assignment_method_select"
    )

    st.markdown("---")
    st.write("**Enter customer data manually or upload a CSV file.**")
    st.write("*CSV file should contain 'Recency', 'Frequency', 'Monetary' columns.*")

    tab1, tab2 = st.tabs(["👤 Manual Input", "📄 CSV Upload"])

    # --- Tab 1: Manual Input ---
    with tab1:
        st.subheader("Enter Single Customer RFM Values")
        col1, col2, col3 = st.columns(3)
        with col1: 
            recency_input = st.number_input("Recency (days)", min_value=0, step=1, key="manual_recency")
        with col2: 
            frequency_input = st.number_input("Frequency (purchases)", min_value=1, step=1, key="manual_frequency")
        with col3: 
            monetary_input = st.number_input("Monetary (value)", min_value=0.0, step=0.01, format="%.2f", key="manual_monetary")

        if st.button(f"Assign Segment using {assignment_method}", key="assign_manual_btn"): 
            input_df = pd.DataFrame({'Recency': [recency_input], 'Frequency': [frequency_input], 'Monetary': [monetary_input]})
            
            # Call the updated assignment function with the selected method
            results_df = assign_segment(input_df, data, assignment_method, showdetails=False) 
            
            if results_df is not None and not results_df.empty:
                 st.success(f"Assignment using {assignment_method} successful!")
                 st.dataframe(results_df) # Display the full result
                 # Optional: Add specific message about which rules were used for manual
                 if assignment_method == "Manual Rules":
                      rules_were_custom = data.get('custom_manual_rules') is not None and isinstance(data['custom_manual_rules'], list) and len(data['custom_manual_rules']) > 0
                      rules_used_msg = "(using saved custom rules)" if rules_were_custom else "(using default notebook rules)"
                      st.caption(f"Manual assignment {rules_used_msg}")
            else:
                 st.warning(f"Could not assign segment using {assignment_method}.")

    # --- Tab 2: CSV Upload ---
    with tab2:
        st.subheader("Upload Customer RFM Data (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")
        
        if uploaded_file is not None:
            try:
                input_df_csv = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(input_df_csv.head())

                required_cols = ['Recency', 'Frequency', 'Monetary']
                if not all(col in input_df_csv.columns for col in required_cols):
                    st.error(f"CSV must contain the columns: {', '.join(required_cols)}")
                else:
                    input_df_to_process = input_df_csv[required_cols].copy()
                    
                    # Call the updated assignment function with the selected method
                    results_df = assign_segment(input_df_to_process, data, assignment_method, showdetails=False) 

                    if results_df is not None:
                        # Merge results back to the original CSV data based on index
                        # Keep original columns and add R, F, M, Assigned_Segment
                        output_df_csv = pd.concat([input_df_csv, results_df[['R', 'F', 'M', 'Assigned_Segment']]], axis=1)
                        
                        st.subheader(f"Results using {assignment_method}")
                        
                        # Optional: Add specific message about which rules were used for manual
                        if assignment_method == "Manual Rules":
                             rules_were_custom = data.get('custom_manual_rules') is not None and isinstance(data['custom_manual_rules'], list) and len(data['custom_manual_rules']) > 0
                             rules_used_msg = "(using saved custom rules)" if rules_were_custom else "(using default notebook rules)"
                             st.info(f"Manual segments assigned {rules_used_msg}")
                             
                        st.dataframe(output_df_csv)

                        csv_output = output_df_csv.to_csv(index=False).encode('utf-8')
                        st.download_button(
                           label=f"Download Results ({assignment_method})",
                           data=csv_output,
                           file_name=f'assigned_{assignment_method.replace(" ", "_").lower()}_segments.csv',
                           mime='text/csv',
                           key='download_csv_btn'
                        )
                    else:
                        st.warning(f"Could not assign segments from CSV using {assignment_method}.")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

def assign_segment(input_data, segmentation_data, method, showdetails = False):
    """
    Assigns a customer segment based on input RFM values using the specified method.

    Parameters:
    input_data (pd.DataFrame): DataFrame with 'Recency', 'Frequency', 'Monetary' columns.
    segmentation_data (dict): Dictionary loaded from load_all_data(). Needs relevant keys for chosen method.
    method (str): The method to use ('Manual Rules', 'KMeans', 'Hierarchical Clustering').
    showdetails (bool): If True, print the results DataFrame within the function.

    Returns:
    pd.DataFrame: Original input DataFrame augmented with 'R', 'F', 'M' score columns 
                  and an 'Assigned_Segment' column, or None on error.
    """
    
    if not isinstance(input_data, pd.DataFrame) or not all(col in input_data.columns for col in ['Recency', 'Frequency', 'Monetary']):
        st.error("Input data must be a pandas DataFrame with 'Recency', 'Frequency', 'Monetary' columns.")
        return None

    try:
        # --- Get Base RFM for Quantiles (needed for R, F, M scores regardless of method) ---
        if 'rfm' not in segmentation_data or 'rfm_df' not in segmentation_data['rfm']:
             st.error("Base RFM DataFrame ('rfm_df') not found in loaded data. Needed for scores.")
             return None
        rfm_df_master = segmentation_data['rfm']['rfm_df']
        
        # --- Calculate Quantiles ---
        quantiles = {
            'Recency': rfm_df_master['Recency'].quantile([0.25, 0.5, 0.75]).tolist(), 
            'Frequency': rfm_df_master['Frequency'].quantile([0.25, 0.5, 0.75]).tolist(),
            'Monetary': rfm_df_master['Monetary'].quantile([0.25, 0.5, 0.75]).tolist()
        }
        
        # --- Calculate R, F, M scores for input data ---
        rfm_scores_df = calculate_rfm_scores(input_data, quantiles) 

        # Initialize
        assigned_segments_series = None 
        
        # --- Apply Selected Method ---
        if method == "Manual Rules":
            custom_rules = segmentation_data.get('custom_manual_rules', None)
            if custom_rules and isinstance(custom_rules, list) and len(custom_rules) > 0:
                def apply_saved_rules(row_scores):
                    for rule in custom_rules:
                        if (row_scores['R'] >= rule['r_min'] and 
                            row_scores['F'] >= rule['f_min'] and 
                            row_scores['M'] >= rule['m_min']):
                            return rule['name'] 
                    return 'Other Customers' 
                assigned_segments_series = rfm_scores_df.apply(apply_saved_rules, axis=1)
            else:
                def get_notebook_manual_segment_from_scores(row):
                    if row['R'] == 1: return 'Inactive Customers' 
                    elif row['F'] >= 4 and row['M'] >= 4: return 'VIP Customers'
                    elif row['F'] >= 2 and row['M'] >= 4: return 'Big Spenders'
                    elif row['F'] >= 2 and row['M'] >= 2: return 'Regular Customers'
                    else: return 'Occasional Customers' 
                assigned_segments_series = rfm_scores_df.apply(get_notebook_manual_segment_from_scores, axis=1)

        elif method == "KMeans":
            if 'kmeans' not in segmentation_data or not all(k in segmentation_data['kmeans'] for k in ['model', 'scaler', 'mapping']):
                 st.error("KMeans model, scaler, or mapping not found in loaded data.")
                 return None
             
            kmeans_model = segmentation_data['kmeans']['model']
            scaler = segmentation_data['kmeans']['scaler']
            cluster_mapping = segmentation_data['kmeans']['mapping'] 

            features_scaled = scaler.transform(input_data[['Recency', 'Frequency', 'Monetary']])
            predicted_clusters = kmeans_model.predict(features_scaled)
            assigned_segments_series = pd.Series(predicted_clusters).map(cluster_mapping)

        elif method == "Hierarchical Clustering":
            # Import required library here if not already at the top
            from scipy.spatial.distance import cdist 
            import numpy as np

            if 'hierarchical' not in segmentation_data or not all(k in segmentation_data['hierarchical'] for k in ['scaler', 'centers_scaled', 'mapping']):
                 st.error("Hierarchical scaler, scaled centers, or mapping not found.")
                 return None

            scaler = segmentation_data['hierarchical']['scaler']
            cluster_centers_scaled = segmentation_data['hierarchical']['centers_scaled'] 
            cluster_mapping = segmentation_data['hierarchical']['mapping'] 

            # --- Debugging Hierarchical Start ---
            # st.write("--- Debug Hierarchical Assignment ---")
            # st.write("Input Data:")
            # st.dataframe(input_data)
            # st.write(f"Scaler Object Type: {type(scaler)}")
            # # st.write(f"Cluster Centers Scaled:\n{cluster_centers_scaled}") # Can be large
            # st.write(f"Cluster Mapping: {cluster_mapping}")
            # --- Debugging Hierarchical End ---

            features_scaled = scaler.transform(input_data[['Recency', 'Frequency', 'Monetary']])

            # --- Debugging Hierarchical Start 2 ---
            # st.write("Scaled Input Features:")
            # st.dataframe(pd.DataFrame(features_scaled, columns=['R_scaled', 'F_scaled', 'M_scaled']))
            # --- Debugging Hierarchical End 2 ---

            distances = cdist(features_scaled, cluster_centers_scaled)
            # Add 1 if your cluster_mapping is 1-indexed
            predicted_cluster_index = np.argmin(distances, axis=1) 
            # Determine base index (0 or 1) from mapping keys
            mapping_base_index = min(cluster_mapping.keys()) 
            # Adjust predicted index if mapping is 1-based
            predicted_clusters = predicted_cluster_index + mapping_base_index 

            # --- Debugging Hierarchical Start 3 ---
            # st.write("Distances to Cluster Centers:")
            # st.dataframe(pd.DataFrame(distances))
            # st.write(f"Index of Minimum Distance (0-based): {predicted_cluster_index[0]}") # Assuming single input for debug clarity
            # st.write(f"Mapping Base Index (from keys): {mapping_base_index}")
            # st.write(f"Final Cluster Index (adjusted): {predicted_clusters[0]}") # Assuming single input
            # --- Debugging Hierarchical End 3 ---

            assigned_segments_series = pd.Series(predicted_clusters).map(cluster_mapping)
            
        else:
             st.error(f"Unsupported assignment method: {method}")
             return None

        # --- Combine Results ---
        results_df = input_data.copy()
        results_df[['R', 'F', 'M']] = rfm_scores_df[['R', 'F', 'M']] # Add scores
        # Handle case where assignment might have failed for some reason
        if assigned_segments_series is not None:
             results_df['Assigned_Segment'] = assigned_segments_series
        else:
             results_df['Assigned_Segment'] = "Error in Assignment" # Or None, or raise error earlier

        # --- Conditionally print the DataFrame ---
        if showdetails:
            st.dataframe(results_df) 

        return results_df

    except KeyError as e:
        st.error(f"Missing expected key: {e}. Check data loading and structures.")
        return None
    except Exception as e:
        st.error(f"An error occurred during segment assignment using {method}: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_rfm_scores(df_input, quantiles):
    """
    Calculates R, F, M scores (1-4) based on input values and pre-defined quantiles.

    Parameters:
    df_input (pd.DataFrame): DataFrame with 'Recency', 'Frequency', 'Monetary' columns.
    quantiles (dict): Dictionary containing quantile boundaries for R, F, M. 
                      Example: {'Recency': [q1, q2, q3], 'Frequency': [q1, q2, q3], 'Monetary': [q1, q2, q3]}

    Returns:
    pd.DataFrame: Input DataFrame with added 'R', 'F', 'M' score columns.
    """
    df_scores = df_input.copy()
    
    # Calculate R score (Lower Recency score means more recent - Adjust if inverted)
    # Assigns 4 for most recent, 1 for least recent based on quantiles
    df_scores['R'] = df_scores['Recency'].apply(lambda x: 4 if x <= quantiles['Recency'][0] else \
                                                          3 if x <= quantiles['Recency'][1] else \
                                                          2 if x <= quantiles['Recency'][2] else 1) 
                                                          
    # Calculate F score (Higher Frequency is better)
    df_scores['F'] = df_scores['Frequency'].apply(lambda x: 4 if x > quantiles['Frequency'][2] else \
                                                            3 if x > quantiles['Frequency'][1] else \
                                                            2 if x > quantiles['Frequency'][0] else 1)
    # Calculate M score (Higher Monetary is better)
    df_scores['M'] = df_scores['Monetary'].apply(lambda x: 4 if x > quantiles['Monetary'][2] else \
                                                           3 if x > quantiles['Monetary'][1] else \
                                                           2 if x > quantiles['Monetary'][0] else 1)
    return df_scores[['R', 'F', 'M']]

def show_homepage():
    # Add a header with a nice icon
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #FF4B4B;'>Customer Segmentation</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-top: 20px;'>
            <h3>✨ Project Overview</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🤖</span>
                <span>This product is the demonstration of customer segmentation using RFM analysis and clustering.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features with icons
        st.markdown("""
        <div style='margin-top: 20px;'>
            <h3>🎯 Key Features</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🔄</span>
                <span>Segment customers based on their RFM scores</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🔍</span>
                <span>Explore customer segments using Kmeans clustering</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>👥</span>
                <span>Explore customer segments using Hierarchical clustering</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>📊</span>
                <span>Assign customers (new data) to segments based on their RFM scores</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Description with better formatting
    st.markdown("""
        <div style='margin-top: 30px;'>
            <h3 style='color:rgb(255, 255, 255);'>⚙️ Technical Description</h3>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>📝</span>
                <span><strong>Manual Segmentation (Rules based)</strong> - Analyzes product descriptions</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🧩</span>
                <span><strong>Kmeans Clustering</strong> - Learns from user ratings</span>
            </div>
            <div style='display: flex; align-items: center; margin: 10px 0;'>
                <span style='font-size: 24px; margin-right: 10px;'>🌳</span>
                <span><strong>Hierarchical Clustering</strong></span>
            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Add a decorative image or illustration
        image_path = 'data/cs.png' # Use forward slashes
        try:
            st.image(image_path, width=1000)
        except FileNotFoundError:
            st.error(f"Image not found at {image_path}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        # Group Members section
        st.markdown("""
        <div style='margin-top: 20px; padding: 20px; border-radius: 1px;'>
            <h3 style='color:rgb(255, 255, 255);'>👨‍💻👩‍💻 Group Members</h3>
            <p style='font-size: 28px;'><br>Châu Nhật Minh - Phạm Đình Anh Duy</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    try:
        # Add navigation in sidebar
        st.markdown(
        """
        <style>
            [data-testid="stSidebar"][aria-expanded="true"]{
                min-width: 300px;
                max-width: 300px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
        st.sidebar.title("Navigation")
        
        # Create buttons for navigation
        if st.sidebar.button("🏠 Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("⛃ Master Segments"):
            st.session_state.page = "Master Segments"

        if st.sidebar.button("👨‍👩‍👧‍👦 Segmentations"):
            st.session_state.page = "Segmentations"
        if st.sidebar.button("🏷️ Segment Assignment"):
            st.session_state.page = "Segment Assignment"
        
        # Initialize session state if not exists
        if 'page' not in st.session_state:
            st.session_state.page = "Home"
        
        # Show the selected page
        if st.session_state.page == "Home":
            show_homepage()
        elif st.session_state.page == "Master Segments":
            show_mastersegments()
        elif st.session_state.page == "Segmentations":
            # Load necessary data for show_segments (just base RFM needed for Get_ManualSegmentation)
            base_data = load_segmentation_data() # Original loader might be sufficient here
            if base_data and 'rfm' in base_data:
                 show_segments() # Pass the whole dict or just rfm_df if refactored
            else:
                 st.error("Failed to load data for Segmentations page.")
             
        elif st.session_state.page == "Segment Assignment":
            show_SegmentAssignment() # This function now handles loading and using rules

        
            # Add empty space to push footer to bottom
        st.sidebar.markdown("<br>" * 13, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

#    # Cosine in Streamlit
#    @st.cache_data
#    def load_cosine_models():
#        with open('models/vectorizer.pkl', 'rb') as f:
#            vectorizer = pickle.load(f)
#        with open('models/tfidf_matrix.pkl', 'rb') as f:
#            tfidf_matrix = pickle.load(f)
#        return vectorizer, tfidf_matrix



#        return vectorizer, tfidf_matrix








