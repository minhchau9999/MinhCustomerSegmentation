import matplotlib.pyplot as plt
import squarify
import numpy as np

def plot_3d_segments(df, x='Recency', y='Frequency', z='Monetary', 
                     color_column='Segment', size_column='Frequency',
                     title='Customer Segments in 3D RFM Space',
                     color_discrete_map=None, color_discrete_sequence=None,
                     additional_hover_data=None):
    """
    Create an interactive 3D scatter plot for customer segmentation visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the customer data with segmentation
    x, y, z : str
        Column names for the x, y, and z axes (default: RFM dimensions)
    color_column : str
        Column name for the segment/cluster to color by
    size_column : str
        Column name to determine point size
    title : str
        Plot title
    color_discrete_map : dict, optional
        Dictionary mapping values in color_column to specific colors
    color_discrete_sequence : list, optional
        List of colors to use for the different values in color_column
    additional_hover_data : list or dict, optional
        Additional columns to display in hover data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive 3D scatter plot
    """
    import plotly.express as px
    
    # Set up hover data
    hover_data = {
        x: True,
        y: True,
        z: ':.2f',
        color_column: True
    }
    
    # Add additional hover data if provided
    if additional_hover_data:
        if isinstance(additional_hover_data, list):
            for col in additional_hover_data:
                hover_data[col] = True
        elif isinstance(additional_hover_data, dict):
            hover_data.update(additional_hover_data)
    
    # Create the interactive 3D scatter plot
    fig = px.scatter_3d(
        df, 
        x=x,
        y=y,
        z=z,
        color=color_column,
        opacity=0.7,
        size=size_column,
        size_max=15,
        hover_name=df.index.name if df.index.name else None,
        hover_data=hover_data,
        title=title,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=color_discrete_sequence
    )

    # Improve layout
    fig.update_layout(
        scene=dict(
            xaxis_title=f'{x} (days)' if x == 'Recency' else x,
            yaxis_title=f'{y} (count)' if y == 'Frequency' else y,
            zaxis_title=f'{z} (total spend)' if z == 'Monetary' else z,
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Example usage:
# fig = plot_3d_segments(rfm_df, color_column='GMM_Segment')
# fig.show()



def plot_segment_treemap(df, segment_column, title='Customer Segments Distribution (Treemap)', 
                         figsize=(12, 8), colormap='viridis', fontsize=12):
    """
    Create a treemap visualization of customer segments
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing segment information
    segment_column : str
        Column name containing segment labels
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height) in inches
    colormap : str
        Matplotlib colormap name to use
    fontsize : int
        Font size for segment labels
        
    Returns:
    --------
    fig : matplotlib Figure
        The treemap figure
    """
    # Calculate the size and percentage of each segment
    segment_counts = df[segment_column].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    segment_counts['Percentage'] = segment_counts['Count'] / segment_counts['Count'].sum() * 100
    segment_counts = segment_counts.sort_values('Count', ascending=False)

    # Prepare data for squarify
    sizes = segment_counts['Count']
    labels = [f"{segment} ({count}, {percentage:.1f}%)" 
              for segment, count, percentage in zip(segment_counts['Segment'], 
                                                   segment_counts['Count'], 
                                                   segment_counts['Percentage'])]
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 0.9, len(sizes)))

    # Create the plot
    fig = plt.figure(figsize=figsize)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, 
                  text_kwargs={'fontsize': fontsize})
    plt.axis('off')
    plt.title(title, fontsize=fontsize+3)
    plt.tight_layout()
    
    return fig