from typing import List, Optional
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scripts.utils import sphere_to_cartesian, angular_dist_score, adjust_sphere, cartesian_to_sphere


from sklearn.decomposition import PCA

def get_direction(coords: np.ndarray) -> np.ndarray:
    """
    Get the direction vector from a list of coordinates.
    """
    pca = PCA(n_components=1)
    pca.fit(coords) 
    direction_vector = pca.components_#type: ignore
    return direction_vector

def get_line(origin_coords, vector, extent:int) -> np.ndarray:
    """Get a line from a point in a direction.

    Args:
        origin_coords (List or Array): The coordinates of the origin of the line.
        vector (List or Array): The direction of the line from the origin.
        extent (int): The extent of the line in both directions.

    Returns:
        array: The coordinates of the line.
    """
    origin_coords = np.array(origin_coords)  # convert to NumPy array
    vector = np.array(vector)  # convert to NumPy array
    below_origin = origin_coords - vector * extent
    above_origin = origin_coords + vector * extent
    line = np.vstack((below_origin, above_origin))
    return line


def get_event_true_values(meta_df: pd.DataFrame, event_id:int) -> dict:
    """Gets the azimuth and zenith values for the input event

    Args:
        meta_df (pd.DataFrame): The dataframe with the metadata about the batches
        event_id (int): The event ID to show

    Returns:
        dict: The two labelled target angles
    """
    assert 'azimuth' in meta_df and 'zenith' in meta_df, f"Missing columns in Meta dataframe"
    
    return meta_df[meta_df['event_id']== event_id][['azimuth','zenith']].iloc[0].to_dict()


def compose_event_df(
    batch_df: pd.DataFrame,
    event_id: str,
    sensor_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Composes the event dataframe from the batch dataframe and the sensor geometry dataframe.

    Args:
        batch_df (pd.DataFrame): _description_
        event_id (str): The event id to extract from the data frame
        sensor_geometry (pd.DataFrame): The dataframe containing the sensor geometry.

    Returns:
        pd.DataFrame: a dataframe containing the data for one event.
    """
    
    # filter the batch dataframe for the event_id
    event_df = batch_df[batch_df['event_id'] == event_id]
    # merge the event dataframe with the sensor geometry dataframe
    event_df = pd.merge(
        left = event_df,
        right = sensor_geometry,
        how='inner',
        on='sensor_id'
    )
    # sort the dataframe by sensor_id
    event_df.sort_values(by=['time'], inplace=True)
    return event_df


def plot_pca(
    df:pd.DataFrame, 
    labels: dict, 
    exclude_axillary: bool = False,
    time_threshold: Optional[List[int]] = None,
    charge_threshold: Optional[List[float,]] = None,
    marker_size: int = 9
    
    ):
    """Plots the line of best fit for the event using PCA.

    Args:
        event_df (pd.DataFrame): The dataframe containing the event data including the sensor geometry.
        labels (dict): The truth labels for the event.
        exclude_axillary (bool, optional): Weather to exclude the axillary readings. Defaults to False.
    """
    
    
    assert set(df.columns) == set(['event_id', 'sensor_id', 'time', 'charge', 'auxiliary', 'x', 'y', 'z']), f"Unexpected columns in event_df: {df.columns}"
    assert 'azimuth' in labels and 'zenith' in labels, f"Missing labels in labels: {labels}"
    
    if exclude_axillary:
        df = df[~df['auxiliary']]
    
    if time_threshold is not None:
        df = df[df['time'] > time_threshold[0]] if time_threshold[0] is not None else df
        df = df[df['time'] < time_threshold[1]] if time_threshold[1] is not None else df
        
    if charge_threshold is not None:
        df = df[df['charge'] > charge_threshold[0]] if charge_threshold[0] is not None else df
        df = df[df['charge'] < charge_threshold[1]] if charge_threshold[1] is not None else df
        
    x, y, z = df['x'], df['y'], df['z']
    coords = np.array((x,y,z)).T
    direction_vector = get_direction(coords)
    origin = np.mean(coords, axis=0)
    euclidean_distance = np.linalg.norm(coords - origin, axis=1)
    extent = np.max(euclidean_distance)
    line = get_line(origin, direction_vector, extent)
    
    azimuth = labels['azimuth']
    zenith= labels['zenith']
    x,y,z = sphere_to_cartesian(azimuth, zenith) #type: ignore

    truth_trace = get_line([0,0,0], [[x,y,z]], extent)
    
    az_pred, zen_pred = adjust_sphere(*cartesian_to_sphere(x,y,z))
    score = angular_dist_score(az_true=azimuth,zen_true=zenith, az_pred=az_pred, zen_pred=zen_pred)
    
    fig3 = go.Figure(data = [
            go.Scatter3d(
                x=df['x'].to_numpy(), y=df['y'].to_numpy(), z=df['z'].to_numpy(),
                mode='markers',
                name="Detected",
                marker=dict(
                    size=df['charge'].to_numpy() * marker_size,
                    color=df['time'].to_numpy(),
                    opacity=0.8,
                    colorbar=dict(
                        title="Time",
                    ),
                ),
                hovertemplate="""
                    <b>X:</b> %{x}
                    <br>
                    <b>Y:</b> %{y}
                    <br>
                    <b>Z:</b> %{z}
                    <br>
                    <b>Charge (x marker_size):</b> %{marker.size:.2f}<br>
                    <b>Time:</b> %{marker.color:.2f}
                """
            ),
            go.Scatter3d(
                x=line[:,0], y=line[:,1], z=line[:,2],
                marker=dict(
                    size=4,
                    color='red',
                ),
                line=dict(
                    color='red',
                    width=3
                ),
                name="Predicted"
            ),
            go.Scatter3d(
                x=truth_trace[:,0], y=truth_trace[:,1], z=truth_trace[:,2],
                marker=dict(
                    size=4,
                    color='green',
                ),
                line=dict(
                    color='green',
                    width=3
                ),
                name="Truth"
            )
    ])

    # Add a legend
    fig3.update_layout(
        # text=f"Event prediction",
        title={
            "text": f"Event prediction",
            "y": 0.9,  # adjust the y position of the title to make room for the subtitle
            "x": 0.5,  # center the title horizontally
            "xanchor": "center",  # center the title horizontally
            "yanchor": "top",  # align the title with the top of the plot
        },
        annotations=[
            go.layout.Annotation(
                x=0.01,  # center the subtitle horizontally
                y=0.01,  # position the subtitle below the title
                xref="paper",
                yref="paper",
                text=f"""
                    <b>exclude_axillary: </b> {exclude_axillary}
                    <br>
                    <b>time_threshold: </b> {time_threshold}
                    <br>
                    <b>charge_threshold: </b> {charge_threshold}
                    <br>
                    <b>Score: </b> {score}
                    <br>
                """,

                showarrow=False,
                align= 'left',
                font=dict(size=16),  # set the font size and color of the subtitle
            )
        ],
        showlegend=True, 
        legend=dict(x=0, y=1),
        height=800,
    )

    fig3.show()