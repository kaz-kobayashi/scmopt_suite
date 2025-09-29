import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, time
from typing import Tuple, Optional, Dict, Any
import requests

def co2(capacity: float, rate: float = 0.5, diesel: bool = False) -> Tuple[float, float]:
    """
    Calculate fuel consumption per ton-km and CO2 emissions
    
    Args:
        capacity: Vehicle capacity in tons
        rate: Loading rate (0-1)
        diesel: True for diesel, False for gasoline
    
    Returns:
        Tuple of (fuel_consumption_L_per_ton_km, co2_emissions_g_per_ton_km)
    """
    if diesel:
        fuel = math.exp(2.67 - 0.927 * math.log(rate) - 0.648 * math.log(capacity))
    else:
        fuel = math.exp(2.71 - 0.812 * math.log(rate) - 0.654 * math.log(capacity))
    
    co2_emissions = fuel * 2.322 * 1000  # Convert to g/ton-km
    
    return fuel, co2_emissions

def time_delta(finish, start) -> float:
    """
    Calculate time difference in seconds
    
    Args:
        finish: End time (datetime or time)
        start: Start time (datetime or time)
    
    Returns:
        Time difference in seconds
    """
    if isinstance(finish, time) and isinstance(start, time):
        # Convert time to datetime for calculation
        base_date = datetime.today().date()
        finish_dt = datetime.combine(base_date, finish)
        start_dt = datetime.combine(base_date, start)
        
        # Handle overnight times
        if finish_dt < start_dt:
            finish_dt += timedelta(days=1)
            
        return (finish_dt - start_dt).total_seconds()
    
    elif isinstance(finish, datetime) and isinstance(start, datetime):
        return (finish - start).total_seconds()
    
    else:
        raise ValueError("Both arguments must be either datetime or time objects")

def add_seconds(start, seconds: float) -> str:
    """
    Add specified seconds to a datetime/time object
    
    Args:
        start: Start time (datetime or time)
        seconds: Seconds to add
    
    Returns:
        Formatted time string
    """
    if isinstance(start, time):
        base_date = datetime.today().date()
        start_dt = datetime.combine(base_date, start)
        result_dt = start_dt + timedelta(seconds=seconds)
        return result_dt.time().strftime("%H:%M:%S")
    
    elif isinstance(start, datetime):
        result_dt = start + timedelta(seconds=seconds)
        return result_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    else:
        raise ValueError("Start must be either datetime or time object")

def compute_durations_simple(cust_df: pd.DataFrame, plnt_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute travel durations and distances between locations (simplified version without OSRM)
    
    Args:
        cust_df: Customer locations dataframe with columns ['name', 'lat', 'lon']
        plnt_df: Plant locations dataframe (optional)
    
    Returns:
        Tuple of (durations, distances, node_df)
    """
    # Combine customer and plant locations
    node_df = cust_df.copy()
    if plnt_df is not None:
        node_df = pd.concat([node_df, plnt_df], ignore_index=True)
    
    n_nodes = len(node_df)
    
    # Calculate distances using Haversine formula
    distances = np.zeros((n_nodes, n_nodes))
    durations = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                lat1, lon1 = node_df.iloc[i]['lat'], node_df.iloc[i]['lon']
                lat2, lon2 = node_df.iloc[j]['lat'], node_df.iloc[j]['lon']
                
                # Haversine formula
                R = 6371  # Earth radius in km
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (math.sin(dlat/2)**2 + 
                     math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                     math.sin(dlon/2)**2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = R * c
                
                distances[i][j] = distance
                # Estimate duration with variable speed based on distance
                # Short distance (< 20km): 30 km/h (urban traffic)
                # Medium distance (20-50km): 40 km/h (suburban)
                # Long distance (> 50km): 50 km/h (highway)
                if distance < 20:
                    avg_speed = 30
                elif distance < 50:
                    avg_speed = 40
                else:
                    avg_speed = 50
                durations[i][j] = distance / avg_speed * 3600  # Convert to seconds
    
    return durations, distances, node_df

def make_time_df(node_df: pd.DataFrame, durations: np.ndarray, distances: np.ndarray) -> pd.DataFrame:
    """
    Create dataframe with origin-destination travel times and distances
    
    Args:
        node_df: Node locations dataframe
        durations: Duration matrix in seconds
        distances: Distance matrix in km
    
    Returns:
        Time dataframe with columns ['orig', 'dest', 'time', 'dist']
    """
    time_data = []
    
    for i in range(len(node_df)):
        for j in range(len(node_df)):
            if i != j:
                time_data.append({
                    'orig': node_df.iloc[i]['name'],
                    'dest': node_df.iloc[j]['name'],
                    'time': durations[i][j],
                    'dist': distances[i][j]
                })
    
    return pd.DataFrame(time_data)

def compute_durations_with_osrm(cust_df: pd.DataFrame, 
                               plnt_df: Optional[pd.DataFrame] = None,
                               toll: bool = True,
                               host: str = "localhost") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute travel durations and distances using OSRM service
    
    Args:
        cust_df: Customer locations dataframe
        plnt_df: Plant locations dataframe (optional)
        toll: Include toll roads
        host: OSRM server host
    
    Returns:
        Tuple of (durations, distances, node_df)
    """
    # This would require a running OSRM instance
    # For now, fall back to simple calculation
    return compute_durations_simple(cust_df, plnt_df)