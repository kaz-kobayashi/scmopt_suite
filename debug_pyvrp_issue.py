# Debug the PyVRP implementation issue
import sys
sys.path.append('.')

from pyvrp_enhanced import *
import numpy as np

# Define basic classes
class Job:
    def __init__(self, id, location=None, location_index=None, service=0, 
                 delivery=[0], pickup=[0], time_windows=None, 
                 priority=1, description=""):
        self.id = id
        self.location = location
        self.location_index = location_index
        self.service = service
        self.delivery = delivery
        self.pickup = pickup
        self.time_windows = time_windows if time_windows else [[0, 86400]]
        self.priority = priority
        self.description = description

class VehicleCosts:
    def __init__(self, fixed=0, per_hour=3600, per_km=0):
        self.fixed = fixed
        self.per_hour = per_hour
        self.per_km = per_km

class Vehicle:
    def __init__(self, id, start=None, end=None, start_index=None, end_index=None,
                 capacity=[1000], time_window=None, description="", costs=None):
        self.id = id
        self.start = start
        self.end = end
        self.start_index = start_index
        self.end_index = end_index
        self.capacity = capacity
        self.time_window = time_window if time_window else [0, 86400]
        self.description = description
        self.costs = costs if costs else VehicleCosts()
        self.max_travel_time = None

class Matrix:
    def __init__(self, durations=None, distances=None):
        self.durations = durations
        self.distances = distances

class Model:
    def __init__(self, jobs=None, shipments=None, vehicles=None, matrices=None):
        self.jobs = jobs if jobs else []
        self.shipments = shipments if shipments else []
        self.vehicles = vehicles if vehicles else []
        self.matrices = matrices if matrices else {}

def debug_conversion():
    """Debug the conversion process step by step"""
    print("=== Debugging PyVRP Conversion ===")
    
    # Create simple test data
    jobs = [
        Job(id=0, location=[0.0001, 0.0000], location_index=1, service=600,
            delivery=[10], pickup=[0], description="Client_0"),
        Job(id=1, location=[0.0000, 0.0001], location_index=2, service=600,
            delivery=[20], pickup=[0], description="Client_1"), 
        Job(id=2, location=[-0.0001, 0.0000], location_index=3, service=600,
            delivery=[15], pickup=[0], description="Client_2"),
        Job(id=3, location=[0.0000, -0.0001], location_index=4, service=600,
            delivery=[25], pickup=[0], description="Client_3"),
    ]
    
    vehicles = [
        Vehicle(id=0, start=[0.0, 0.0], end=[0.0, 0.0], 
               start_index=0, end_index=0, capacity=[100], 
               description="Vehicle_0"),
    ]
    
    # Create distance matrix (5x5: 4 clients + 1 depot)
    durations = [
        [   0,  141,  100,  141,  100],  # from depot (index 0)
        [ 141,    0,  141,  200,  141],  # from client 0 (index 1)
        [ 100,  141,    0,  141,  200],  # from client 1 (index 2)
        [ 141,  200,  141,    0,  141],  # from client 2 (index 3)
        [ 100,  141,  200,  141,    0],  # from client 3 (index 4)
    ]
    
    distances = [[d + 300 for d in row] for row in durations]
    
    matrices = {
        'car': Matrix(durations=durations, distances=distances)
    }
    
    model = Model(jobs=jobs, vehicles=vehicles, matrices=matrices)
    
    print(f"Model created:")
    print(f"  Jobs: {len(model.jobs)}")
    print(f"  Vehicles: {len(model.vehicles)}")
    print(f"  Matrix size: {len(model.matrices['car'].durations)}")
    
    # Debug conversion step by step
    print("\n=== Starting Conversion ===")
    builder = convert_metro_to_enhanced_pyvrp(model)
    
    print(f"After conversion:")
    print(f"  Location mapping: {builder.location_mapping}")
    print(f"  Client mapping: {builder.client_mapping}")
    print(f"  Depot mapping: {builder.depot_mapping}")
    print(f"  Vehicle mapping: {builder.vehicle_mapping}")
    
    # Check PyVRP model data
    data = builder.model.data()
    print(f"\nPyVRP model data:")
    print(f"  Number of depots: {len(list(data.depots()))}")
    print(f"  Number of clients: {len(list(data.clients()))}")
    print(f"  Number of vehicle types: {len(list(data.vehicle_types()))}")
    
    # List depots
    for i, depot in enumerate(data.depots()):
        print(f"  Depot {i}: x={depot.x}, y={depot.y}")
    
    # List clients
    for i, client in enumerate(data.clients()):
        print(f"  Client {i}: x={client.x}, y={client.y}, delivery={client.delivery}, pickup={client.pickup}")
    
    # List vehicle types
    for i, vt in enumerate(data.vehicle_types()):
        print(f"  Vehicle type {i}: capacity={vt.capacity}, num_available={vt.num_available}")
        print(f"    start_depot={vt.start_depot}, end_depot={vt.end_depot}")
    
    # Check distance and duration matrices
    print(f"\nChecking distance/duration matrices:")
    print(f"Number of distance matrices: {data.num_profiles}")
    
    if data.num_profiles > 0:
        distance_matrix = data.distance_matrix(0)  # Profile 0
        duration_matrix = data.duration_matrix(0)
        
        print(f"Distance matrix shape: {distance_matrix.shape}")
        print(f"Duration matrix shape: {duration_matrix.shape}")
        
        print(f"Distance matrix (first 5x5):")
        for i in range(min(5, distance_matrix.shape[0])):
            row = []
            for j in range(min(5, distance_matrix.shape[1])):
                row.append(str(distance_matrix[i, j]))
            print(f"  [{', '.join(row)}]")
            
        print(f"Duration matrix (first 5x5):")
        for i in range(min(5, duration_matrix.shape[0])):
            row = []
            for j in range(min(5, duration_matrix.shape[1])):
                row.append(str(duration_matrix[i, j]))
            print(f"  [{', '.join(row)}]")
    else:
        print("No distance/duration matrices found!")
    
    # Manual edge addition test
    print(f"\nManual edge test:")
    try:
        locations = list(data.depots()) + list(data.clients())
        test_edge = builder.model.add_edge(locations[0], locations[1], distance=999, duration=888)
        print(f"Successfully added test edge: {test_edge}")
        
        # Check matrix again
        data2 = builder.model.data()
        if data2.num_profiles > 0:
            new_distance_matrix = data2.distance_matrix(0)
            print(f"After manual add - matrix[0][1] = {new_distance_matrix[0, 1]}")
        
    except Exception as e:
        print(f"Error adding manual edge: {e}")
        import traceback
        traceback.print_exc()
    
    return builder

if __name__ == "__main__":
    builder = debug_conversion()