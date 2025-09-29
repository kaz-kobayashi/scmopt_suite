# Test PyVRP with a simple example to verify our implementation
# Create a simple VRP instance and solve it with both PyVRP directly and our implementation

import sys
import numpy as np
sys.path.append('.')

from pyvrp import Model as PyVRPModel
from pyvrp.stop import MaxRuntime
from pyvrp_enhanced import *

# Define basic classes for our implementation
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

def create_simple_pyvrp_example():
    """Create a simple VRP example that matches PyVRP documentation style"""
    
    # Create PyVRP model directly
    model = PyVRPModel()
    
    # Add depot at origin (0, 0)
    depot = model.add_depot(x=0, y=0)
    print(f"Added depot: {depot}")
    
    # Add 4 clients with different demands
    clients = []
    client_coords = [(100, 0), (0, 100), (-100, 0), (0, -100)]  # Simple square layout
    client_demands = [10, 20, 15, 25]  # Total demand = 70
    
    for i, ((x, y), demand) in enumerate(zip(client_coords, client_demands)):
        client = model.add_client(
            x=x, 
            y=y, 
            delivery=[demand], 
            pickup=[0],
            service_duration=600,  # 10 minutes service
            name=f"Client_{i}"
        )
        clients.append(client)
        print(f"Added client {i}: coordinates=({x}, {y}), demand={demand}")
    
    # Add vehicle type with capacity 100
    vehicle_type = model.add_vehicle_type(
        num_available=1,  # 1 vehicle available
        capacity=[100],   # capacity 100 units
        start_depot=depot,
        end_depot=depot,
        name="Vehicle"
    )
    print(f"Added vehicle type: capacity=100, num_available=1")
    
    # Add edges between all location pairs
    locations = [depot] + clients
    for i, from_loc in enumerate(locations):
        for j, to_loc in enumerate(locations):
            if i != j:
                # Calculate Euclidean distance between locations
                from_coords = client_coords[i-1] if i > 0 else (0, 0)
                to_coords = client_coords[j-1] if j > 0 else (0, 0)
                
                distance = int(np.sqrt((from_coords[0] - to_coords[0])**2 + 
                                     (from_coords[1] - to_coords[1])**2))
                duration = distance + 300  # Add some service time
                
                model.add_edge(from_loc, to_loc, distance=distance, duration=duration)
    
    return model

def solve_with_pyvrp_directly():
    """Solve the problem directly with PyVRP"""
    print("=== Solving with PyVRP directly ===")
    
    model = create_simple_pyvrp_example()
    
    # Solve
    result = model.solve(stop=MaxRuntime(10), seed=1234)
    
    print(f"Direct PyVRP solution:")
    print(f"  Cost: {result.cost()}")
    print(f"  Number of routes: {len(result.best.routes())}")
    
    # Show routes
    routes = result.best.routes()
    for i, route in enumerate(routes):
        print(f"  Route {i}: {list(route)}")
    
    return result

def solve_with_our_implementation():
    """Solve with our enhanced implementation"""
    print("\n=== Solving with our Enhanced Implementation ===")
    
    # Create model using our classes
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
    
    # Create Euclidean distance matrix to match direct PyVRP
    import numpy as np
    coordinates = [
        [0.0, 0.0],      # depot
        [0.0001, 0.0],   # client 0 -> (100, 0) after scaling  
        [0.0, 0.0001],   # client 1 -> (0, 100) after scaling
        [-0.0001, 0.0],  # client 2 -> (-100, 0) after scaling
        [0.0, -0.0001],  # client 3 -> (0, -100) after scaling
    ]
    
    # Calculate Euclidean distances and durations
    n = len(coordinates)
    durations = []
    distances = []
    
    for i in range(n):
        duration_row = []
        distance_row = []
        for j in range(n):
            if i == j:
                duration_row.append(0)
                distance_row.append(0)
            else:
                # Calculate Euclidean distance in scaled coordinates (multiply by 1M)
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                scaled_x1, scaled_y1 = x1 * 1000000, y1 * 1000000
                scaled_x2, scaled_y2 = x2 * 1000000, y2 * 1000000
                
                euclidean_dist = int(np.sqrt((scaled_x1 - scaled_x2)**2 + (scaled_y1 - scaled_y2)**2))
                duration = euclidean_dist  # Use distance as duration
                
                duration_row.append(duration)
                distance_row.append(euclidean_dist)
        
        durations.append(duration_row)  
        distances.append(distance_row)
    
    print(f"Calculated Euclidean distances: depot->client0 = {distances[0][1]}")
    
    matrices = {
        'car': Matrix(durations=durations, distances=distances)
    }
    
    model = Model(jobs=jobs, vehicles=vehicles, matrices=matrices)
    
    # Solve with our implementation
    input_dic, output_dic, error = optimize_vrp_with_enhanced_pyvrp(
        model, max_runtime=10, seed=1234
    )
    
    if error:
        print(f"Error: {error}")
        return None
    
    print(f"Our implementation solution:")
    print(f"  Cost: {output_dic['summary']['cost']}")
    print(f"  Number of routes: {output_dic['summary']['routes']}")
    print(f"  Unassigned: {output_dic['summary']['unassigned']}")
    
    # Show routes
    for i, route in enumerate(output_dic['routes']):
        route_clients = [step['id'] for step in route['steps'] 
                        if step['type'] == 'job']
        print(f"  Route {i}: {route_clients}")
    
    return output_dic

def compare_results():
    """Compare results between direct PyVRP and our implementation"""
    print("\n" + "="*50)
    print("Comparing PyVRP Direct vs Our Implementation")
    print("="*50)
    
    # Solve with both methods
    direct_result = solve_with_pyvrp_directly()
    our_result = solve_with_our_implementation()
    
    if our_result is None:
        print("Our implementation failed!")
        return
    
    print("\n=== Comparison Results ===")
    print(f"Direct PyVRP cost: {direct_result.cost()}")
    print(f"Our implementation cost: {our_result['summary']['cost']}")
    
    cost_diff = abs(direct_result.cost() - our_result['summary']['cost'])
    cost_ratio = cost_diff / direct_result.cost() * 100
    
    print(f"Cost difference: {cost_diff} ({cost_ratio:.2f}%)")
    
    if cost_ratio < 5:  # Within 5% is considered good for metaheuristic
        print("✅ Results are very close! Implementation is working correctly.")
    elif cost_ratio < 15:
        print("⚠️ Results are reasonably close, but may need fine-tuning.")
    else:
        print("❌ Results differ significantly. Implementation needs review.")

if __name__ == "__main__":
    compare_results()