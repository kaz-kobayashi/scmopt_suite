# Debug the vehicle configuration issue
import sys
sys.path.append('.')

from pyvrp_enhanced import *

# Basic classes (same as before)
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

def debug_vehicle_constraints():
    """Debug why clients aren't being assigned to vehicles"""
    print("=== Debugging Vehicle Constraints ===")
    
    # Create simple test data - matching direct PyVRP exactly
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
    
    # Single vehicle with NO fixed costs - to match direct test
    vehicles = [
        Vehicle(id=0, start=[0.0, 0.0], end=[0.0, 0.0], 
               start_index=0, end_index=0, capacity=[100], 
               description="Vehicle_0", costs=VehicleCosts(fixed=0)),
    ]
    
    # Same distance matrix
    durations = [
        [   0,  141,  100,  141,  100],  # from depot
        [ 141,    0,  141,  200,  141],  # from client 0
        [ 100,  141,    0,  141,  200],  # from client 1  
        [ 141,  200,  141,    0,  141],  # from client 2
        [ 100,  141,  200,  141,    0],  # from client 3
    ]
    
    distances = [[d + 300 for d in row] for row in durations]
    matrices = {'car': Matrix(durations=durations, distances=distances)}
    
    model = Model(jobs=jobs, vehicles=vehicles, matrices=matrices)
    
    # Check total demand vs capacity
    total_demand = sum(job.delivery[0] for job in jobs)
    total_capacity = sum(vehicle.capacity[0] for vehicle in vehicles)
    
    print(f"Total demand: {total_demand}")
    print(f"Total capacity: {total_capacity}")
    print(f"Capacity sufficient: {total_capacity >= total_demand}")
    
    # Convert and debug
    builder = convert_metro_to_enhanced_pyvrp(model)
    data = builder.model.data()
    
    print(f"\nConverted model:")
    print(f"  Depots: {len(list(data.depots()))}")
    print(f"  Clients: {len(list(data.clients()))}")
    print(f"  Vehicle types: {len(list(data.vehicle_types()))}")
    
    # Detailed client info
    print(f"\nClient details:")
    for i, client in enumerate(data.clients()):
        print(f"  Client {i}: delivery={client.delivery}, pickup={client.pickup}")
        print(f"    service_duration={client.service_duration}")
        print(f"    required={client.required}, prize={client.prize}")
        print(f"    time window: [{client.tw_early}, {client.tw_late}]")
    
    # Detailed vehicle info
    print(f"\nVehicle details:")
    for i, vt in enumerate(data.vehicle_types()):
        print(f"  Vehicle type {i}:")
        print(f"    capacity={vt.capacity}")
        print(f"    num_available={vt.num_available}")
        print(f"    fixed_cost={vt.fixed_cost}")
        print(f"    time window: [{vt.tw_early}, {vt.tw_late}]")
        print(f"    start_depot={vt.start_depot}, end_depot={vt.end_depot}")
        print(f"    max_duration={vt.max_duration}")
    
    # Check if constraints match what we expect
    print(f"\nConstraint analysis:")
    print(f"  Total client delivery demand: {sum(c.delivery[0] if c.delivery else 0 for c in data.clients())}")
    print(f"  Vehicle capacity: {list(data.vehicle_types())[0].capacity[0] if len(list(data.vehicle_types())) > 0 else 'N/A'}")
    
    return builder, model

if __name__ == "__main__":
    builder, model = debug_vehicle_constraints()
    
    # Try to optimize and see detailed error/warnings
    print(f"\n=== Optimization Attempt ===")
    try:
        input_dic, output_dic, error = optimize_vrp_with_enhanced_pyvrp(
            model, max_runtime=5, seed=1234
        )
        
        if error:
            print(f"Error: {error}")
        else:
            print(f"Success: cost={output_dic['summary']['cost']}, routes={output_dic['summary']['routes']}")
            
    except Exception as e:
        print(f"Exception during optimization: {e}")
        import traceback
        traceback.print_exc()