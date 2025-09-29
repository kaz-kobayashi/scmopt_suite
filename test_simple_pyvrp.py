# Simple PyVRP Test to check API
import sys
sys.path.append('.')

from pyvrp import Model as PyVRPModel

def test_simple_pyvrp_api():
    """PyVRPの実際のAPIを調べる"""
    
    # モデル作成
    model = PyVRPModel()
    
    # デポ追加
    depot = model.add_depot(x=0, y=0)
    print(f"Added depot: {depot}")
    
    # 顧客追加
    client1 = model.add_client(x=1000, y=1000, delivery=[10], pickup=[0])
    client2 = model.add_client(x=2000, y=2000, delivery=[15], pickup=[5])
    print(f"Added clients: {client1}, {client2}")
    
    # 車両追加
    vehicle_type = model.add_vehicle_type(
        num_available=1,
        capacity=[100],
        start_depot=depot,
        end_depot=depot
    )
    print(f"Added vehicle type: {vehicle_type}")
    
    # エッジ追加
    locations = [depot, client1, client2]
    for i, from_loc in enumerate(locations):
        for j, to_loc in enumerate(locations):
            if i != j:
                distance = abs(i - j) * 1000 + 500
                duration = distance + 300
                model.add_edge(from_loc, to_loc, distance=distance, duration=duration)
    
    # 最適化実行
    from pyvrp.stop import MaxRuntime
    result = model.solve(stop=MaxRuntime(10), seed=1234)
    
    print(f"\nResult type: {type(result)}")
    print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
    
    print(f"\nCost: {result.cost()}")
    print(f"Best solution type: {type(result.best)}")
    print(f"Best solution attributes: {[attr for attr in dir(result.best) if not attr.startswith('_')]}")
    
    # ルート情報を調べる
    try:
        routes = result.best.routes()
        print(f"\nRoutes: {routes}")
        print(f"Number of routes: {len(routes)}")
        
        for i, route in enumerate(routes):
            print(f"Route {i}: {list(route)}")
            print(f"Route type: {type(route)}")
            print(f"Route attributes: {[attr for attr in dir(route) if not attr.startswith('_')]}")
            
    except Exception as e:
        print(f"Routes error: {e}")
    
    # 未割り当ての調べ方
    try:
        unassigned = result.best.unassigned()
        print(f"Unassigned: {list(unassigned)}")
    except AttributeError:
        print("No unassigned() method")
        try:
            # 他の可能性を探す
            data = model.data()
            all_clients = list(range(len([c for c in data.clients()])))
            visited_clients = set()
            for route in result.best.routes():
                for client_id in route:
                    if client_id not in [d.idx for d in data.depots()]:
                        visited_clients.add(client_id)
            
            unassigned = [c for c in all_clients if c not in visited_clients]
            print(f"Calculated unassigned: {unassigned}")
        except Exception as e:
            print(f"Unassigned calculation error: {e}")

if __name__ == "__main__":
    test_simple_pyvrp_api()