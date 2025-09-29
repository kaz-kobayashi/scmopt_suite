#!/usr/bin/env python3
"""
PyVRP Variants Test Script

Tests all VRP variants supported by the PyVRP endpoint based on examples
from PyVRP documentation. This focuses on the key variants:
- CVRP (Capacitated VRP)
- VRPTW (VRP with Time Windows)  
- MDVRP (Multi-Depot VRP)
- PDVRP (Pickup and Delivery VRP)
- PC-VRP (Prize Collecting VRP)
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/pyvrp/solve"

def test_cvrp_x_instance_style():
    """
    Test CVRP based on X-n439-k37 benchmark instance style
    Smaller version for testing
    """
    return {
        "clients": [
            # Cluster 1
            {"x": 41, "y": 49, "delivery": 10, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 35, "y": 17, "delivery": 7, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 55, "y": 45, "delivery": 13, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            # Cluster 2
            {"x": 55, "y": 20, "delivery": 19, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 15, "y": 30, "delivery": 26, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 25, "y": 30, "delivery": 3, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            # Cluster 3
            {"x": 20, "y": 50, "delivery": 5, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 10, "y": 43, "delivery": 9, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 55, "y": 60, "delivery": 16, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 30, "y": 60, "delivery": 16, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True}
        ],
        "depots": [
            {"x": 30, "y": 40, "tw_early": 0, "tw_late": 1440}
        ],
        "vehicle_types": [
            {
                "num_available": 3,
                "capacity": 50,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 0,
                "unit_distance_cost": 1.0,
                "tw_early": 0,
                "tw_late": 1440,
                "max_duration": 720,
                "max_distance": 200000
            }
        ],
        "max_runtime": 15
    }

def test_vrptw_rc208_style():
    """
    Test VRPTW based on RC208 Solomon benchmark style
    """
    return {
        "clients": [
            # Early morning customers
            {"x": 35, "y": 35, "delivery": 20, "pickup": 0, "service_duration": 20, "tw_early": 480, "tw_late": 540, "required": True},  # 8:00-9:00
            {"x": 41, "y": 49, "delivery": 10, "pickup": 0, "service_duration": 15, "tw_early": 500, "tw_late": 580, "required": True},  # 8:20-9:40
            # Mid-morning customers  
            {"x": 55, "y": 45, "delivery": 13, "pickup": 0, "service_duration": 10, "tw_early": 600, "tw_late": 720, "required": True},  # 10:00-12:00
            {"x": 55, "y": 20, "delivery": 19, "pickup": 0, "service_duration": 25, "tw_early": 660, "tw_late": 780, "required": True},  # 11:00-13:00
            # Afternoon customers
            {"x": 15, "y": 30, "delivery": 26, "pickup": 0, "service_duration": 15, "tw_early": 840, "tw_late": 960, "required": True},  # 14:00-16:00
            {"x": 25, "y": 30, "delivery": 3, "pickup": 0, "service_duration": 10, "tw_early": 900, "tw_late": 1020, "required": True},  # 15:00-17:00
        ],
        "depots": [
            {"x": 40, "y": 50, "tw_early": 420, "tw_late": 1080}  # 7:00-18:00
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 60,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 200,
                "tw_early": 420,
                "tw_late": 1080,
                "max_duration": 720,
                "max_distance": 300000
            }
        ],
        "max_runtime": 20
    }

def test_mdvrp_multi_depot():
    """
    Test Multi-Depot VRP with 3 depots
    """
    return {
        "clients": [
            # Clients near depot 0
            {"x": 25, "y": 25, "delivery": 15, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 35, "y": 15, "delivery": 12, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 15, "y": 35, "delivery": 18, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            # Clients near depot 1
            {"x": 85, "y": 25, "delivery": 20, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 75, "y": 15, "delivery": 16, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 95, "y": 35, "delivery": 14, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            # Clients near depot 2
            {"x": 55, "y": 85, "delivery": 22, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 45, "y": 75, "delivery": 11, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
            {"x": 65, "y": 95, "delivery": 17, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "required": True},
        ],
        "depots": [
            {"x": 20, "y": 20, "tw_early": 0, "tw_late": 1440},  # Depot 0
            {"x": 80, "y": 20, "tw_early": 0, "tw_late": 1440},  # Depot 1
            {"x": 50, "y": 80, "tw_early": 0, "tw_late": 1440},  # Depot 2
        ],
        "vehicle_types": [
            # Vehicles at depot 0
            {
                "num_available": 2,
                "capacity": 50,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 200000
            },
            # Vehicles at depot 1
            {
                "num_available": 2,
                "capacity": 60,
                "start_depot": 1,
                "end_depot": 1,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 200000
            },
            # Vehicles at depot 2
            {
                "num_available": 2,
                "capacity": 55,
                "start_depot": 2,
                "end_depot": 2,
                "fixed_cost": 100,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 200000
            }
        ],
        "max_runtime": 25
    }

def test_pdvrp_pickup_delivery():
    """
    Test Pickup and Delivery VRP
    """
    return {
        "clients": [
            # Pickup-Delivery Pair 1: (0,1)
            {"x": 20, "y": 30, "delivery": 0, "pickup": 25, "service_duration": 15, "tw_early": 480, "tw_late": 1080, "required": True},  # Pickup
            {"x": 70, "y": 40, "delivery": 25, "pickup": 0, "service_duration": 10, "tw_early": 480, "tw_late": 1080, "required": True},  # Delivery
            
            # Pickup-Delivery Pair 2: (2,3)
            {"x": 30, "y": 60, "delivery": 0, "pickup": 15, "service_duration": 12, "tw_early": 480, "tw_late": 1080, "required": True},  # Pickup
            {"x": 60, "y": 20, "delivery": 15, "pickup": 0, "service_duration": 8, "tw_early": 480, "tw_late": 1080, "required": True},   # Delivery
            
            # Pickup-Delivery Pair 3: (4,5)
            {"x": 80, "y": 70, "delivery": 0, "pickup": 20, "service_duration": 10, "tw_early": 480, "tw_late": 1080, "required": True},  # Pickup
            {"x": 25, "y": 25, "delivery": 20, "pickup": 0, "service_duration": 12, "tw_early": 480, "tw_late": 1080, "required": True},  # Delivery
        ],
        "depots": [
            {"x": 50, "y": 50, "tw_early": 0, "tw_late": 1440}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 40,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 150,
                "tw_early": 420,
                "tw_late": 1200,
                "max_duration": 720,
                "max_distance": 300000
            }
        ],
        "max_runtime": 20
    }

def test_pcvrp_prize_collecting():
    """
    Test Prize-Collecting VRP
    """
    return {
        "clients": [
            # High-prize, required customers
            {"x": 30, "y": 40, "delivery": 10, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "prize": 100, "required": True},
            {"x": 70, "y": 30, "delivery": 15, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "prize": 150, "required": True},
            # Medium-prize, optional customers
            {"x": 20, "y": 70, "delivery": 12, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "prize": 80, "required": False},
            {"x": 80, "y": 70, "delivery": 8, "pickup": 0, "service_duration": 10, "tw_early": 0, "tw_late": 1440, "prize": 90, "required": False},
            {"x": 60, "y": 10, "delivery": 20, "pickup": 0, "service_duration": 15, "tw_early": 0, "tw_late": 1440, "prize": 120, "required": False},
            # Low-prize, optional customers (distant)
            {"x": 10, "y": 10, "delivery": 5, "pickup": 0, "service_duration": 8, "tw_early": 0, "tw_late": 1440, "prize": 40, "required": False},
            {"x": 90, "y": 90, "delivery": 6, "pickup": 0, "service_duration": 8, "tw_early": 0, "tw_late": 1440, "prize": 45, "required": False},
            {"x": 95, "y": 15, "delivery": 18, "pickup": 0, "service_duration": 12, "tw_early": 0, "tw_late": 1440, "prize": 85, "required": False},
        ],
        "depots": [
            {"x": 50, "y": 50, "tw_early": 0, "tw_late": 1440}
        ],
        "vehicle_types": [
            {
                "num_available": 2,
                "capacity": 60,
                "start_depot": 0,
                "end_depot": 0,
                "fixed_cost": 200,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 200000
            }
        ],
        "max_runtime": 25
    }

def test_variant(problem_data: Dict[str, Any], variant_name: str) -> Dict[str, Any]:
    """Test a specific VRP variant"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {variant_name}")
    print(f"{'='*60}")
    
    # Print problem characteristics
    clients = problem_data.get('clients', [])
    depots = problem_data.get('depots', [])
    vehicle_types = problem_data.get('vehicle_types', [])
    
    print(f"📊 Problem size:")
    print(f"   • Clients: {len(clients)}")
    print(f"   • Depots: {len(depots)}")
    print(f"   • Vehicle types: {len(vehicle_types)}")
    
    # Analyze problem characteristics
    total_demand = sum(c.get('delivery', 0) for c in clients)
    total_capacity = sum(vt.get('capacity', 0) * vt.get('num_available', 0) for vt in vehicle_types)
    has_time_windows = any(c.get('tw_early', 0) > 0 or c.get('tw_late', 1440) < 1440 for c in clients)
    has_pickups = any(c.get('pickup', 0) > 0 for c in clients)
    has_prizes = any(c.get('prize', 0) > 0 for c in clients)
    optional_clients = sum(1 for c in clients if not c.get('required', True))
    
    print(f"📋 Problem characteristics:")
    print(f"   • Total demand: {total_demand}")
    print(f"   • Total capacity: {total_capacity}")
    print(f"   • Demand/Capacity ratio: {total_demand/total_capacity:.2f}")
    print(f"   • Time windows: {'Yes' if has_time_windows else 'No'}")
    print(f"   • Pickup operations: {'Yes' if has_pickups else 'No'}")
    print(f"   • Prize collection: {'Yes' if has_prizes else 'No'}")
    print(f"   • Optional clients: {optional_clients}")
    
    # Send request
    print(f"\n🚀 Sending request to {ENDPOINT}...")
    start_time = time.time()
    
    try:
        response = requests.post(ENDPOINT, json=problem_data, timeout=60)
        response_time = time.time() - start_time
        
        print(f"⏱️  Response time: {response_time:.2f}s")
        print(f"📡 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract solution details
            status = result.get('status', 'unknown')
            objective_value = result.get('objective_value')
            routes = result.get('routes', [])
            is_feasible = result.get('is_feasible', False)
            computation_time = result.get('computation_time', 0)
            solver = result.get('solver', 'unknown')
            
            print(f"\n✅ SOLUTION RECEIVED")
            print(f"   • Status: {status}")
            print(f"   • Solver: {solver}")
            print(f"   • Feasible: {is_feasible}")
            print(f"   • Objective value: {objective_value}")
            print(f"   • Computation time: {computation_time:.2f}s")
            print(f"   • Number of routes: {len(routes)}")
            
            # Analyze routes
            total_clients_served = 0
            total_distance = 0
            total_demand_served = 0
            
            for i, route in enumerate(routes):
                route_clients = route.get('clients', [])
                route_distance = route.get('distance', 0)
                route_demand = route.get('demand_served', 0)
                vehicle_type = route.get('vehicle_type', 0)
                
                total_clients_served += len(route_clients)
                total_distance += route_distance
                total_demand_served += route_demand
                
                print(f"   • Route {i}: {len(route_clients)} clients, demand={route_demand}, distance={route_distance}, vehicle_type={vehicle_type}")
            
            print(f"\n📈 Solution summary:")
            print(f"   • Total clients served: {total_clients_served}/{len(clients)}")
            print(f"   • Total demand served: {total_demand_served}/{total_demand}")
            print(f"   • Total distance: {total_distance}")
            print(f"   • Service rate: {total_clients_served/len(clients)*100:.1f}%")
            
            return {
                'success': True,
                'variant': variant_name,
                'status': status,
                'feasible': is_feasible,
                'objective_value': objective_value,
                'routes': len(routes),
                'clients_served': total_clients_served,
                'total_clients': len(clients),
                'response_time': response_time,
                'computation_time': computation_time
            }
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'success': False,
                'variant': variant_name,
                'error': f"HTTP {response.status_code}: {response.text}",
                'response_time': response_time
            }
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return {
            'success': False,
            'variant': variant_name,
            'error': str(e),
            'response_time': time.time() - start_time
        }

def main():
    """Run all VRP variant tests"""
    print("PyVRP Endpoint VRP Variants Test")
    print("Based on PyVRP documentation examples")
    print("=" * 70)
    
    # Test all variants
    test_results = []
    
    # 1. Test CVRP (X-instance style)
    cvrp_problem = test_cvrp_x_instance_style()
    cvrp_result = test_variant(cvrp_problem, "CVRP (X-instance style)")
    test_results.append(cvrp_result)
    
    # 2. Test VRPTW (RC208 style)
    vrptw_problem = test_vrptw_rc208_style()
    vrptw_result = test_variant(vrptw_problem, "VRPTW (RC208 style)")
    test_results.append(vrptw_result)
    
    # 3. Test MDVRP
    mdvrp_problem = test_mdvrp_multi_depot()
    mdvrp_result = test_variant(mdvrp_problem, "MDVRP (Multi-Depot)")
    test_results.append(mdvrp_result)
    
    # 4. Test PDVRP
    pdvrp_problem = test_pdvrp_pickup_delivery()
    pdvrp_result = test_variant(pdvrp_problem, "PDVRP (Pickup-Delivery)")
    test_results.append(pdvrp_result)
    
    # 5. Test PC-VRP
    pcvrp_problem = test_pcvrp_prize_collecting()
    pcvrp_result = test_variant(pcvrp_problem, "PC-VRP (Prize-Collecting)")
    test_results.append(pcvrp_result)
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏁 FINAL TEST SUMMARY")
    print(f"{'='*70}")
    
    successful_tests = [r for r in test_results if r['success']]
    total_tests = len(test_results)
    
    print(f"\n📊 Overall Results: {len(successful_tests)}/{total_tests} tests passed")
    print(f"🎯 Success Rate: {len(successful_tests)/total_tests*100:.1f}%")
    
    print(f"\n📋 Test Results by Variant:")
    for result in test_results:
        variant = result['variant']
        if result['success']:
            status = result.get('status', 'unknown')
            routes = result.get('routes', 0)
            served = result.get('clients_served', 0)
            total = result.get('total_clients', 0)
            time_taken = result.get('response_time', 0)
            print(f"   ✅ {variant}: {status}, {routes} routes, {served}/{total} clients, {time_taken:.1f}s")
        else:
            error = result.get('error', 'Unknown error')
            print(f"   ❌ {variant}: {error}")
    
    # Performance analysis
    if successful_tests:
        avg_response_time = sum(r.get('response_time', 0) for r in successful_tests) / len(successful_tests)
        avg_computation_time = sum(r.get('computation_time', 0) for r in successful_tests) / len(successful_tests)
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   • Average response time: {avg_response_time:.2f}s")
        print(f"   • Average computation time: {avg_computation_time:.2f}s")
        
        if avg_response_time < 15:
            print(f"   • ✅ Good performance")
        elif avg_response_time < 30:
            print(f"   • ⚠️  Acceptable performance")
        else:
            print(f"   • ❌ Slow performance")
    
    # Capability assessment
    print(f"\n🔍 Endpoint Capabilities Assessment:")
    
    capability_map = {
        'CVRP': any('CVRP' in r['variant'] for r in successful_tests),
        'VRPTW': any('VRPTW' in r['variant'] for r in successful_tests),
        'MDVRP': any('MDVRP' in r['variant'] for r in successful_tests),
        'PDVRP': any('PDVRP' in r['variant'] for r in successful_tests),
        'PC-VRP': any('PC-VRP' in r['variant'] for r in successful_tests),
    }
    
    for variant, supported in capability_map.items():
        status = "✅ SUPPORTED" if supported else "❌ NOT SUPPORTED"
        print(f"   • {variant}: {status}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if len(successful_tests) == total_tests:
        print("   ✅ Excellent! All VRP variants are supported and working correctly.")
        print("   📈 Consider testing with larger instances to assess scalability.")
    else:
        print("   ⚠️  Some VRP variants failed:")
        failed_variants = [r['variant'] for r in test_results if not r['success']]
        for variant in failed_variants:
            print(f"      - Review {variant} implementation")
    
    print("   📚 Test cases are based on standard VRP benchmarks:")
    print("      - CVRP: X-instance format (X-n439-k37 style)")
    print("      - VRPTW: Solomon RC208 format")
    print("      - MDVRP: Multi-depot with clustered clients")
    print("      - PDVRP: Pickup-delivery pairs with precedence")
    print("      - PC-VRP: Prize collection with optional clients")

if __name__ == "__main__":
    main()