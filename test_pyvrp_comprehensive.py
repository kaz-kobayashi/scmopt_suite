#!/usr/bin/env python3
"""
Comprehensive PyVRP Endpoint Test Script

This script tests the PyVRP endpoint at /api/pyvrp/solve with all VRP variants
based on examples from https://pyvrp.org/examples/. Tests the following variants:
- Basic CVRP (Capacitated VRP)
- VRPTW (VRP with Time Windows)  
- MDVRP (Multi-Depot VRP)
- PVRP (Pickup and Delivery VRP)
- PC-VRP (Prize Collecting VRP)

The script validates:
- Request/response structure
- Solution feasibility
- Endpoint capabilities
- Error handling
"""

import requests
import json
import time
import random
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"  # Adjust as needed
ENDPOINT = f"{BASE_URL}/api/pyvrp/solve"
TIMEOUT = 120  # 2 minutes timeout for each test

@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    success: bool
    response_time: float
    status_code: int
    error_message: str = ""
    objective_value: Optional[float] = None
    num_routes: int = 0
    is_feasible: bool = False
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

class PyVRPEndpointTester:
    """Comprehensive PyVRP endpoint testing class"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/pyvrp/solve"
        self.test_results: List[TestResult] = []
        
    def generate_coordinates(self, n: int, center_x: int = 50, center_y: int = 50, radius: int = 30) -> List[tuple]:
        """Generate realistic coordinates for testing"""
        coords = []
        for i in range(n):
            if i == 0:  # Depot at center
                coords.append((center_x, center_y))
            else:
                # Random coordinates around center
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(5, radius)
                x = int(center_x + r * math.cos(angle))
                y = int(center_y + r * math.sin(angle))
                coords.append((x, y))
        return coords
    
    def create_basic_cvrp_problem(self) -> Dict[str, Any]:
        """
        Create a basic CVRP test problem similar to X-n439-k37 benchmark
        but smaller for testing purposes
        """
        n_clients = 20
        coords = self.generate_coordinates(n_clients + 1)  # +1 for depot
        
        # Create clients with random demands
        clients = []
        for i in range(1, n_clients + 1):  # Skip depot
            x, y = coords[i]
            clients.append({
                "x": x,
                "y": y,
                "delivery": random.randint(10, 50),
                "pickup": 0,
                "service_duration": 10,
                "tw_early": 0,
                "tw_late": 1440,  # Full day
                "prize": 0,
                "required": True
            })
        
        # Create depot
        depot_x, depot_y = coords[0]
        depots = [{
            "x": depot_x,
            "y": depot_y,
            "tw_early": 0,
            "tw_late": 1440
        }]
        
        # Create vehicle types
        vehicle_types = [{
            "num_available": 5,
            "capacity": 200,
            "start_depot": 0,
            "end_depot": 0,
            "fixed_cost": 100,
            "unit_distance_cost": 1.0,
            "unit_duration_cost": 0.1,
            "tw_early": 480,  # 8:00 AM
            "tw_late": 1080,  # 6:00 PM
            "max_duration": 600,  # 10 hours
            "max_distance": 200000
        }]
        
        return {
            "clients": clients,
            "depots": depots,
            "vehicle_types": vehicle_types,
            "max_runtime": 30
        }
    
    def create_vrptw_problem(self) -> Dict[str, Any]:
        """
        Create VRPTW test problem similar to RC208 benchmark
        with time window constraints
        """
        n_clients = 15
        coords = self.generate_coordinates(n_clients + 1)
        
        # Create clients with tight time windows
        clients = []
        for i in range(1, n_clients + 1):
            x, y = coords[i]
            
            # Create realistic time windows (business hours)
            earliest = random.randint(480, 600)  # 8:00-10:00 AM
            latest = earliest + random.randint(60, 180)  # 1-3 hour window
            
            clients.append({
                "x": x,
                "y": y,
                "delivery": random.randint(10, 30),
                "pickup": 0,
                "service_duration": random.randint(15, 30),  # Longer service times
                "tw_early": earliest,
                "tw_late": latest,
                "prize": 0,
                "required": True
            })
        
        depot_x, depot_y = coords[0]
        depots = [{
            "x": depot_x,
            "y": depot_y,
            "tw_early": 420,  # 7:00 AM
            "tw_late": 1200   # 8:00 PM
        }]
        
        vehicle_types = [{
            "num_available": 4,
            "capacity": 150,
            "start_depot": 0,
            "end_depot": 0,
            "fixed_cost": 200,
            "tw_early": 420,
            "tw_late": 1200,
            "max_duration": 720,  # 12 hours
            "max_distance": 300000
        }]
        
        return {
            "clients": clients,
            "depots": depots,
            "vehicle_types": vehicle_types,
            "max_runtime": 45
        }
    
    def create_mdvrp_problem(self) -> Dict[str, Any]:
        """
        Create Multi-Depot VRP test problem
        """
        n_clients = 16
        n_depots = 3
        
        # Generate coordinates with multiple depot locations
        depot_coords = [(20, 20), (80, 20), (50, 80)]
        client_coords = []
        
        # Generate clients clustered around depots
        for i in range(n_clients):
            depot_idx = i % n_depots
            depot_x, depot_y = depot_coords[depot_idx]
            
            # Add some randomness around each depot
            x = depot_x + random.randint(-15, 15)
            y = depot_y + random.randint(-15, 15)
            client_coords.append((x, y))
        
        # Create clients
        clients = []
        for i, (x, y) in enumerate(client_coords):
            clients.append({
                "x": x,
                "y": y,
                "delivery": random.randint(10, 40),
                "pickup": 0,
                "service_duration": 10,
                "tw_early": 0,
                "tw_late": 1440,
                "prize": 0,
                "required": True
            })
        
        # Create multiple depots
        depots = []
        for x, y in depot_coords:
            depots.append({
                "x": x,
                "y": y,
                "tw_early": 0,
                "tw_late": 1440
            })
        
        # Create vehicle types for each depot
        vehicle_types = []
        for depot_idx in range(n_depots):
            vehicle_types.append({
                "num_available": 2,
                "capacity": 120,
                "start_depot": depot_idx,
                "end_depot": depot_idx,
                "fixed_cost": 150,
                "tw_early": 480,
                "tw_late": 1080,
                "max_duration": 600,
                "max_distance": 200000
            })
        
        return {
            "clients": clients,
            "depots": depots,
            "vehicle_types": vehicle_types,
            "max_runtime": 60
        }
    
    def create_pdvrp_problem(self) -> Dict[str, Any]:
        """
        Create Pickup and Delivery VRP test problem
        """
        n_pairs = 8  # 8 pickup-delivery pairs
        coords = self.generate_coordinates(n_pairs * 2 + 1)  # pairs + depot
        
        clients = []
        depot_x, depot_y = coords[0]
        
        # Create pickup-delivery pairs
        for i in range(n_pairs):
            pickup_idx = i * 2 + 1
            delivery_idx = i * 2 + 2
            
            pickup_x, pickup_y = coords[pickup_idx]
            delivery_x, delivery_y = coords[delivery_idx]
            
            demand = random.randint(15, 35)
            
            # Pickup location
            clients.append({
                "x": pickup_x,
                "y": pickup_y,
                "delivery": 0,
                "pickup": demand,
                "service_duration": 15,
                "tw_early": 480,  # 8:00 AM
                "tw_late": 1080,  # 6:00 PM
                "prize": 0,
                "required": True
            })
            
            # Delivery location
            clients.append({
                "x": delivery_x,
                "y": delivery_y,
                "delivery": demand,
                "pickup": 0,
                "service_duration": 10,
                "tw_early": 480,
                "tw_late": 1080,
                "prize": 0,
                "required": True
            })
        
        depots = [{
            "x": depot_x,
            "y": depot_y,
            "tw_early": 0,
            "tw_late": 1440
        }]
        
        vehicle_types = [{
            "num_available": 4,
            "capacity": 100,
            "start_depot": 0,
            "end_depot": 0,
            "fixed_cost": 100,
            "tw_early": 420,
            "tw_late": 1200,
            "max_duration": 720,
            "max_distance": 250000
        }]
        
        return {
            "clients": clients,
            "depots": depots,
            "vehicle_types": vehicle_types,
            "max_runtime": 45
        }
    
    def create_pcvrp_problem(self) -> Dict[str, Any]:
        """
        Create Prize-Collecting VRP test problem
        """
        n_clients = 20
        coords = self.generate_coordinates(n_clients + 1)
        
        clients = []
        total_prize = 0
        
        for i in range(1, n_clients + 1):
            x, y = coords[i]
            prize = random.randint(50, 200)
            demand = random.randint(5, 25)
            
            # Some clients are optional (not required)
            required = random.random() > 0.3  # 70% required, 30% optional
            
            clients.append({
                "x": x,
                "y": y,
                "delivery": demand,
                "pickup": 0,
                "service_duration": 10,
                "tw_early": 0,
                "tw_late": 1440,
                "prize": prize,
                "required": required
            })
            
            total_prize += prize
        
        depot_x, depot_y = coords[0]
        depots = [{
            "x": depot_x,
            "y": depot_y,
            "tw_early": 0,
            "tw_late": 1440
        }]
        
        vehicle_types = [{
            "num_available": 3,
            "capacity": 150,
            "start_depot": 0,
            "end_depot": 0,
            "fixed_cost": 200,
            "tw_early": 480,
            "tw_late": 1080,
            "max_duration": 600,
            "max_distance": 200000
        }]
        
        return {
            "clients": clients,
            "depots": depots,
            "vehicle_types": vehicle_types,
            "max_runtime": 45,
            "solver_config": {
                "target_objective": total_prize * 0.7  # Aim for 70% of total prize
            }
        }
    
    def send_request(self, problem_data: Dict[str, Any], test_name: str) -> TestResult:
        """Send request to PyVRP endpoint and validate response"""
        logger.info(f"Testing {test_name}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.endpoint,
                json=problem_data,
                timeout=TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = time.time() - start_time
            
            # Basic response validation
            if response.status_code != 200:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
            
            # Parse JSON response
            try:
                result_data = response.json()
            except json.JSONDecodeError as e:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    error_message=f"Invalid JSON response: {e}"
                )
            
            # Validate response structure
            validation_errors = self._validate_response_structure(result_data)
            
            # Extract key metrics
            objective_value = result_data.get("objective_value")
            routes = result_data.get("routes", [])
            is_feasible = result_data.get("is_feasible", False)
            status = result_data.get("status", "unknown")
            
            # Validate solution feasibility
            feasibility_errors = self._validate_solution_feasibility(
                problem_data, result_data
            )
            validation_errors.extend(feasibility_errors)
            
            return TestResult(
                test_name=test_name,
                success=len(validation_errors) == 0 and status != "error",
                response_time=response_time,
                status_code=response.status_code,
                objective_value=objective_value,
                num_routes=len(routes),
                is_feasible=is_feasible,
                validation_errors=validation_errors
            )
            
        except requests.exceptions.Timeout:
            return TestResult(
                test_name=test_name,
                success=False,
                response_time=TIMEOUT,
                status_code=0,
                error_message="Request timeout"
            )
        except requests.exceptions.ConnectionError:
            return TestResult(
                test_name=test_name,
                success=False,
                response_time=time.time() - start_time,
                status_code=0,
                error_message="Connection error - server not available"
            )
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                response_time=time.time() - start_time,
                status_code=0,
                error_message=f"Unexpected error: {e}"
            )
    
    def _validate_response_structure(self, response_data: Dict[str, Any]) -> List[str]:
        """Validate the structure of the response matches UnifiedVRPSolution"""
        errors = []
        
        required_fields = [
            "status", "objective_value", "routes", "computation_time",
            "solver", "is_feasible"
        ]
        
        for field in required_fields:
            if field not in response_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate routes structure
        routes = response_data.get("routes", [])
        if not isinstance(routes, list):
            errors.append("Routes must be a list")
        else:
            for i, route in enumerate(routes):
                if not isinstance(route, dict):
                    errors.append(f"Route {i} must be a dictionary")
                    continue
                
                route_required_fields = [
                    "vehicle_type", "start_depot", "end_depot", "clients",
                    "distance", "duration", "total_cost"
                ]
                
                for field in route_required_fields:
                    if field not in route:
                        errors.append(f"Route {i} missing field: {field}")
        
        return errors
    
    def _validate_solution_feasibility(
        self, problem_data: Dict[str, Any], solution_data: Dict[str, Any]
    ) -> List[str]:
        """Validate that the solution is feasible for the given problem"""
        errors = []
        
        routes = solution_data.get("routes", [])
        clients = problem_data.get("clients", [])
        vehicle_types = problem_data.get("vehicle_types", [])
        
        # Check if all required clients are visited
        visited_clients = set()
        for route in routes:
            route_clients = route.get("clients", [])
            for client_idx in route_clients:
                if client_idx >= len(clients):
                    errors.append(f"Invalid client index: {client_idx}")
                else:
                    visited_clients.add(client_idx)
        
        # Check required clients
        for i, client in enumerate(clients):
            if client.get("required", True) and i not in visited_clients:
                errors.append(f"Required client {i} not visited")
        
        # Check capacity constraints
        for route_idx, route in enumerate(routes):
            vehicle_type_idx = route.get("vehicle_type", 0)
            if vehicle_type_idx >= len(vehicle_types):
                errors.append(f"Invalid vehicle type index in route {route_idx}")
                continue
            
            vehicle_capacity = vehicle_types[vehicle_type_idx].get("capacity", 0)
            route_demand = route.get("demand_served", 0)
            
            if route_demand > vehicle_capacity:
                errors.append(
                    f"Route {route_idx} exceeds capacity: {route_demand} > {vehicle_capacity}"
                )
        
        return errors
    
    def run_all_tests(self) -> None:
        """Run all VRP variant tests"""
        logger.info("Starting comprehensive PyVRP endpoint testing...")
        
        # Test basic CVRP
        cvrp_problem = self.create_basic_cvrp_problem()
        cvrp_result = self.send_request(cvrp_problem, "Basic CVRP")
        self.test_results.append(cvrp_result)
        
        # Test VRPTW
        vrptw_problem = self.create_vrptw_problem()
        vrptw_result = self.send_request(vrptw_problem, "VRPTW")
        self.test_results.append(vrptw_result)
        
        # Test MDVRP
        mdvrp_problem = self.create_mdvrp_problem()
        mdvrp_result = self.send_request(mdvrp_problem, "MDVRP")
        self.test_results.append(mdvrp_result)
        
        # Test PDVRP
        pdvrp_problem = self.create_pdvrp_problem()
        pdvrp_result = self.send_request(pdvrp_problem, "PDVRP (Pickup-Delivery)")
        self.test_results.append(pdvrp_result)
        
        # Test PC-VRP
        pcvrp_problem = self.create_pcvrp_problem()
        pcvrp_result = self.send_request(pcvrp_problem, "PC-VRP (Prize-Collecting)")
        self.test_results.append(pcvrp_result)
        
        # Test error handling with invalid data
        invalid_problem = {"clients": [], "depots": [], "vehicle_types": []}
        error_result = self.send_request(invalid_problem, "Error Handling Test")
        self.test_results.append(error_result)
    
    def print_summary(self) -> None:
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("PYVRP ENDPOINT COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        
        print(f"\nOverall Results: {successful_tests}/{total_tests} tests passed")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in self.test_results:
            status_icon = "✅" if result.success else "❌"
            print(f"\n{status_icon} {result.test_name}")
            print(f"   Status Code: {result.status_code}")
            print(f"   Response Time: {result.response_time:.2f}s")
            
            if result.success:
                print(f"   Objective Value: {result.objective_value}")
                print(f"   Number of Routes: {result.num_routes}")
                print(f"   Feasible Solution: {result.is_feasible}")
            else:
                print(f"   Error: {result.error_message}")
                if result.validation_errors:
                    print(f"   Validation Errors:")
                    for error in result.validation_errors:
                        print(f"     - {error}")
        
        print("\n" + "="*80)
        print("ENDPOINT CAPABILITIES ASSESSMENT")
        print("="*80)
        
        # Analyze capabilities
        vrp_variants = {
            "Basic CVRP": next((r for r in self.test_results if "CVRP" in r.test_name), None),
            "VRPTW": next((r for r in self.test_results if "VRPTW" in r.test_name), None),
            "MDVRP": next((r for r in self.test_results if "MDVRP" in r.test_name), None),
            "PDVRP": next((r for r in self.test_results if "Pickup-Delivery" in r.test_name), None),
            "PC-VRP": next((r for r in self.test_results if "Prize-Collecting" in r.test_name), None),
        }
        
        print("\nVRP Variant Support:")
        for variant, result in vrp_variants.items():
            if result:
                status = "✅ SUPPORTED" if result.success else "❌ FAILED"
                print(f"  {variant}: {status}")
                if result.success and result.response_time:
                    print(f"    - Avg response time: {result.response_time:.2f}s")
                    if result.objective_value:
                        print(f"    - Solution quality: Objective = {result.objective_value}")
        
        print(f"\nError Handling:")
        error_test = next((r for r in self.test_results if "Error" in r.test_name), None)
        if error_test:
            if error_test.status_code == 400 or error_test.status_code == 422:
                print("  ✅ Proper error handling for invalid input")
            else:
                print("  ❌ Error handling needs improvement")
        
        # Performance assessment
        avg_response_time = sum(r.response_time for r in self.test_results if r.success) / max(1, successful_tests)
        print(f"\nPerformance Assessment:")
        print(f"  Average response time: {avg_response_time:.2f}s")
        
        if avg_response_time < 10:
            print("  ✅ Good performance for test instances")
        elif avg_response_time < 30:
            print("  ⚠️  Acceptable performance")
        else:
            print("  ❌ Slow performance - may need optimization")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if successful_tests == total_tests:
            print("✅ All tests passed! The endpoint successfully handles all VRP variants.")
        else:
            print("⚠️  Some tests failed. Consider the following improvements:")
            
            for result in self.test_results:
                if not result.success:
                    print(f"  - Fix issues with {result.test_name}: {result.error_message}")
        
        print("\n📝 Additional recommendations:")
        print("  - Add more comprehensive error messages for debugging")
        print("  - Consider adding solution validation in the response")
        print("  - Test with larger instances to assess scalability")
        print("  - Add benchmark comparison with known optimal solutions")

def main():
    """Main test execution"""
    print("PyVRP Endpoint Comprehensive Test Script")
    print("Testing endpoint:", ENDPOINT)
    print("Timeout per test:", TIMEOUT, "seconds")
    print("-" * 50)
    
    # Initialize tester
    tester = PyVRPEndpointTester()
    
    # Run all tests
    tester.run_all_tests()
    
    # Print results
    tester.print_summary()
    
    # Save detailed results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"pyvrp_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        results_data = {
            "timestamp": timestamp,
            "endpoint": ENDPOINT,
            "summary": {
                "total_tests": len(tester.test_results),
                "successful_tests": sum(1 for r in tester.test_results if r.success),
                "avg_response_time": sum(r.response_time for r in tester.test_results if r.success) / max(1, sum(1 for r in tester.test_results if r.success))
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "response_time": r.response_time,
                    "status_code": r.status_code,
                    "error_message": r.error_message,
                    "objective_value": r.objective_value,
                    "num_routes": r.num_routes,
                    "is_feasible": r.is_feasible,
                    "validation_errors": r.validation_errors
                }
                for r in tester.test_results
            ]
        }
        json.dump(results_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()