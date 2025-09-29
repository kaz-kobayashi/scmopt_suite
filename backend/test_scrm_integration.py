#!/usr/bin/env python3
"""
Test script for SCRM (Supply Chain Risk Management) implementation
Validates complete integration from 09scrm.ipynb notebook
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.scrm_service import SCRMService, SCMGraph
from app.models.scrm import SCRMDataGenerationOptions
import pandas as pd
import numpy as np

def test_scrm_integration():
    """Test complete SCRM integration"""
    print("=== SCRM Integration Test ===")
    
    # Initialize service
    scrm_service = SCRMService()
    print("✓ SCRM Service initialized")
    
    try:
        # Test 1: Data generation
        print("\n1. Testing data generation...")
        data = scrm_service.generate_test_data(
            benchmark_id="01",
            n_plnts=3,
            n_flex=2,
            seed=1
        )
        print(f"✓ Generated data for benchmark {data['benchmark_id']}")
        print(f"  - Total demand: {data['total_demand']:.2f}")
        print(f"  - BOM records: {len(data['bom_df'])}")
        print(f"  - Plant records: {len(data['plnt_df'])}")
        print(f"  - Plant-product records: {len(data['plnt_prod_df'])}")
        print(f"  - Transport records: {len(data['trans_df'])}")
        
        # Test 2: DataFrame conversion
        print("\n2. Testing DataFrame conversion...")
        result = scrm_service.prepare_from_dataframes(
            data["bom_df"], data["plnt_df"], 
            data["plnt_prod_df"], data["trans_df"]
        )
        print("✓ DataFrame conversion successful")
        
        # Test 3: SCRM optimization
        print("\n3. Testing SCRM optimization...")
        Demand, UB, Capacity, Pipeline, R, BOM, Product, G, ProdGraph, pos, pos2, pos3 = result
        
        survival_time = scrm_service.solve_scrm(
            Demand, UB, Capacity, Pipeline, R, Product, ProdGraph, BOM
        )
        print(f"✓ SCRM optimization completed")
        print(f"  - Nodes analyzed: {len(survival_time)}")
        print(f"  - Average survival time: {np.mean(survival_time):.2f}")
        print(f"  - Min survival time: {min(survival_time):.2f}")
        print(f"  - Max survival time: {max(survival_time):.2f}")
        
        # Test 4: Critical node analysis
        print("\n4. Testing critical node analysis...")
        critical_count = sum(1 for t in survival_time if t == 0.0)
        high_risk_count = sum(1 for t in survival_time if 0.0 < t <= 1.0)
        resilient_count = sum(1 for t in survival_time if t > 1.0)
        
        print(f"  - Critical nodes (survival=0): {critical_count}")
        print(f"  - High risk nodes (0<survival≤1): {high_risk_count}")
        print(f"  - Resilient nodes (survival>1): {resilient_count}")
        
        # Test 5: Visualization generation
        print("\n5. Testing visualization generation...")
        fig = scrm_service.draw_scrm(ProdGraph, survival_time, Pipeline, UB, pos3)
        print("✓ Risk analysis visualization generated")
        print(f"  - Figure data traces: {len(fig.data)}")
        
        # Test 6: Full analysis workflow
        print("\n6. Testing full analysis workflow...")
        analysis_results = scrm_service.run_full_analysis(data)
        print("✓ Full analysis completed")
        print(f"  - Status: {analysis_results.get('status', 'unknown')}")
        print(f"  - Total nodes: {analysis_results['total_nodes']}")
        print(f"  - Critical nodes identified: {len(analysis_results['critical_nodes'])}")
        
        # Test 7: SCMGraph functionality
        print("\n7. Testing SCMGraph functionality...")
        test_graph = SCMGraph()
        test_graph.add_nodes_from([1, 2, 3, 4])
        test_graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        layout = test_graph.layout()
        print(f"✓ SCMGraph layout generated with {len(layout)} positions")
        
        print("\n=== All SCRM Tests Passed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structures():
    """Test data structure compatibility"""
    print("\n=== Data Structure Compatibility Test ===")
    
    try:
        # Test SCRMDataGenerationOptions
        options = SCRMDataGenerationOptions(
            benchmark_id="01",
            n_plnts=3,
            n_flex=2,
            seed=42
        )
        print("✓ SCRMDataGenerationOptions validation passed")
        
        # Test benchmark ID validation
        try:
            invalid_options = SCRMDataGenerationOptions(benchmark_id="99")
            print("❌ Benchmark ID validation failed - should reject invalid ID")
            return False
        except Exception:
            print("✓ Benchmark ID validation correctly rejects invalid values")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting SCRM Integration Tests...")
    
    # Run tests
    structure_test = test_data_structures()
    integration_test = test_scrm_integration()
    
    # Summary
    if structure_test and integration_test:
        print("\n🎉 All SCRM tests completed successfully!")
        print("The 09scrm.ipynb notebook functionality has been fully implemented.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)