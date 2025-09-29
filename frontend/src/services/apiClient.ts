import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // Increased to 2 minutes for long-running analyses
});

// Add request interceptor to include JWT token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('scmopt_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('scmopt_token');
      localStorage.removeItem('scmopt_user');
      window.location.reload(); // Redirect to login
    }
    return Promise.reject(error);
  }
);

export interface ABCAnalysisResult {
  aggregated_data: Array<{
    [key: string]: string | number;
    abc: string;
    rank: number;
  }>;
  classified_data: Array<{
    [key: string]: string | number;
    abc: string;
    rank: number;
  }>;
  categories: {
    [key: number]: string[];
  };
  summary: {
    total_items: number;
    total_value: number;
    thresholds: number[];
  };
}

export interface EOQResult {
  optimal_order_quantity: number;
  total_annual_cost: number;
  annual_ordering_cost: number;
  annual_holding_cost: number;
  cycle_time_periods: number;
  parameters: {
    fixed_cost: number;
    demand_rate: number;
    holding_cost: number;
    backorder_cost: number;
    interest_rate: number;
    unit_cost: number;
    service_level: number;
  };
}

export interface SimulationResult {
  simulation_results: {
    policy_type: string;
    average_cost_per_period: number;
    cost_standard_deviation: number;
    cost_range: {
      min: number;
      max: number;
    };
    confidence_interval_95: {
      lower: number;
      upper: number;
    };
    average_inventory_level: number;
    number_of_simulations: number;
    periods_per_simulation: number;
  };
  parameters: {
    [key: string]: number | null;
  };
}

export interface CO2Result {
  emissions_calculation: {
    fuel_consumption_L_per_ton_km: number;
    co2_emissions_g_per_ton_km: number;
    fuel_type: string;
    vehicle_capacity_tons: number;
    loading_rate: number;
  };
  annual_estimates: {
    estimated_annual_distance_km: number;
    estimated_annual_fuel_consumption_L: number;
    estimated_annual_co2_emissions_kg: number;
  };
  optimization_suggestions: {
    improve_loading_rate: boolean;
    consider_larger_vehicle: boolean;
    efficiency_score: number;
  };
}

export interface DistanceMatrixResult {
  distance_matrix: {
    [location: string]: {
      [destination: string]: number;
    };
  };
  duration_matrix: {
    [location: string]: {
      [destination: string]: number;
    };
  };
  locations: string[];
  matrix_size: number;
  units: {
    distance: string;
    duration: string;
  };
}

// PyVRP Types
export interface LocationModel {
  name: string;
  lat: number;
  lon: number;
  demand: number;
}

export interface TimeWindow {
  earliest: number;
  latest: number;
}

export interface PickupDeliveryPair {
  pickup_location_idx: number;
  delivery_location_idx: number;
  demand: number;
}

export interface DepotModel {
  name: string;
  lat: number;
  lon: number;
  capacity: number;
  num_vehicles: number;
  time_window?: TimeWindow;
}

export interface RouteModel {
  route_id: number;
  sequence: number[];
  locations: string[];
  distance: number;
  total_demand: number;
  num_stops: number;
  arrival_times?: number[];
  total_prize?: number;
  capacity_utilization?: number;
  pickup_delivery_info?: any[];
}

export interface VRPSolution {
  status: string;
  objective_value: number;
  routes: RouteModel[];
  total_distance: number;
  num_vehicles_used: number;
  computation_time: number;
  solver: string;
  problem_type: string;
  message?: string;
  total_demand_served?: number;
  time_window_violations?: number;
  routes_by_depot?: { [key: string]: RouteModel[] };
  total_prize?: number;
  min_prize_met?: boolean;
}

export interface CVRPRequest {
  locations: LocationModel[];
  depot_index?: number;
  vehicle_capacity: number;
  num_vehicles?: number;
  max_runtime?: number;
}

export interface VRPTWRequest {
  locations: LocationModel[];
  time_windows: TimeWindow[];
  service_times: number[];
  depot_index?: number;
  vehicle_capacity: number;
  num_vehicles?: number;
  max_runtime?: number;
}

export interface MDVRPRequest {
  locations: LocationModel[];
  depots: DepotModel[];
  depot_indices: number[];
  max_runtime?: number;
}

export interface PDVRPRequest {
  locations: LocationModel[];
  pickup_delivery_pairs: PickupDeliveryPair[];
  depot_index?: number;
  vehicle_capacity: number;
  max_runtime?: number;
}

export interface PCVRPRequest {
  locations: LocationModel[];
  prizes: number[];
  depot_index?: number;
  vehicle_capacity: number;
  min_prize: number;
  max_runtime?: number;
}

export interface VRPVariantInfo {
  name: string;
  description: string;
  constraints: string[];
  objectives: string[];
  complexity: string;
}

// Unified PyVRP API Types
export interface ClientModel {
  x: number;
  y: number;
  delivery: number | number[];
  pickup?: number | number[];
  service_duration?: number;
  tw_early?: number;
  tw_late?: number;
  release_time?: number;
  prize?: number;
  required?: boolean;
}

export interface UnifiedDepotModel {
  x: number;
  y: number;
}

export interface VehicleTypeModel {
  num_available: number;
  capacity: number | number[];
  start_depot: number;
  end_depot?: number;
  fixed_cost?: number;
  tw_early?: number;
  tw_late?: number;
  max_duration?: number;
  max_distance?: number;
}

export interface VRPProblemData {
  clients: ClientModel[];
  depots: UnifiedDepotModel[];
  vehicle_types: VehicleTypeModel[];
  distance_matrix?: number[][];
  duration_matrix?: number[][];
  max_runtime?: number;
}

export interface UnifiedRouteModel {
  vehicle_type: number;
  depot: number;
  clients: number[];
  distance: number;
  duration: number;
  demand_served: number | number[];
}

export interface UnifiedVRPSolution {
  status: string;
  objective_value: number;
  routes: UnifiedRouteModel[];
  computation_time: number;
  solver: string;
}

class ApiService {
  // Analytics endpoints
  static async performABCAnalysis(
    file: File,
    threshold: string = "0.7,0.2,0.1",
    aggCol: string = "prod",
    value: string = "demand",
    abcName: string = "abc",
    rankName: string = "rank"
  ): Promise<ABCAnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);
    formData.append('agg_col', aggCol);
    formData.append('value', value);
    formData.append('abc_name', abcName);
    formData.append('rank_name', rankName);

    const response = await apiClient.post<ABCAnalysisResult>('/analytics/abc-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async generateTreemap(
    file: File,
    parent: string = "cust",
    value: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('parent', parent);
    formData.append('value', value);

    const response = await apiClient.post('/analytics/treemap', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async analyzeRiskPooling(
    file: File,
    aggPeriod: string = "1w"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agg_period', aggPeriod);

    const response = await apiClient.post('/analytics/risk-pooling', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async performParetoAnalysis(
    file: File,
    aggCol: string = "prod",
    value: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agg_col', aggCol);
    formData.append('value', value);

    const response = await apiClient.post('/analytics/pareto-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Inventory endpoints
  static async calculateEOQ(params: {
    K: number;
    d: number;
    h: number;
    b?: number;
    r?: number;
    c?: number;
    theta?: number;
  }): Promise<EOQResult> {
    const formData = new FormData();
    formData.append('K', params.K.toString());
    formData.append('d', params.d.toString());
    formData.append('h', params.h.toString());
    if (params.b !== undefined) formData.append('b', params.b.toString());
    if (params.r !== undefined) formData.append('r', params.r.toString());
    if (params.c !== undefined) formData.append('c', params.c.toString());
    if (params.theta !== undefined) formData.append('theta', params.theta.toString());

    const response = await apiClient.post<EOQResult>('/inventory/eoq', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  static async optimizeMultiEchelon(
    networkData: any,
    demandData: any,
    costParams: any
  ): Promise<any> {
    const payload = {
      network_structure: networkData,
      demand_data: demandData,
      cost_parameters: costParams
    };

    const response = await apiClient.post('/inventory/multi-echelon-json', payload, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  }

  static async performInventoryABC(
    file: File,
    threshold: string = "0.7,0.2,0.1",
    aggCol: string = "item",
    value: string = "annual_usage"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);
    formData.append('agg_col', aggCol);
    formData.append('value', value);

    const response = await apiClient.post('/inventory/inventory-abc', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async simulateInventory(params: {
    n_samples?: number;
    n_periods?: number;
    mu: number;
    sigma: number;
    LT?: number;
    Q: number;
    R: number;
    b?: number;
    h?: number;
    fc?: number;
    S?: number;
  }): Promise<SimulationResult> {
    const formData = new FormData();
    if (params.n_samples) formData.append('n_samples', params.n_samples.toString());
    if (params.n_periods) formData.append('n_periods', params.n_periods.toString());
    formData.append('mu', params.mu.toString());
    formData.append('sigma', params.sigma.toString());
    if (params.LT) formData.append('LT', params.LT.toString());
    formData.append('Q', params.Q.toString());
    formData.append('R', params.R.toString());
    if (params.b) formData.append('b', params.b.toString());
    if (params.h) formData.append('h', params.h.toString());
    if (params.fc) formData.append('fc', params.fc.toString());
    if (params.S) formData.append('S', params.S.toString());

    const response = await apiClient.post<SimulationResult>('/inventory/simulate', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  // Routing endpoints
  static async calculateCO2(
    capacity: number,
    rate: number = 0.5,
    diesel: boolean = false
  ): Promise<CO2Result> {
    const formData = new FormData();
    formData.append('capacity', capacity.toString());
    formData.append('rate', rate.toString());
    formData.append('diesel', diesel.toString());

    const response = await apiClient.post<CO2Result>('/routing/co2-calculation', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  static async generateDistanceMatrix(
    file: File,
    outputFormat: string = "json"
  ): Promise<DistanceMatrixResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('output_format', outputFormat);

    const response = await apiClient.post<DistanceMatrixResult>('/routing/distance-matrix', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async optimizeRoutes(
    file: File,
    vehicleCapacity: number,
    maxRoutes: number = 5,
    depotName: string = "Depot",
    maxRuntime: number = 30
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('vehicle_capacity', vehicleCapacity.toString());
    formData.append('max_routes', maxRoutes.toString());
    formData.append('depot_name', depotName);
    formData.append('max_runtime', maxRuntime.toString());

    const response = await apiClient.post('/routing/route-optimization', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: (maxRuntime + 10) * 1000, // Add 10 second buffer to HTTP timeout
    });
    return response.data;
  }

  static async optimizeAdvancedVRP(
    file: File,
    vehicleCapacity: number,
    maxRoutes: number = 5,
    depotName: string = "Depot",
    maxRuntime: number = 30
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('vehicle_capacity', vehicleCapacity.toString());
    formData.append('max_routes', maxRoutes.toString());
    formData.append('depot_name', depotName);
    formData.append('max_runtime', maxRuntime.toString());

    const response = await apiClient.post('/routing/advanced-vrp', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: (maxRuntime + 10) * 1000, // Add 10 second buffer to HTTP timeout
    });
    return response.data;
  }

  static async createDeliverySchedule(
    file: File,
    workingStart: number = 8,
    workingEnd: number = 18,
    serviceTime: number = 30
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('working_start', workingStart.toString());
    formData.append('working_end', workingEnd.toString());
    formData.append('service_time', serviceTime.toString());

    const response = await apiClient.post('/routing/delivery-schedule', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async analyzeRouteEmissions(
    distanceKm: number,
    capacityKg: number,
    loadingRate: number = 0.7,
    fuelType: string = "gasoline"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('distance_km', distanceKm.toString());
    formData.append('capacity_kg', capacityKg.toString());
    formData.append('loading_rate', loadingRate.toString());
    formData.append('fuel_type', fuelType);

    const response = await apiClient.post('/routing/emissions-analysis', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  static async solveVRPLIB(
    file: File,
    maxRuntime: number = 60,
    maxIterations: number = 10000,
    seed: number = 42
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('max_runtime', maxRuntime.toString());
    formData.append('max_iterations', maxIterations.toString());
    formData.append('seed', seed.toString());

    const response = await apiClient.post('/routing/vrplib-solve', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: (maxRuntime + 20) * 1000, // Add 20 second buffer for VRPLIB instances
    });
    return response.data;
  }

  static async performRankAnalysis(
    file: File,
    aggCol: string = "prod",
    value: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agg_col', aggCol);
    formData.append('value', value);

    const response = await apiClient.post('/analytics/rank-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async performRankAnalysisPeriods(
    file: File,
    aggCol: string = "prod",
    value: string = "demand",
    aggPeriod: string = "1w"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agg_col', aggCol);
    formData.append('value', value);
    formData.append('agg_period', aggPeriod);

    const response = await apiClient.post('/analytics/rank-analysis-periods', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async performMeanCVAnalysis(
    demandFile: File,
    productFile?: File,
    showName: boolean = true
  ): Promise<any> {
    const formData = new FormData();
    formData.append('demand_file', demandFile);
    if (productFile) {
      formData.append('product_file', productFile);
    }
    formData.append('show_name', showName.toString());

    const response = await apiClient.post('/analytics/mean-cv-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async generateTreemapWithABC(
    file: File,
    abcCol: string = "abc"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('abc_col', abcCol);

    const response = await apiClient.post('/analytics/treemap-with-abc', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async performComprehensiveABCAnalysis(
    file: File,
    value: string = "demand",
    cumsum: boolean = true,
    custThres: string = "0.7, 0.2, 0.1",
    prodThres: string = "0.7, 0.2, 0.1"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('value', value);
    formData.append('cumsum', cumsum.toString());
    formData.append('cust_thres', custThres);
    formData.append('prod_thres', prodThres);

    const response = await apiClient.post('/analytics/comprehensive-abc-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async performAdvancedRankAnalysis(
    file: File,
    value: string = "demand",
    aggPeriod: string = "1m",
    topRank: number = 10
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('value', value);
    formData.append('agg_period', aggPeriod);
    formData.append('top_rank', topRank.toString());

    const response = await apiClient.post('/analytics/advanced-rank-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async visualizeInventoryReduction(
    file: File,
    aggPeriod: string = "1w"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('agg_period', aggPeriod);

    const response = await apiClient.post('/analytics/inventory-reduction-visualization', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Logistics Network Design (LND) endpoints
  static async calculateWeiszfeldLocation(
    file: File,
    latCol: string = "lat",
    lonCol: string = "lon", 
    demandCol: string = "demand",
    maxIterations: number = 1000,
    tolerance: number = 1e-6
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('max_iterations', maxIterations.toString());
    formData.append('tolerance', tolerance.toString());

    const response = await apiClient.post('/lnd/weiszfeld-location', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async calculateMultiFacilityWeiszfeld(
    file: File,
    numFacilities: number = 3,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand",
    maxIterations: number = 1000,
    tolerance: number = 1e-4,
    randomState: number = 42
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('max_iterations', maxIterations.toString());
    formData.append('tolerance', tolerance.toString());
    formData.append('random_state', randomState.toString());

    const response = await apiClient.post('/lnd/multi-facility-weiszfeld', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async calculateRepeatedMultiFacilityWeiszfeld(
    file: File,
    numFacilities: number = 3,
    numRuns: number = 10,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand",
    maxIterations: number = 1000,
    tolerance: number = 1e-4,
    baseRandomState: number = 42
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('num_runs', numRuns.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('max_iterations', maxIterations.toString());
    formData.append('tolerance', tolerance.toString());
    formData.append('base_random_state', baseRandomState.toString());

    const response = await apiClient.post('/lnd/repeated-multi-facility-weiszfeld', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async clusterCustomers(
    file: File,
    method: string = "kmeans",
    nClusters: number = 3,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand",
    linkageMethod: string = "ward",
    randomState: number = 42
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', method);
    formData.append('n_clusters', nClusters.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('linkage_method', linkageMethod);
    formData.append('random_state', randomState.toString());

    const response = await apiClient.post('/lnd/customer-clustering', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async solveKMedian(
    customerFile: File,
    k: number = 3,
    candidateMethod: string = "grid",
    nCandidates: number = 20,
    latCol: string = "lat",
    lonCol: string = "lon", 
    demandCol: string = "demand",
    maxIterations: number = 100,
    lambdaStep: number = 0.1
  ): Promise<any> {
    const formData = new FormData();
    formData.append('customer_file', customerFile);
    formData.append('k', k.toString());
    formData.append('candidate_method', candidateMethod);
    formData.append('n_candidates', nCandidates.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('max_iterations', maxIterations.toString());
    formData.append('lambda_step', lambdaStep.toString());

    const response = await apiClient.post('/lnd/k-median-optimization', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async analyzeServiceArea(
    file: File,
    facilityLat: number,
    facilityLon: number,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('facility_lat', facilityLat.toString());
    formData.append('facility_lon', facilityLon.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);

    const response = await apiClient.post('/lnd/service-area-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async generateFacilityCandidates(
    file: File,
    method: string = "grid",
    nCandidates: number = 20,
    latCol: string = "lat",
    lonCol: string = "lon"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', method);
    formData.append('n_candidates', nCandidates.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);

    const response = await apiClient.post('/lnd/generate-candidates', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async calculateDistance(
    originLat: number,
    originLon: number,
    destLat: number,
    destLon: number
  ): Promise<any> {
    const formData = new FormData();
    formData.append('origin_lat', originLat.toString());
    formData.append('origin_lon', originLon.toString());
    formData.append('dest_lat', destLat.toString());
    formData.append('dest_lon', destLon.toString());

    const response = await apiClient.post('/lnd/distance-calculation', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }


  static async solveMultipleSourceLND(
    customerFile: File,
    warehouseFile: File,
    factoryFile: File,
    productFile: File,
    demandFile: File,
    factoryCapacityFile: File,
    transportationCost: number = 1.0,
    deliveryCost: number = 2.0,
    warehouseFixedCost: number = 10000.0,
    warehouseVariableCost: number = 1.0,
    numWarehouses?: number,
    singleSourcing: boolean = false,
    maxRuntime: number = 300
  ): Promise<any> {
    const formData = new FormData();
    formData.append('customer_file', customerFile);
    formData.append('warehouse_file', warehouseFile);
    formData.append('factory_file', factoryFile);
    formData.append('product_file', productFile);
    formData.append('demand_file', demandFile);
    formData.append('factory_capacity_file', factoryCapacityFile);
    formData.append('transportation_cost', transportationCost.toString());
    formData.append('delivery_cost', deliveryCost.toString());
    formData.append('warehouse_fixed_cost', warehouseFixedCost.toString());
    formData.append('warehouse_variable_cost', warehouseVariableCost.toString());
    if (numWarehouses !== undefined) {
      formData.append('num_warehouses', numWarehouses.toString());
    }
    formData.append('single_sourcing', singleSourcing.toString());
    formData.append('max_runtime', maxRuntime.toString());

    const response = await apiClient.post('/lnd/multiple-source-lnd', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: (maxRuntime + 30) * 1000, // Add 30 second buffer to HTTP timeout
    });
    return response.data;
  }

  static async solveSingleSourceLND(
    customerFile: File,
    warehouseFile: File,
    factoryFile: File,
    productFile: File,
    demandFile: File,
    factoryCapacityFile: File,
    transportationCost: number = 1.0,
    deliveryCost: number = 2.0,
    warehouseFixedCost: number = 10000.0,
    warehouseVariableCost: number = 1.0,
    numWarehouses?: number,
    maxRuntime: number = 300
  ): Promise<any> {
    const formData = new FormData();
    formData.append('customer_file', customerFile);
    formData.append('warehouse_file', warehouseFile);
    formData.append('factory_file', factoryFile);
    formData.append('product_file', productFile);
    formData.append('demand_file', demandFile);
    formData.append('factory_capacity_file', factoryCapacityFile);
    formData.append('transportation_cost', transportationCost.toString());
    formData.append('delivery_cost', deliveryCost.toString());
    formData.append('warehouse_fixed_cost', warehouseFixedCost.toString());
    formData.append('warehouse_variable_cost', warehouseVariableCost.toString());
    if (numWarehouses !== undefined) {
      formData.append('num_warehouses', numWarehouses.toString());
    }
    formData.append('max_runtime', maxRuntime.toString());

    const response = await apiClient.post('/lnd/single-source-lnd', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: (maxRuntime + 30) * 1000, // Add 30 second buffer to HTTP timeout
    });
    return response.data;
  }

  static async performElbowMethodAnalysis(
    file: File,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol?: string,
    minFacilities: number = 1,
    maxFacilities: number = 10,
    algorithm: string = "weiszfeld",
    maxIterations: number = 1000,
    tolerance: number = 1e-4,
    randomState: number = 42
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    if (demandCol) {
      formData.append('demand_col', demandCol);
    }
    formData.append('min_facilities', minFacilities.toString());
    formData.append('max_facilities', maxFacilities.toString());
    formData.append('algorithm', algorithm);
    formData.append('max_iterations', maxIterations.toString());
    formData.append('tolerance', tolerance.toString());
    formData.append('random_state', randomState.toString());

    const response = await apiClient.post('/lnd/elbow-method-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes timeout for elbow method analysis
    });
    return response.data;
  }

  static async calculateHierarchicalClusteringAdvanced(
    file: File,
    numFacilities: number = 2,
    linkage: string = "average",
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('linkage', linkage);
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);

    const response = await apiClient.post('/lnd/hierarchical-clustering-advanced', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async solveKMedianAdvanced(
    file: File,
    numFacilities: number = 2,
    maxIter: number = 100,
    maxLr: number = 0.01,
    momLow: number = 0.85,
    momHigh: number = 0.95,
    convergence: number = 1e-5,
    adam: boolean = false,
    capacity?: number,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('max_iter', maxIter.toString());
    formData.append('max_lr', maxLr.toString());
    formData.append('mom_low', momLow.toString());
    formData.append('mom_high', momHigh.toString());
    formData.append('convergence', convergence.toString());
    formData.append('adam', adam.toString());
    if (capacity !== undefined) {
      formData.append('capacity', capacity.toString());
    }
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);

    const response = await apiClient.post('/lnd/k-median-advanced', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async solveKMedianLrFinderAdvanced(
    file: File,
    numFacilities: number = 2,
    maxIter: number = 100,
    momLow: number = 0.85,
    momHigh: number = 0.95,
    capacity?: number,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('max_iter', maxIter.toString());
    formData.append('mom_low', momLow.toString());
    formData.append('mom_high', momHigh.toString());
    if (capacity !== undefined) {
      formData.append('capacity', capacity.toString());
    }
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);

    const response = await apiClient.post('/lnd/k-median-lr-finder-advanced', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Excel Integration endpoints
  static async generateExcelTemplate(): Promise<Blob> {
    const response = await apiClient.get('/lnd/excel-template', {
      responseType: 'blob'
    });
    return response.data;
  }

  static async parseExcelData(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/lnd/excel-parse', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Customer Aggregation endpoints
  static async performCustomerAggregation(
    customerFile: File,
    productFile: File,
    demandFile: File,
    numFacilities: number = 3,
    linkage: string = "complete",
    toll: boolean = true,
    osrmHost: string = "localhost",
    latCol: string = "lat",
    lonCol: string = "lon",
    nameCol: string = "name"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('customer_file', customerFile);
    formData.append('product_file', productFile);
    formData.append('demand_file', demandFile);
    formData.append('num_facilities', numFacilities.toString());
    formData.append('linkage', linkage);
    formData.append('toll', toll.toString());
    formData.append('osrm_host', osrmHost);
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('name_col', nameCol);

    const response = await apiClient.post('/lnd/customer-aggregation', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 180000, // 3 minutes timeout for aggregation
    });
    return response.data;
  }

  // Service Information endpoint
  static async getServiceInfo(): Promise<any> {
    const response = await apiClient.get('/lnd/service-info');
    return response.data;
  }

  // Carbon Footprint Analysis endpoints
  static async performCarbonFootprintAnalysis(
    facilityFile: File,
    customerFile: File,
    transportationCostPerKm: number = 1.0,
    carbonConstraintKg?: number,
    carbonPricePerKg: number = 0.0,
    latCol: string = "lat",
    lonCol: string = "lon",
    demandCol: string = "demand",
    costCol: string = "fixed_cost",
    capacityCol: string = "vehicle_capacity",
    loadingRateCol: string = "loading_rate",
    fuelTypeCol: string = "fuel_type"
  ): Promise<any> {
    const formData = new FormData();
    formData.append('facility_file', facilityFile);
    formData.append('customer_file', customerFile);
    formData.append('transportation_cost_per_km', transportationCostPerKm.toString());
    if (carbonConstraintKg !== undefined) {
      formData.append('carbon_constraint_kg', carbonConstraintKg.toString());
    }
    formData.append('carbon_price_per_kg', carbonPricePerKg.toString());
    formData.append('lat_col', latCol);
    formData.append('lon_col', lonCol);
    formData.append('demand_col', demandCol);
    formData.append('cost_col', costCol);
    formData.append('capacity_col', capacityCol);
    formData.append('loading_rate_col', loadingRateCol);
    formData.append('fuel_type_col', fuelTypeCol);

    const response = await apiClient.post('/lnd/carbon-footprint-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 240000, // 4 minutes timeout for carbon analysis
    });
    return response.data;
  }

  static async calculateCO2Emissions(
    capacityKg: number,
    loadingRate: number = 0.7,
    fuelType: string = "diesel",
    distanceKm: number,
    weightTons: number
  ): Promise<any> {
    const formData = new FormData();
    formData.append('capacity_kg', capacityKg.toString());
    formData.append('loading_rate', loadingRate.toString());
    formData.append('fuel_type', fuelType);
    formData.append('distance_km', distanceKm.toString());
    formData.append('weight_tons', weightTons.toString());

    const response = await apiClient.post('/lnd/co2-emission-calculation', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  static async getCarbonEmissionFactors(): Promise<any> {
    const response = await apiClient.get('/lnd/carbon-emission-factors');
    return response.data;
  }

  // Sample Data endpoints
  static async getSampleDataInfo(): Promise<any> {
    const response = await apiClient.get('/lnd/sample-data-info');
    return response.data;
  }

  static async downloadSampleData(datasetType: string): Promise<Blob> {
    const response = await apiClient.get(`/lnd/sample-data/${datasetType}`, {
      responseType: 'blob'
    });
    return response.data;
  }

  // PyVRP API endpoints
  static async solveCVRP(request: CVRPRequest): Promise<VRPSolution> {
    const response = await apiClient.post<VRPSolution>('/pyvrp/solve/cvrp', request);
    return response.data;
  }

  static async solveVRPTW(request: VRPTWRequest): Promise<VRPSolution> {
    const response = await apiClient.post<VRPSolution>('/pyvrp/solve/vrptw', request);
    return response.data;
  }

  static async solveMDVRP(request: MDVRPRequest): Promise<VRPSolution> {
    const response = await apiClient.post<VRPSolution>('/pyvrp/solve/mdvrp', request);
    return response.data;
  }

  static async solvePDVRP(request: PDVRPRequest): Promise<VRPSolution> {
    const response = await apiClient.post<VRPSolution>('/pyvrp/solve/pdvrp', request);
    return response.data;
  }

  static async solvePCVRP(request: PCVRPRequest): Promise<VRPSolution> {
    const response = await apiClient.post<VRPSolution>('/pyvrp/solve/pcvrp', request);
    return response.data;
  }

  static async getVRPVariants(): Promise<{ variants: { [key: string]: VRPVariantInfo } }> {
    const response = await apiClient.get('/pyvrp/variants');
    return response.data;
  }

  static async uploadVRPData(file: File, dataType: string): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('data_type', dataType);

    const response = await apiClient.post('/pyvrp/upload/csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async compareVRPSolutions(solutions: VRPSolution[]): Promise<any> {
    const response = await apiClient.post('/pyvrp/compare', solutions);
    return response.data;
  }

  // Unified PyVRP API
  static async solveUnifiedVRP(request: VRPProblemData): Promise<UnifiedVRPSolution> {
    const response = await apiClient.post<UnifiedVRPSolution>('/pyvrp/solve', request);
    return response.data;
  }
}

export { ApiService };
export default apiClient;