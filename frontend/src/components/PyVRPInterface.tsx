import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  CloudUpload as UploadIcon,
  LocalShipping as TruckIcon,
  LocationOn as LocationIcon,
  Schedule as TimeIcon,
  ExpandMore as ExpandMoreIcon,
  Store as DepotIcon,
} from '@mui/icons-material';
import VRPMapVisualization from './VRPMapVisualization';
import PyVRPScheduleViewer from './PyVRPScheduleViewer';
import PyVRPDataEditor from './PyVRPDataEditor';
import SolverConfigPanel from './SolverConfigPanel';
import AdvancedVehicleConfigPanel from './AdvancedVehicleConfigPanel';
import ClientConstraintsPanel from './ClientConstraintsPanel';
import DataManagementPanel from './DataManagementPanel';
import VRPGanttChart from './VRPGanttChart';

// Data structures for PyVRP
interface Client {
  x: number;
  y: number;
  delivery: number;
  pickup?: number;
  service_duration?: number;
  tw_early?: number;
  tw_late?: number;
  required?: boolean;
  prize?: number;
}

interface Depot {
  x: number;
  y: number;
  tw_early?: number;
  tw_late?: number;
}

interface VehicleType {
  num_available: number;
  capacity: number;
  start_depot: number;
  end_depot?: number;
  fixed_cost?: number;
  tw_early?: number;
  tw_late?: number;
  max_duration?: number;
  max_distance?: number;
}

interface ProblemData {
  clients: Client[];
  depots: Depot[];
  vehicle_types: VehicleType[];
}

// Solver configuration interface
interface SolverConfig {
  time_limit: number;
  population_size: number;
  random_seed?: number;
  crossover_probability: number;
  mutation_probability: number;
  repair_probability: number;
  max_iterations_without_improvement: number;
  target_gap: number;
  penalty_scaling: number;
  penalty_start: number;
  enable_local_search: boolean;
  enable_diversity_management: boolean;
  fleet_minimization: boolean;
}

// Advanced vehicle configuration interfaces
interface CapacityConstraint {
  id: string;
  name: string;
  unit: string;
  value: number;
}

interface VehicleSkill {
  id: string;
  name: string;
  description: string;
}

interface AdvancedVehicleType {
  id: string;
  name: string;
  num_available: number;
  start_depot: number;
  end_depot?: number;
  capacities: CapacityConstraint[];
  max_distance?: number;
  max_duration?: number;
  max_working_time?: number;
  fixed_cost: number;
  distance_cost_per_km: number;
  time_cost_per_hour: number;
  overtime_cost_multiplier: number;
  skills: string[];
  required_skills: string[];
  tw_early: number;
  tw_late: number;
  break_duration?: number;
  break_earliest?: number;
  break_latest?: number;
  speed_factor: number;
  priority: number;
  fuel_type: 'gasoline' | 'diesel' | 'electric' | 'hybrid';
  emissions_factor: number;
}

interface PyVRPSolution {
  status: string;
  objective_value: number;
  routes: any[];
  computation_time: number;
  solver: string;
  is_feasible: boolean;
  problem_type: string;
  problem_size: any;
}

// Generate random coordinates within Kanto region
const generateKantoCoordinates = () => {
  // Kanto region approximate bounds
  const minLat = 35.0; // Southern boundary
  const maxLat = 36.5; // Northern boundary
  const minLon = 138.5; // Western boundary
  const maxLon = 140.5; // Eastern boundary
  
  const lat = minLat + Math.random() * (maxLat - minLat);
  const lon = minLon + Math.random() * (maxLon - minLon);
  
  // Return actual lat/lon coordinates
  return {
    x: Math.round(lon * 10000) / 10000, // Round to 4 decimal places
    y: Math.round(lat * 10000) / 10000  // Round to 4 decimal places
  };
};

// Predefined problem examples with Kanto region coordinates
const EXAMPLE_PROBLEMS = {
  cvrp: {
    name: "CVRP - Capacitated VRP",
    description: "Basic capacitated vehicle routing problem",
    data: {
      clients: [
        { x: 139.4000, y: 35.7000, delivery: 5, service_duration: 10 }, // Tokyo area
        { x: 139.6000, y: 35.6500, delivery: 7, service_duration: 12 }, // Yokohama area
        { x: 139.3000, y: 35.8000, delivery: 4, service_duration: 8 },  // Saitama area
        { x: 139.5000, y: 35.5000, delivery: 6, service_duration: 15 }, // Chiba area
        { x: 139.2000, y: 35.9000, delivery: 5, service_duration: 11 }  // Gunma area
      ],
      depots: [
        { x: 139.4500, y: 35.7500 } // Central Tokyo
      ],
      vehicle_types: [
        {
          num_available: 2,
          capacity: 100,
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 100,
          tw_early: 480,  // 8:00 AM
          tw_late: 1080   // 6:00 PM
        }
      ]
    }
  },
  vrptw: {
    name: "VRPTW - VRP with Time Windows",
    description: "VRP with time window constraints",
    data: {
      clients: [
        { x: 139.4000, y: 35.7000, delivery: 3, service_duration: 10, tw_early: 480, tw_late: 720 }, // 8:00-12:00 Tokyo (広い時間窓)
        { x: 139.6000, y: 35.6000, delivery: 4, service_duration: 15, tw_early: 540, tw_late: 780 }, // 9:00-13:00 Yokohama (広い時間窓)
        { x: 139.0000, y: 35.8000, delivery: 2, service_duration: 10, tw_early: 480, tw_late: 660 }, // 8:00-11:00 Saitama (広い時間窓)
      ],
      depots: [
        { x: 139.4500, y: 35.7500, tw_early: 480, tw_late: 1080 } // Central Tokyo
      ],
      vehicle_types: [
        {
          num_available: 3,
          capacity: 20,
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 50,
          tw_early: 480,
          tw_late: 1080
        }
      ]
    }
  },
  mdvrp: {
    name: "MDVRP - Multi-Depot VRP",
    description: "VRP with multiple depots",
    data: {
      clients: [
        { x: 139.4000, y: 35.7000, delivery: 4, service_duration: 10 }, // Tokyo
        { x: 139.6000, y: 35.6000, delivery: 6, service_duration: 15 }, // Yokohama
        { x: 139.3000, y: 35.8000, delivery: 3, service_duration: 8 },  // Saitama
        { x: 139.8000, y: 35.5000, delivery: 7, service_duration: 12 }, // Chiba
        { x: 139.2000, y: 35.9000, delivery: 4, service_duration: 10 }  // Gunma
      ],
      depots: [
        { x: 139.3500, y: 35.8000 }, // Depot 0 - Saitama
        { x: 139.7000, y: 35.5500 }  // Depot 1 - Chiba
      ],
      vehicle_types: [
        {
          num_available: 1,
          capacity: 40,
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 40,
          tw_early: 480,  // 8:00 AM
          tw_late: 1080   // 6:00 PM
        },
        {
          num_available: 1,
          capacity: 40,
          start_depot: 1,
          end_depot: 1,
          fixed_cost: 40,
          tw_early: 480,  // 8:00 AM
          tw_late: 1080   // 6:00 PM
        }
      ]
    }
  },
  pdvrp: {
    name: "PDVRP - Pickup and Delivery VRP",
    description: "VRP with pickup and delivery constraints",
    data: {
      clients: [
        { x: 139.4000, y: 35.7000, delivery: 5, pickup: 0, service_duration: 12 },    // Delivery only - Tokyo
        { x: 139.6000, y: 35.6000, delivery: 0, pickup: 7, service_duration: 10 },    // Pickup only - Yokohama
        { x: 139.3000, y: 35.8000, delivery: 4, pickup: 2, service_duration: 15 },     // Both - Saitama
        { x: 139.8000, y: 35.5000, delivery: 0, pickup: 6, service_duration: 8 }      // Pickup only - Chiba
      ],
      depots: [
        { x: 139.4500, y: 35.7500 } // Central Tokyo
      ],
      vehicle_types: [
        {
          num_available: 2,
          capacity: 50,
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 30,
          tw_early: 480,  // 8:00 AM
          tw_late: 1080   // 6:00 PM
        }
      ]
    }
  },
  pcvrp: {
    name: "PC-VRP - Prize Collecting VRP",
    description: "VRP with optional clients and prizes",
    data: {
      clients: [
        { x: 139.4000, y: 35.7000, delivery: 5, service_duration: 12, required: true, prize: 0 },   // Tokyo (required)
        { x: 139.6000, y: 35.6000, delivery: 4, service_duration: 10, required: false, prize: 50 }, // Yokohama (optional)
        { x: 139.3000, y: 35.8000, delivery: 7, service_duration: 15, required: false, prize: 80 }, // Saitama (optional)
        { x: 139.8000, y: 35.5000, delivery: 3, service_duration: 8, required: false, prize: 30 },  // Chiba (optional)
        { x: 139.2000, y: 35.9000, delivery: 6, service_duration: 12, required: false, prize: 60 }  // Gunma (optional)
      ],
      depots: [
        { x: 139.4500, y: 35.7500 } // Central Tokyo
      ],
      vehicle_types: [
        {
          num_available: 1,
          capacity: 40,
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 60,
          tw_early: 480,  // 8:00 AM
          tw_late: 1080   // 6:00 PM
        }
      ]
    }
  }
};

const PyVRPInterface: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<keyof typeof EXAMPLE_PROBLEMS>('cvrp');
  // Convert initial coordinates to integers for API compatibility
  const getConvertedExampleData = (exampleKey: keyof typeof EXAMPLE_PROBLEMS) => {
    const exampleData = EXAMPLE_PROBLEMS[exampleKey].data;
    return {
      clients: exampleData.clients.map((client: any) => ({
        ...client,
        x: Math.round(client.x * 10000), // Convert lat/lon to scaled integer
        y: Math.round(client.y * 10000), // Convert lat/lon to scaled integer
        delivery: Math.round(client.delivery),
        service_duration: Math.round(client.service_duration || 10),
        tw_early: Math.round(client.tw_early || 0),
        tw_late: Math.round(client.tw_late || 1440)
      })),
      depots: exampleData.depots.map((depot: any) => ({
        ...depot,
        x: Math.round(depot.x * 10000), // Convert lat/lon to scaled integer
        y: Math.round(depot.y * 10000), // Convert lat/lon to scaled integer
        tw_early: Math.round(depot.tw_early || 0),
        tw_late: Math.round(depot.tw_late || 1440)
      })),
      vehicle_types: exampleData.vehicle_types.map((vt: any) => ({
        ...vt,
        num_available: Math.round(vt.num_available),
        capacity: Math.round(vt.capacity),
        start_depot: Math.round(vt.start_depot),
        end_depot: vt.end_depot !== undefined ? Math.round(vt.end_depot) : Math.round(vt.start_depot),
        fixed_cost: Math.round(vt.fixed_cost || 0),
        tw_early: Math.round(vt.tw_early || 0),
        tw_late: Math.round(vt.tw_late || 1440),
        max_duration: vt.max_duration ? Math.round(vt.max_duration) : undefined,
        max_distance: vt.max_distance ? Math.round(vt.max_distance) : undefined
      }))
    };
  };
  const [problemData, setProblemData] = useState<ProblemData>(getConvertedExampleData('cvrp'));
  const [solution, setSolution] = useState<PyVRPSolution | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [mainTabValue, setMainTabValue] = useState(0);
  const [showDataEditor, setShowDataEditor] = useState(false);
  
  // Solver configuration state
  const [solverConfig, setSolverConfig] = useState<SolverConfig>({
    time_limit: 60,
    population_size: 25,
    random_seed: undefined,
    crossover_probability: 0.95,
    mutation_probability: 0.02,
    repair_probability: 0.5,
    max_iterations_without_improvement: 1000,
    target_gap: 1.0,
    penalty_scaling: 1.0,
    penalty_start: 100,
    enable_local_search: true,
    enable_diversity_management: true,
    fleet_minimization: false
  });
  
  // Advanced vehicle types state
  const [advancedVehicleTypes, setAdvancedVehicleTypes] = useState<AdvancedVehicleType[]>([]);
  const [availableSkills, setAvailableSkills] = useState<VehicleSkill[]>([
    { id: 'refrigerated', name: '冷蔵', description: '冷蔵商品の配送' },
    { id: 'heavy_duty', name: '重量', description: '重量物の配送' },
    { id: 'fragile', name: '壊れ物', description: '壊れやすい商品の配送' },
    { id: 'hazmat', name: '危険物', description: '危険物の配送' }
  ]);
  
  // Configuration panel visibility
  const [showSolverConfig, setShowSolverConfig] = useState(false);
  const [showAdvancedVehicles, setShowAdvancedVehicles] = useState(false);
  const [showClientConstraints, setShowClientConstraints] = useState(false);
  const [showDataManagement, setShowDataManagement] = useState(false);
  
  // Advanced clients state
  const [advancedClients, setAdvancedClients] = useState<any[]>([]);
  
  // Success/Error message handlers for DataManagement
  const handleDataSuccess = (message: string) => {
    setError(null);
    // You could add a success state here if needed
    console.log('Success:', message);
  };
  
  const handleDataError = (message: string) => {
    setError(message);
  };

  // Format time for display (minutes to HH:MM)
  const formatTime = (minutes: number): string => {
    // Handle invalid or negative values
    if (!minutes || isNaN(minutes) || minutes < 0) {
      return '00:00';
    }
    
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  // Load example problem
  const loadExample = (exampleKey: keyof typeof EXAMPLE_PROBLEMS) => {
    setSelectedExample(exampleKey);
    setProblemData(getConvertedExampleData(exampleKey));
    setSolution(null);
    setError(null);
  };

  // Convert advanced vehicle types to PyVRP format
  const convertToBasicVehicleTypes = (): VehicleType[] => {
    if (advancedVehicleTypes.length === 0) {
      return problemData.vehicle_types;
    }
    
    return advancedVehicleTypes.map(avt => ({
      num_available: avt.num_available,
      capacity: avt.capacities.find(c => c.name === '重量')?.value || avt.capacities[0]?.value || 100,
      start_depot: avt.start_depot,
      end_depot: avt.end_depot,
      fixed_cost: avt.fixed_cost,
      tw_early: avt.tw_early,
      tw_late: avt.tw_late,
      max_duration: avt.max_duration,
      max_distance: avt.max_distance
    }));
  };
  
  // Convert advanced clients to basic PyVRP format and merge with existing clients
  const mergeClients = (): Client[] => {
    // Start with basic clients from sample data
    const basicClients = [...problemData.clients];
    
    // Convert and add advanced clients
    const convertedAdvancedClients = advancedClients.map(ac => ({
      x: Math.round(ac.x * 10000),  // Convert lat/lon to integer (scaled by 10000)
      y: Math.round(ac.y * 10000),  // Convert lat/lon to integer (scaled by 10000)
      delivery: Math.round(ac.delivery || 0),  // Ensure integer delivery
      pickup: ac.pickup ? Math.round(ac.pickup) : undefined,
      service_duration: Math.round(ac.service_details?.base_time || 10),
      // Use the first time window if available, otherwise use defaults
      tw_early: Math.round(ac.time_windows?.[0]?.start || 480),
      tw_late: Math.round(ac.time_windows?.[0]?.end || 1080),
      required: !ac.required_sequence_before || ac.required_sequence_before.length === 0,
      prize: ac.priority === 'high' ? 100 : ac.priority === 'medium' ? 50 : 0
    }));
    
    // Combine both arrays
    return [...basicClients, ...convertedAdvancedClients];
  };
  
  // Create a mapping of client indices to names
  const getClientNames = (): string[] => {
    const names = problemData.clients.map((_, index) => `Client ${index}`);
    const advancedNames = advancedClients.map(ac => ac.name);
    return [...names, ...advancedNames];
  };
  
  // Get client name by index
  const getClientName = (index: number): string => {
    const names = getClientNames();
    return names[index] || `Client ${index}`;
  };

  // Solve the VRP problem
  const solveProblem = async () => {
    // Validate data before solving
    const allClients = mergeClients();
    const allVehicleTypes = convertToBasicVehicleTypes();
    
    if (allClients.length === 0) {
      setError('顧客データが必要です。問題例を選択するか、顧客データをインポートしてください。');
      return;
    }
    
    if (allVehicleTypes.length === 0) {
      setError('車両データが必要です。車両タイプを設定してください。');
      return;
    }
    
    if (problemData.depots.length === 0) {
      setError('デポデータが必要です。デポを設定してください。');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSolution(null);

    try {
      // Prepare problem data with merged clients and advanced vehicle types
      const finalProblemData = {
        ...problemData,
        clients: mergeClients(),
        vehicle_types: convertToBasicVehicleTypes(),
        depots: problemData.depots.map(depot => ({
          ...depot,
          x: Number.isInteger(depot.x) ? depot.x : Math.round(depot.x * 10000),
          y: Number.isInteger(depot.y) ? depot.y : Math.round(depot.y * 10000)
        })),
        solver_config: solverConfig
      };

      const response = await fetch('/api/pyvrp/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(finalProblemData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('PyVRP Solve Result:', result);
      console.log('Routes in result:', result.routes);
      if (result.routes && result.routes.length > 0) {
        result.routes.forEach((route: any, index: number) => {
          console.log(`Route ${index}:`, route);
          console.log(`  Clients: [${route.clients?.join(', ') || 'none'}]`);
          console.log(`  Start time: ${route.start_time}, End time: ${route.end_time}`);
          console.log(`  Distance: ${route.distance}, Duration: ${route.duration}`);
          if (route.arrival_times) console.log(`  Arrival times: [${route.arrival_times.join(', ')}]`);
          if (route.departure_times) console.log(`  Departure times: [${route.departure_times.join(', ')}]`);
        });
      }
      setSolution(result);
    } catch (err: any) {
      setError(`Solving failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Render route details table
  const renderRouteDetails = () => {
    if (!solution?.routes) return null;

    return (
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Route</TableCell>
              <TableCell>Vehicle</TableCell>
              <TableCell>Clients</TableCell>
              <TableCell>Distance</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>Load</TableCell>
              <TableCell>Cost</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {solution.routes.map((route: any, index: number) => (
              <TableRow key={index}>
                <TableCell>
                  <Chip
                    icon={<TruckIcon />}
                    label={`Route ${index + 1}`}
                    color="primary"
                    size="small"
                  />
                </TableCell>
                <TableCell>Type {route.vehicle_type || 0}</TableCell>
                <TableCell>
                  {route.clients?.length || 0} clients
                  {route.clients && (
                    <Typography variant="caption" display="block">
                      [{route.clients.map((clientIndex: number) => getClientName(clientIndex)).join(', ')}]
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  {route.distance ? `${(route.distance / 1000).toFixed(1)} km` : 'N/A'}
                </TableCell>
                <TableCell>
                  {route.duration ? formatTime(route.duration) : 'N/A'}
                  {route.start_time && route.end_time && (
                    <Typography variant="caption" display="block">
                      {formatTime(route.start_time)} - {formatTime(route.end_time)}
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  {route.demand_served || 0}/{route.capacity_utilization ? 
                    Math.round(route.demand_served / route.capacity_utilization) : 'N/A'}
                </TableCell>
                <TableCell>{route.total_cost?.toFixed(2) || 'N/A'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  // Render detailed schedule
  const renderDetailedSchedule = () => {
    if (!solution?.routes) return null;

    return solution.routes.map((route: any, routeIndex: number) => (
      <Accordion key={routeIndex} sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            Route {routeIndex + 1} - Detailed Schedule
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer component={Paper}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Activity</TableCell>
                  <TableCell>Location</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Details</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {/* Departure from depot */}
                <TableRow>
                  <TableCell>
                    {formatTime(route.start_time || 480)}
                  </TableCell>
                  <TableCell>
                    <Chip
                      icon={<DepotIcon />}
                      label="Departure"
                      size="small"
                      color="success"
                    />
                  </TableCell>
                  <TableCell>Depot {route.start_depot || 0}</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>Start from depot</TableCell>
                </TableRow>

                {/* Client visits */}
                {route.clients?.map((clientIndex: number, visitIndex: number) => {
                  const client = problemData.clients[clientIndex];
                  if (!client) return null;

                  const arrivalTime = route.arrival_times?.[visitIndex] || 
                    (route.start_time || 480) + visitIndex * 30;
                  const departureTime = route.departure_times?.[visitIndex] || 
                    arrivalTime + (client.service_duration || 10);

                  return (
                    <React.Fragment key={visitIndex}>
                      {/* Travel to client */}
                      {visitIndex > 0 && (
                        <TableRow>
                          <TableCell>
                            {formatTime(arrivalTime - 15)} {/* Estimated travel */}
                          </TableCell>
                          <TableCell>
                            <Chip
                              icon={<TruckIcon />}
                              label="Travel"
                              size="small"
                              color="warning"
                            />
                          </TableCell>
                          <TableCell>
                            {getClientName(route.clients[visitIndex - 1])} → {getClientName(clientIndex)}
                          </TableCell>
                          <TableCell>15 min</TableCell>
                          <TableCell>Travel between clients</TableCell>
                        </TableRow>
                      )}

                      {/* Service at client */}
                      <TableRow>
                        <TableCell>
                          {formatTime(arrivalTime)}
                        </TableCell>
                        <TableCell>
                          <Chip
                            icon={<LocationIcon />}
                            label="Service"
                            size="small"
                            color="info"
                          />
                        </TableCell>
                        <TableCell>{getClientName(clientIndex)}</TableCell>
                        <TableCell>{client.service_duration || 10} min</TableCell>
                        <TableCell>
                          Delivery: {client.delivery}
                          {client.pickup !== undefined && client.pickup > 0 ? `, Pickup: ${client.pickup}` : ''}
                          {client.prize !== undefined && client.prize > 0 ? `, Prize: ${client.prize}` : ''}
                        </TableCell>
                      </TableRow>
                    </React.Fragment>
                  );
                })}

                {/* Return to depot */}
                <TableRow>
                  <TableCell>
                    {formatTime(route.end_time || ((route.start_time || 480) + (route.duration || 180)))}
                  </TableCell>
                  <TableCell>
                    <Chip
                      icon={<DepotIcon />}
                      label="Return"
                      size="small"
                      color="error"
                    />
                  </TableCell>
                  <TableCell>Depot {route.end_depot || route.start_depot || 0}</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>Return to depot</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>
    ));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        PyVRP - 車両配送最適化
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={mainTabValue} onChange={(_, newValue) => setMainTabValue(newValue)}>
          <Tab label="問題設定" icon={<LocationIcon />} iconPosition="start" />
          <Tab label="データ管理" icon={<UploadIcon />} iconPosition="start" />
          <Tab label="車両設定" icon={<TruckIcon />} iconPosition="start" />
          <Tab label="制約設定" icon={<TimeIcon />} iconPosition="start" />
          <Tab label="ソルバー設定" icon={<PlayIcon />} iconPosition="start" />
          <Tab label="結果分析" icon={<DepotIcon />} iconPosition="start" />
        </Tabs>
      </Box>

      {/* Tab 0: 問題設定 */}
      {mainTabValue === 0 && (
        <Grid container spacing={3}>
          {/* Left Panel - Problem Setup */}
          <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Problem Examples
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Select Example</InputLabel>
                <Select
                  value={selectedExample}
                  label="Select Example"
                  onChange={(e) => loadExample(e.target.value as keyof typeof EXAMPLE_PROBLEMS)}
                >
                  {Object.entries(EXAMPLE_PROBLEMS).map(([key, problem]) => (
                    <MenuItem key={key} value={key}>
                      {problem.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                {EXAMPLE_PROBLEMS[selectedExample].description}
              </Typography>

              {/* Problem Statistics */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Problem Size:</Typography>
                <Typography variant="body2" color={mergeClients().length === 0 ? 'error' : 'textPrimary'}>
                  • Clients: {problemData.clients.length}
                  {advancedClients.length > 0 && (
                    <span style={{ color: 'green' }}> + {advancedClients.length} (拡張)</span>
                  )}
                  {mergeClients().length === 0 && ' ⚠️'}
                </Typography>
                <Typography variant="body2" color={problemData.depots.length === 0 ? 'error' : 'textPrimary'}>
                  • Depots: {problemData.depots.length}
                  {problemData.depots.length === 0 && ' ⚠️'}
                </Typography>
                <Typography variant="body2" color={convertToBasicVehicleTypes().length === 0 ? 'error' : 'textPrimary'}>
                  • Vehicles: {advancedVehicleTypes.length > 0 
                    ? advancedVehicleTypes.reduce((sum, vt) => sum + vt.num_available, 0)
                    : problemData.vehicle_types.reduce((sum, vt) => sum + vt.num_available, 0)}
                  {advancedVehicleTypes.length > 0 && (
                    <span style={{ color: 'green' }}> (拡張)</span>
                  )}
                  {convertToBasicVehicleTypes().length === 0 && ' ⚠️'}
                </Typography>
                {(mergeClients().length === 0 || problemData.depots.length === 0 || convertToBasicVehicleTypes().length === 0) && (
                  <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.5 }}>
                    ⚠️ データが不足しています
                  </Typography>
                )}
              </Box>

              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={solveProblem}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
                sx={{ mb: 1 }}
              >
                {loading ? 'Solving...' : 'Solve Problem'}
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={() => setShowDataEditor(!showDataEditor)}
                startIcon={<UploadIcon />}
                sx={{ mb: 1 }}
              >
                {showDataEditor ? 'Hide Editor' : 'Edit Problem Data'}
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={() => setShowSolverConfig(!showSolverConfig)}
                sx={{ mb: 1 }}
              >
                {showSolverConfig ? 'Hide Solver Config' : 'Solver Settings'}
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={() => setShowAdvancedVehicles(!showAdvancedVehicles)}
                sx={{ mb: 1 }}
              >
                {showAdvancedVehicles ? 'Hide Vehicle Config' : 'Advanced Vehicles'}
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={() => setShowClientConstraints(!showClientConstraints)}
                sx={{ mb: 1 }}
              >
                {showClientConstraints ? 'Hide Client Constraints' : 'Client Constraints'}
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={() => setShowDataManagement(!showDataManagement)}
                sx={{ mb: 2 }}
              >
                {showDataManagement ? 'Hide Data Management' : 'Data Management'}
              </Button>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>

          
          {/* Solver Configuration */}
          {showSolverConfig && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <SolverConfigPanel
                  config={solverConfig}
                  onConfigChange={setSolverConfig}
                />
              </CardContent>
            </Card>
          )}
          
          {/* Advanced Vehicle Configuration */}
          {showAdvancedVehicles && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <AdvancedVehicleConfigPanel
                  vehicleTypes={advancedVehicleTypes}
                  onVehicleTypesChange={setAdvancedVehicleTypes}
                  availableSkills={availableSkills}
                  onSkillsChange={setAvailableSkills}
                />
              </CardContent>
            </Card>
          )}
          
          {/* Client Constraints Configuration */}
          {showClientConstraints && (
            <Card sx={{ mt: 2 }}>
              <CardContent>
                <ClientConstraintsPanel
                  clients={advancedClients}
                  onClientsChange={setAdvancedClients}
                  availableVehicleTypes={advancedVehicleTypes.map(vt => vt.name)}
                  availableSkills={availableSkills}
                />
              </CardContent>
            </Card>
          )}
          
        </Grid>

        {/* Right Panel - Solution Display, Data Management, or Data Editor */}
        <Grid item xs={12} md={8}>
          {/* Data Management Panel - Full Width */}
          {showDataManagement && (
            <Card>
              <CardContent>
                <DataManagementPanel
                  problemData={problemData}
                  solution={solution}
                  onProblemDataChange={setProblemData}
                  onError={handleDataError}
                  onSuccess={handleDataSuccess}
                />
              </CardContent>
            </Card>
          )}
          
          {/* Data Editor Panel - Full Width */}
          {showDataEditor && (
            <Card>
              <CardContent>
                <PyVRPDataEditor
                  problemData={problemData}
                  onDataChange={setProblemData}
                  onSave={() => {
                    setSolution(null);
                    setError(null);
                    setShowDataEditor(false);
                  }}
                />
              </CardContent>
            </Card>
          )}
          
          {/* Solution Display */}
          {!showDataManagement && !showDataEditor && solution && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Solution Results
                </Typography>

                {/* Solution Summary */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        {solution.status}
                      </Typography>
                      <Typography variant="caption">Status</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        {solution.objective_value?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="caption">Objective Value</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        {solution.routes?.length || 0}
                      </Typography>
                      <Typography variant="caption">Routes</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        {solution.computation_time?.toFixed(2) || 'N/A'}s
                      </Typography>
                      <Typography variant="caption">Solve Time</Typography>
                    </Paper>
                  </Grid>
                </Grid>

                {/* Tabs for different views */}
                <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                  <Tab label="Route Summary" />
                  <Tab label="Detailed Schedule" />
                  <Tab label="Map View" />
                </Tabs>

                {/* Tab Content */}
                {tabValue === 0 && renderRouteDetails()}
                {tabValue === 1 && (
                  <Box sx={{ mt: 2 }}>
                    <PyVRPScheduleViewer
                      solution={solution}
                      clients={mergeClients()}
                      depots={problemData.depots}
                      vehicleTypes={problemData.vehicle_types}
                      getClientName={getClientName}
                    />
                  </Box>
                )}
                {tabValue === 2 && (
                  <Box sx={{ mt: 2 }}>
                    <VRPMapVisualization
                      solution={solution}
                      clients={problemData.clients}
                      depots={problemData.depots}
                      vehicleTypes={problemData.vehicle_types}
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {!showDataManagement && !showDataEditor && !solution && !loading && (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 8 }}>
                <TruckIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="textSecondary">
                  Select a problem and click "Solve Problem" to see results
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
      )}

      {/* Tab 1: データ管理 */}
      {mainTabValue === 1 && (
        <Card>
          <CardContent>
            <DataManagementPanel
              problemData={problemData}
              solution={solution}
              onProblemDataChange={setProblemData}
              onError={handleDataError}
              onSuccess={handleDataSuccess}
            />
          </CardContent>
        </Card>
      )}

      {/* Tab 2: 車両設定 */}
      {mainTabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  標準車両設定
                </Typography>
                <PyVRPDataEditor
                  problemData={problemData}
                  onDataChange={setProblemData}
                  onSave={() => {
                    setSolution(null);
                    setError(null);
                  }}
                />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  高度な車両設定
                </Typography>
                <AdvancedVehicleConfigPanel
                  vehicleTypes={advancedVehicleTypes}
                  onVehicleTypesChange={setAdvancedVehicleTypes}
                  availableSkills={availableSkills}
                  onSkillsChange={setAvailableSkills}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Tab 3: 制約設定 */}
      {mainTabValue === 3 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
              顧客制約設定
            </Typography>
            <ClientConstraintsPanel
              clients={advancedClients}
              onClientsChange={setAdvancedClients}
              availableVehicleTypes={advancedVehicleTypes.map(vt => vt.name)}
              availableSkills={availableSkills}
            />
          </CardContent>
        </Card>
      )}

      {/* Tab 4: ソルバー設定 */}
      {mainTabValue === 4 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ソルバーパラメータ
                </Typography>
                <SolverConfigPanel
                  config={solverConfig}
                  onConfigChange={setSolverConfig}
                />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  最適化実行
                </Typography>
                <Box sx={{ mt: 3 }}>
                  <Button
                    fullWidth
                    variant="contained"
                    color="primary"
                    onClick={solveProblem}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
                    sx={{ mb: 2 }}
                  >
                    {loading ? '最適化中...' : '最適化実行'}
                  </Button>
                  {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {error}
                    </Alert>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Tab 5: 結果分析 */}
      {mainTabValue === 5 && (
        <Box>
          {solution ? (
            <Grid container spacing={3}>
              {/* Solution Summary */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      最適化結果サマリー
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {solution.status}
                          </Typography>
                          <Typography variant="caption">ステータス</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {solution.objective_value?.toFixed(2) || 'N/A'}
                          </Typography>
                          <Typography variant="caption">目的関数値</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {solution.routes?.length || 0}
                          </Typography>
                          <Typography variant="caption">ルート数</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {solution.computation_time?.toFixed(2) || 'N/A'}s
                          </Typography>
                          <Typography variant="caption">計算時間</Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Result Details */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                        <Tab label="ルート詳細" />
                        <Tab label="スケジュール表示" />
                        <Tab label="地図表示" />
                        <Tab label="ガントチャート" />
                        <Tab label="統計情報" />
                      </Tabs>
                    </Box>

                    {/* Tab Content */}
                    {tabValue === 0 && renderRouteDetails()}
                    {tabValue === 1 && (
                      <Box sx={{ mt: 2 }}>
                        <PyVRPScheduleViewer
                          solution={solution}
                          clients={mergeClients()}
                          depots={problemData.depots}
                          vehicleTypes={problemData.vehicle_types}
                          getClientName={getClientName}
                        />
                      </Box>
                    )}
                    {tabValue === 2 && (
                      <Box sx={{ mt: 2 }}>
                        <VRPMapVisualization
                          solution={solution}
                          clients={problemData.clients}
                          depots={problemData.depots}
                          vehicleTypes={problemData.vehicle_types}
                        />
                      </Box>
                    )}
                    {tabValue === 3 && (
                      <Box sx={{ mt: 2 }}>
                        <VRPGanttChart
                          solution={solution}
                          clients={mergeClients()}
                          depots={problemData.depots}
                          vehicleTypes={problemData.vehicle_types}
                        />
                      </Box>
                    )}
                    {tabValue === 4 && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="h6" gutterBottom>
                          統計情報
                        </Typography>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="subtitle1" gutterBottom>
                                ルート統計
                              </Typography>
                              <Typography variant="body2">
                                総走行距離: {solution.objective_value?.toFixed(2) || 'N/A'} km
                              </Typography>
                              <Typography variant="body2">
                                計算時間: {solution.computation_time?.toFixed(2) || 'N/A'} 秒
                              </Typography>
                              <Typography variant="body2">
                                ルート数: {solution.routes?.length || 'N/A'}
                              </Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="subtitle1" gutterBottom>
                                車両統計
                              </Typography>
                              <Typography variant="body2">
                                使用車両数: {solution.routes?.length || 0} / {problemData.vehicle_types.reduce((sum, vt) => sum + vt.num_available, 0)}
                              </Typography>
                              <Typography variant="body2">
                                車両稼働率: {((solution.routes?.length || 0) / problemData.vehicle_types.reduce((sum, vt) => sum + vt.num_available, 0) * 100).toFixed(1)}%
                              </Typography>
                            </Paper>
                          </Grid>
                        </Grid>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 8 }}>
                <TruckIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="textSecondary">
                  最適化を実行すると結果がここに表示されます
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<PlayIcon />}
                  onClick={() => setMainTabValue(4)}
                  sx={{ mt: 3 }}
                >
                  ソルバー設定へ移動
                </Button>
              </CardContent>
            </Card>
          )}
        </Box>
      )}
    </Box>
  );
};

export default PyVRPInterface;