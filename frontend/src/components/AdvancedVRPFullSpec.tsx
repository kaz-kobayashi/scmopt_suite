import React, { useState, useEffect, useRef } from 'react';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Collapse,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Checkbox,
  FormGroup,
  Divider,
  Slider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  PlayArrow as PlayIcon,
  LocalShipping as TruckIcon,
  Timeline as TimelineIcon,
  LocationOn as LocationIcon,
  KeyboardArrowDown as ArrowDownIcon,
  KeyboardArrowUp as ArrowUpIcon,
  Info as InfoIcon,
  AccessTime as TimeIcon,
  Store as StoreIcon,
  LocalOffer as PrizeIcon,
  SwapHoriz as SwapIcon,
  ExpandMore as ExpandMoreIcon,
  Group as GroupIcon,
  Settings as SettingsIcon,
  Route as RouteIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  RestoreFromTrash as ReloadIcon,
  Schedule as ScheduleIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import VRPMapVisualization from './VRPMapVisualization';
import VRPGanttChart from './VRPGanttChart';

// Fix for default markers in React-Leaflet
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// CSV Data interfaces
interface ProductData {
  name: string;
  weight: number;
  volume: number;
  cust_value: number;
  dc_value: number;
  plnt_value: number;
  fixed_cost: number;
}

interface CustomerData {
  id: number;
  name: string;
  lat: number;
  lon: number;
}

interface DemandData {
  date: string;
  cust: string;
  prod: string;
  promo_0: number;
  promo_1: number;
  demand: number;
}

// Enhanced data types for full PyVRP specification
interface MultipleTimeWindow {
  early: number;
  late: number;
}

interface ClientGroup {
  group_id: string;
  client_indices: number[];
  required: boolean;
  mutually_exclusive: boolean;
  penalty?: number;
}

interface FullClientModel {
  x: number;
  y: number;
  delivery: number | number[];
  pickup: number | number[];
  service_duration: number;
  tw_early?: number;
  tw_late?: number;
  time_windows?: MultipleTimeWindow[];
  release_time?: number;
  prize?: number;
  required: boolean;
  group_id?: string;
  allowed_vehicle_types?: number[];
  priority?: number;
  service_time_multiplier?: number;
  name?: string; // For display purposes
}

interface FullDepotModel {
  x: number;
  y: number;
  tw_early?: number;
  tw_late?: number;
  capacity?: number | number[];
  is_reload_depot: boolean;
  reload_time?: number;
  depot_type?: string;
}

interface FullVehicleTypeModel {
  num_available: number;
  capacity: number | number[];
  start_depot: number;
  end_depot?: number;
  fixed_cost: number;
  unit_distance_cost?: number;
  unit_duration_cost?: number;
  tw_early?: number;
  tw_late?: number;
  max_duration?: number;
  max_distance?: number;
  profile?: string;
  can_reload: boolean;
  max_reloads?: number;
  reload_depots?: number[];
  max_work_duration?: number;
  break_duration?: number;
  forbidden_locations?: number[];
  required_locations?: number[];
}

interface RoutingProfile {
  profile_name: string;
  distance_matrix: number[][];
  duration_matrix: number[][];
  description?: string;
}

interface SolverConfig {
  max_runtime: number;
  max_iterations?: number;
  target_objective?: number;
  population_size?: number;
  min_population_size?: number;
  generation_size?: number;
  seed?: number;
  penalty_capacity?: number;
  penalty_time_window?: number;
  penalty_distance?: number;
  penalty_duration?: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`vrp-tabpanel-${index}`}
      aria-labelledby={`vrp-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const AdvancedVRPFullSpec: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Enhanced data structures
  const [clients, setClients] = useState<FullClientModel[]>([]);
  const [depots, setDepots] = useState<FullDepotModel[]>([]);
  const [vehicleTypes, setVehicleTypes] = useState<FullVehicleTypeModel[]>([]);
  const [clientGroups, setClientGroups] = useState<ClientGroup[]>([]);
  const [routingProfiles, setRoutingProfiles] = useState<RoutingProfile[]>([]);
  const [solverConfig, setSolverConfig] = useState<SolverConfig>({
    max_runtime: 60,
    population_size: 25,
    seed: 42,
    penalty_capacity: 100.0,
    penalty_time_window: 100.0,
  });

  // CSV Data state
  const [csvData, setCsvData] = useState({
    customers: [] as CustomerData[],
    products: [] as ProductData[],
    demands: [] as DemandData[]
  });
  const [dataLoaded, setDataLoaded] = useState(false);
  const [selectedDate, setSelectedDate] = useState<string>('2019-01-01');
  const [selectedCustomers, setSelectedCustomers] = useState<string[]>([]);
  const [selectedProducts, setSelectedProducts] = useState<string[]>([]);

  // UI state
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState<string>('default');
  const [editingClient, setEditingClient] = useState<number | null>(null);
  const [editingVehicle, setEditingVehicle] = useState<number | null>(null);
  const [groupDialogOpen, setGroupDialogOpen] = useState(false);
  const [newGroup, setNewGroup] = useState<Partial<ClientGroup>>({});

  // Results
  const [solution, setSolution] = useState<any>(null);
  const [showMap, setShowMap] = useState(true);
  const [expandedRoute, setExpandedRoute] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize with sample data immediately
  useEffect(() => {
    console.log('Initializing with sample data on mount...');
    generateSampleData();
  }, []);

  // Load CSV data on mount
  useEffect(() => {
    loadCSVData();
  }, []);

  // Initialize with sample data or CSV data
  useEffect(() => {
    // Only generate data if we don't have any data yet
    if (clients.length === 0 && depots.length === 0 && vehicleTypes.length === 0) {
      if (dataLoaded && csvData.customers.length > 0) {
        console.log('Generating data from CSV...');
        generateDataFromCSV();
      } else {
        console.log('Generating sample data...');
        generateSampleData();
      }
    }
  }, [dataLoaded]); // Remove other dependencies to prevent re-triggering

  // CSV data loading functions
  const loadCSVData = async () => {
    try {
      console.log('Starting CSV data load...');
      
      // Load customers
      const custResponse = await fetch('/data/cust.csv');
      if (!custResponse.ok) {
        throw new Error(`Failed to load cust.csv: ${custResponse.status}`);
      }
      const custText = await custResponse.text();
      const customers = parseCSV(custText).slice(1).map((row: string[]) => ({
        id: parseInt(row[0]),
        name: row[1],
        lat: parseFloat(row[2]),
        lon: parseFloat(row[3])
      }));
      console.log(`Loaded ${customers.length} customers`);

      // Load products
      const prodResponse = await fetch('/data/prod.csv');
      const prodText = await prodResponse.text();
      const products = parseCSV(prodText).slice(1).map((row: string[]) => ({
        name: row[1],
        weight: parseFloat(row[2]),
        volume: parseFloat(row[3]),
        cust_value: parseFloat(row[4]),
        dc_value: parseFloat(row[5]),
        plnt_value: parseFloat(row[6]),
        fixed_cost: parseFloat(row[7])
      }));

      // Load demands (sample first 1000 rows for performance)
      const demandResponse = await fetch('/data/demand_with_promo_all.csv');
      const demandText = await demandResponse.text();
      const demandRows = parseCSV(demandText).slice(1, 1001); // First 1000 rows
      const demands = demandRows.map((row: string[]) => ({
        date: row[1],
        cust: row[2],
        prod: row[3],
        promo_0: parseInt(row[4]),
        promo_1: parseInt(row[5]),
        demand: parseInt(row[6])
      }));

      setCsvData({ customers, products, demands });
      setDataLoaded(true);
      
      console.log('CSV data loaded successfully:', {
        customers: customers.length,
        products: products.length,
        demands: demands.length,
        sampleCustomer: customers[0]
      });
      
      // Set initial selections
      setSelectedCustomers(customers.slice(0, 10).map(c => c.name)); // First 10 cities
      setSelectedProducts(['A', 'B', 'C']); // First 3 products
    } catch (error) {
      console.error('Failed to load CSV data:', error);
      setDataLoaded(false);
    }
  };

  const parseCSV = (text: string): string[][] => {
    const rows = text.trim().split('\n');
    return rows.map(row => row.split(','));
  };

  const generateDataFromCSV = () => {
    if (!dataLoaded || csvData.customers.length === 0) return;

    // Filter customers by selection
    const filteredCustomers = csvData.customers.filter(c => 
      selectedCustomers.includes(c.name)
    );

    // Convert customers to client format
    const clients: FullClientModel[] = filteredCustomers.map((customer, index) => {
      // Find demand for this customer and date
      const customerDemands = csvData.demands.filter(d => 
        d.date === selectedDate && 
        d.cust === customer.name && 
        selectedProducts.includes(d.prod)
      );
      
      const totalDemand = customerDemands.reduce((sum, d) => sum + d.demand, 0);
      
      // Convert lat/lon to approximate x/y coordinates (simplified)
      const x = Math.round((customer.lon + 180) * 10000);
      const y = Math.round((customer.lat + 90) * 10000);
      
      return {
        x,
        y,
        delivery: Math.max(totalDemand, 1), // Ensure at least 1 unit demand
        pickup: 0,
        service_duration: Math.max(10, Math.floor(totalDemand / 5)), // Service time based on demand
        required: true,
        prize: 0,
        priority: 1,
        service_time_multiplier: 1.0,
        tw_early: 8 * 60, // 8 AM
        tw_late: 18 * 60, // 6 PM
        name: customer.name // Additional info for display
      };
    });

    // Set depot at Tokyo (Shinjuku)
    const tokyoCustomer = csvData.customers.find(c => c.name === '新宿区');
    const depotX = tokyoCustomer ? Math.round((tokyoCustomer.lon + 180) * 10000) : 1396503;
    const depotY = tokyoCustomer ? Math.round((tokyoCustomer.lat + 90) * 10000) : 356762;

    const depots: FullDepotModel[] = [{
      x: depotX,
      y: depotY,
      tw_early: 0,
      tw_late: 1440, // 24 hours
      is_reload_depot: true,
      reload_time: 30,
      depot_type: 'main'
    }];

    // Calculate vehicle requirements based on total demand
    const totalDemand = clients.reduce((sum, c) => sum + (Array.isArray(c.delivery) ? c.delivery[0] : c.delivery), 0);
    const vehicleCapacity = 100; // Base capacity
    const numVehicles = Math.max(1, Math.ceil(totalDemand / vehicleCapacity));

    const vehicleTypes: FullVehicleTypeModel[] = [{
      num_available: numVehicles,
      capacity: vehicleCapacity,
      start_depot: 0,
      end_depot: 0,
      fixed_cost: 100,
      unit_distance_cost: 0.001,
      unit_duration_cost: 0.5,
      can_reload: true,
      max_reloads: 2,
      reload_depots: [0],
      max_work_duration: 480, // 8 hours
      break_duration: 30,
      profile: 'truck'
    }];

    setClients(clients);
    setDepots(depots);
    setVehicleTypes(vehicleTypes);
  };

  const generateSampleData = () => {
    // Sample clients (converted to PyVRP format)
    const sampleClients: FullClientModel[] = [
      { x: 1396503, y: 356762, delivery: 20, pickup: 0, service_duration: 10, required: true, prize: 0, priority: 1, service_time_multiplier: 1.0 },
      { x: 1397531, y: 356854, delivery: 30, pickup: 0, service_duration: 15, required: true, prize: 0, priority: 1, service_time_multiplier: 1.0 },
      { x: 1397454, y: 356586, delivery: 25, pickup: 0, service_duration: 12, required: true, prize: 0, priority: 1, service_time_multiplier: 1.0 },
      { x: 1396909, y: 356908, delivery: 15, pickup: 0, service_duration: 8, required: true, prize: 0, priority: 1, service_time_multiplier: 1.0 },
      { x: 1397103, y: 356471, delivery: 35, pickup: 0, service_duration: 18, required: true, prize: 0, priority: 1, service_time_multiplier: 1.0 },
    ];
    
    // Sample depot
    const sampleDepots: FullDepotModel[] = [
      { 
        x: 1396503, 
        y: 356762, 
        tw_early: 0, 
        tw_late: 1440, 
        is_reload_depot: true,
        reload_time: 30,
        depot_type: 'main'
      }
    ];
    
    // Sample vehicle types
    const sampleVehicleTypes: FullVehicleTypeModel[] = [
      {
        num_available: 1,  // Default to 1 vehicle
        capacity: 1000,
        start_depot: 0,
        end_depot: 0,
        fixed_cost: 100,
        unit_distance_cost: 0.001,
        unit_duration_cost: 0.5,
        can_reload: true,
        max_reloads: 2,
        reload_depots: [0],
        max_work_duration: 480,
        break_duration: 30,
        profile: 'truck'
      }
    ];

    setClients(sampleClients);
    setDepots(sampleDepots);
    setVehicleTypes(sampleVehicleTypes);
    setSuccess('Sample data generated successfully');
  };

  const generateSampleDataWithSize = (numClients: number) => {
    console.log(`Generating sample data with ${numClients} clients...`);
    console.log('CSV data state:', {
      customers: csvData.customers?.length || 0,
      products: csvData.products?.length || 0,
      demands: csvData.demands?.length || 0,
      dataLoaded
    });
    
    if (!csvData.customers || csvData.customers.length === 0) {
      console.warn('CSV data not loaded, using fallback coordinates...');
      // Fallback to sample coordinates if CSV data not loaded
      const fallbackCustomers = Array.from({ length: numClients }, (_, i) => ({
        id: i + 1,
        name: `Customer_${i + 1}`,
        lat: 35.6894 + (Math.random() - 0.5) * 2, // Around Tokyo
        lon: 139.6917 + (Math.random() - 0.5) * 2
      }));
      
      const sampleClients: FullClientModel[] = fallbackCustomers.map((customer, index) => ({
        x: Math.round(customer.lon * 100),
        y: Math.round(customer.lat * 100),
        delivery: Math.round(Math.random() * 20 + 5),
        pickup: 0,
        service_duration: Math.round(Math.random() * 10 + 5),
        tw_early: 0,
        tw_late: 1440,
        required: true,
        prize: 0,
        priority: 1,
        service_time_multiplier: 1.0
      }));
      
      const sampleDepots: FullDepotModel[] = [{
        x: Math.round(139.6917 * 100), // Tokyo coordinates
        y: Math.round(35.6894 * 100),
        tw_early: 0,
        tw_late: 1440,
        is_reload_depot: true,
        reload_time: 30,
        depot_type: 'main'
      }];
      
      const totalDemand = sampleClients.reduce((sum, client) => {
        const delivery = Array.isArray(client.delivery) ? client.delivery.reduce((a, b) => a + b, 0) : client.delivery;
        return sum + delivery;
      }, 0);
      const existingNumVehicles = vehicleTypes.length > 0 ? vehicleTypes[0].num_available : 1;
      const vehicleCapacity = Math.max(200, Math.ceil(totalDemand * 1.5 / existingNumVehicles));
      const numVehicles = existingNumVehicles;
      
      const sampleVehicleTypes: FullVehicleTypeModel[] = [{
        num_available: numVehicles,
        capacity: vehicleCapacity,
        start_depot: 0,
        end_depot: 0,
        fixed_cost: 1000,
        unit_distance_cost: 0.001,
        unit_duration_cost: 0.5,
        tw_early: 0,
        tw_late: 1440,
        max_duration: 480,
        max_distance: 200000,
        can_reload: true,
        max_reloads: 2,
        reload_depots: [0],
        max_work_duration: 480,
        break_duration: 30,
        profile: 'truck'
      }];
      
      setClients(sampleClients);
      setDepots(sampleDepots);
      setVehicleTypes(sampleVehicleTypes);
      setSuccess(`Generated ${numClients} clients with fallback coordinates (CSV data not loaded)`);
      return;
    }
    
    // Filter customers to Tokyo area only (latitude around 35.6-35.8, longitude around 139.6-139.8)
    const tokyoCustomers = csvData.customers.filter(customer => 
      customer.lat >= 35.5 && customer.lat <= 35.8 && 
      customer.lon >= 139.4 && customer.lon <= 139.9
    );
    
    console.log(`Found ${tokyoCustomers.length} customers in Tokyo area from ${csvData.customers.length} total`);
    
    // Use predefined Tokyo locations (23 special wards) or filtered Tokyo customers
    const tokyoWards = [
      { name: "新宿区", lat: 35.6938, lon: 139.7034 },
      { name: "渋谷区", lat: 35.6581, lon: 139.7016 },
      { name: "港区", lat: 35.6583, lon: 139.7514 },
      { name: "千代田区", lat: 35.6938, lon: 139.7531 },
      { name: "中央区", lat: 35.6717, lon: 139.7731 },
      { name: "台東区", lat: 35.7131, lon: 139.7796 },
      { name: "墨田区", lat: 35.7100, lon: 139.8017 },
      { name: "江東区", lat: 35.6717, lon: 139.8173 },
      { name: "品川区", lat: 35.6092, lon: 139.7309 },
      { name: "目黒区", lat: 35.6415, lon: 139.6983 },
      { name: "大田区", lat: 35.5614, lon: 139.7164 },
      { name: "世田谷区", lat: 35.6464, lon: 139.6533 },
      { name: "中野区", lat: 35.7058, lon: 139.6650 },
      { name: "杉並区", lat: 35.6994, lon: 139.6364 },
      { name: "練馬区", lat: 35.7350, lon: 139.6531 },
      { name: "板橋区", lat: 35.7514, lon: 139.7089 },
      { name: "北区", lat: 35.7539, lon: 139.7342 },
      { name: "豊島区", lat: 35.7297, lon: 139.7156 },
      { name: "文京区", lat: 35.7081, lon: 139.7531 },
      { name: "荒川区", lat: 35.7364, lon: 139.7831 }
    ];
    
    let selectedCustomers = [];
    
    if (tokyoCustomers.length >= numClients) {
      // Use filtered Tokyo customers if available
      const availableCustomers = tokyoCustomers.slice();
      for (let i = 0; i < numClients && availableCustomers.length > 0; i++) {
        const randomIndex = Math.floor(Math.random() * availableCustomers.length);
        selectedCustomers.push(availableCustomers.splice(randomIndex, 1)[0]);
      }
    } else {
      // Use predefined Tokyo wards
      for (let i = 0; i < numClients; i++) {
        selectedCustomers.push(tokyoWards[i % tokyoWards.length]);
      }
    }
    
    // Generate clients using real customer coordinates (simplified coordinates)
    const sampleClients: FullClientModel[] = selectedCustomers.map((customer, index) => ({
      x: Math.round(customer.lon * 100), // Smaller scale for testing
      y: Math.round(customer.lat * 100), // Smaller scale for testing
      delivery: Math.round(Math.random() * 20 + 5), // 5-25 units (reduced demand)
      pickup: 0,
      service_duration: Math.round(Math.random() * 10 + 5), // 5-15 minutes
      tw_early: 0, // No time window constraints (0:00 AM)
      tw_late: 1440, // No time window constraints (24:00)
      required: true,
      prize: 0,
      priority: 1,
      service_time_multiplier: 1.0
    }));
    
    // Use Shinjuku as depot location (central Tokyo)
    const tokyoCustomer = { lat: 35.6938, lon: 139.7034, name: "新宿区" };
    const sampleDepots: FullDepotModel[] = [
      { 
        x: Math.round(tokyoCustomer.lon * 100), 
        y: Math.round(tokyoCustomer.lat * 100), 
        tw_early: 0, 
        tw_late: 1440, 
        is_reload_depot: true,
        reload_time: 30,
        depot_type: 'main'
      }
    ];

    // Calculate appropriate number of vehicles and capacity based on client count
    const totalDemand = sampleClients.reduce((sum, client) => {
      const delivery = Array.isArray(client.delivery) ? client.delivery.reduce((a, b) => a + b, 0) : client.delivery;
      return sum + delivery;
    }, 0);
    // Make sure we have enough capacity
    // If already have vehicle types, preserve the number of vehicles
    const existingNumVehicles = vehicleTypes.length > 0 ? vehicleTypes[0].num_available : 1;
    const vehicleCapacity = Math.max(200, Math.ceil(totalDemand * 1.5 / existingNumVehicles)); // Adjust capacity for number of vehicles
    const numVehicles = existingNumVehicles; // Use existing vehicle count

    // Sample vehicle types with appropriate capacity
    const sampleVehicleTypes: FullVehicleTypeModel[] = [
      {
        num_available: numVehicles,
        capacity: vehicleCapacity,
        start_depot: 0,
        end_depot: 0,
        fixed_cost: 1000,
        unit_distance_cost: 0.001,
        unit_duration_cost: 0.5,
        tw_early: 0,
        tw_late: 1440,
        max_duration: 480, // 8 hours
        max_distance: 200000, // 200km
        can_reload: true,
        max_reloads: 2,
        reload_depots: [0],
        max_work_duration: 480,
        break_duration: 30,
        profile: 'truck'
      }
    ];

    setClients(sampleClients);
    setDepots(sampleDepots);
    setVehicleTypes(sampleVehicleTypes);
    
    setSuccess(`Generated ${numClients} clients with ${numVehicles} vehicles (capacity: ${vehicleCapacity} each)`);
  };

  const addClient = () => {
    // Use Tokyo coordinates as base (Shinjuku area)
    const baseLatLon = csvData.customers.find(c => c.name === '新宿区') || { lat: 35.6894, lon: 139.6917 };
    
    const newClient: FullClientModel = {
      x: Math.round((baseLatLon.lon + (Math.random() - 0.5) * 0.2) * 100), // Small variation around Tokyo
      y: Math.round((baseLatLon.lat + (Math.random() - 0.5) * 0.2) * 100), // Small variation around Tokyo
      delivery: Math.round(Math.random() * 20 + 5),
      pickup: 0,
      service_duration: 10,
      tw_early: 0,
      tw_late: 1440,
      required: true,
      prize: 0,
      priority: 1,
      service_time_multiplier: 1.0
    };
    setClients([...clients, newClient]);
  };

  const updateClient = (index: number, updates: Partial<FullClientModel>) => {
    const updatedClients = [...clients];
    updatedClients[index] = { ...updatedClients[index], ...updates };
    setClients(updatedClients);
  };

  const removeClient = (index: number) => {
    setClients(clients.filter((_, i) => i !== index));
  };

  const addVehicleType = () => {
    const newVehicleType: FullVehicleTypeModel = {
      num_available: 1,
      capacity: 1000,
      start_depot: 0,
      end_depot: 0,
      fixed_cost: 100,
      unit_distance_cost: 0.001,
      unit_duration_cost: 0.5,
      can_reload: false,
      profile: 'default'
    };
    setVehicleTypes([...vehicleTypes, newVehicleType]);
  };

  const updateVehicleType = (index: number, updates: Partial<FullVehicleTypeModel>) => {
    const updatedVehicleTypes = [...vehicleTypes];
    updatedVehicleTypes[index] = { ...updatedVehicleTypes[index], ...updates };
    setVehicleTypes(updatedVehicleTypes);
  };

  const removeVehicleType = (index: number) => {
    setVehicleTypes(vehicleTypes.filter((_, i) => i !== index));
  };

  const addClientGroup = () => {
    if (newGroup.group_id && newGroup.client_indices) {
      const group: ClientGroup = {
        group_id: newGroup.group_id,
        client_indices: newGroup.client_indices,
        required: newGroup.required || false,
        mutually_exclusive: newGroup.mutually_exclusive || false,
        penalty: newGroup.penalty
      };
      setClientGroups([...clientGroups, group]);
      setNewGroup({});
      setGroupDialogOpen(false);
    }
  };

  const solveFullSpecVRP = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    console.log('Solving VRP with data:', {
      clients: clients.length,
      depots: depots.length,
      vehicle_types: vehicleTypes.length,
      solver_config: solverConfig
    });
    console.log('Actual clients data:', clients);
    console.log('Actual depots data:', depots);
    console.log('Actual vehicle types data:', vehicleTypes);
    
    // Validate data before sending
    if (!clients || clients.length === 0) {
      setError('No clients data available. Please generate sample data first.');
      setLoading(false);
      return;
    }
    
    if (!depots || depots.length === 0) {
      setError('No depot data available. Please generate sample data first.');
      setLoading(false);
      return;
    }
    
    if (!vehicleTypes || vehicleTypes.length === 0) {
      setError('No vehicle type data available. Please generate sample data first.');
      setLoading(false);
      return;
    }
    
    try {
      const problemData = {
        clients,
        depots,
        vehicle_types: vehicleTypes,
        client_groups: clientGroups.length > 0 ? clientGroups : undefined,
        routing_profiles: routingProfiles.length > 0 ? routingProfiles : undefined,
        solver_config: solverConfig,
        max_runtime: solverConfig.max_runtime
      };

      // Debug: Log the request data
      console.log('Sending request to /api/pyvrp/solve with data:', JSON.stringify(problemData, null, 2));
      
      // Call the unified PyVRP API
      const response = await fetch('/api/pyvrp/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(problemData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error Response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      const result = await response.json();
      setSolution(result);
      setSuccess(`Optimization completed: ${result.status}`);
      setTabValue(4); // Switch to results tab
    } catch (err: any) {
      setError(err.message || 'Optimization failed');
    } finally {
      setLoading(false);
    }
  };

  const getRouteColor = (index: number) => {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48dbfb', '#ff9ff3', '#54a0ff'];
    return colors[index % colors.length];
  };

  // Analysis helper functions
  const getTopCustomersByDemand = () => {
    const filteredDemands = csvData.demands.filter(d => 
      d.date === selectedDate && 
      selectedProducts.includes(d.prod)
    );

    const customerDemands = filteredDemands.reduce((acc: Record<string, number>, d) => {
      acc[d.cust] = (acc[d.cust] || 0) + d.demand;
      return acc;
    }, {});

    return Object.entries(customerDemands)
      .map(([customer, demand]) => ({ customer, demand }))
      .sort((a, b) => b.demand - a.demand);
  };

  const getTopProductsByDemand = () => {
    const filteredDemands = csvData.demands.filter(d => 
      d.date === selectedDate && 
      selectedCustomers.includes(d.cust)
    );

    const productDemands = filteredDemands.reduce((acc: Record<string, number>, d) => {
      acc[d.prod] = (acc[d.prod] || 0) + d.demand;
      return acc;
    }, {});

    return Object.entries(productDemands)
      .map(([product, demand]) => ({ product, demand }))
      .sort((a, b) => b.demand - a.demand);
  };

  const getProductWeight = (productName: string) => {
    const product = csvData.products.find(p => p.name === productName);
    return product ? product.weight : 0;
  };

  const renderDataSelection = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        データ選択とフィルタリング
      </Typography>
      
      {!dataLoaded ? (
        <Alert severity="info" sx={{ mb: 2 }}>
          CSV データを読み込み中...
        </Alert>
      ) : (
        <Alert severity="success" sx={{ mb: 2 }}>
          データ読み込み完了: {csvData.customers.length}都市, {csvData.products.length}製品, {csvData.demands.length}需要レコード
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Date Selection */}
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="対象日付"
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
          />
        </Grid>

        {/* Customer Selection */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>顧客都市選択</InputLabel>
            <Select
              multiple
              value={selectedCustomers}
              onChange={(e) => setSelectedCustomers(e.target.value as string[])}
              renderValue={(selected) => `${selected.length}都市選択済み`}
            >
              {csvData.customers.map((customer) => (
                <MenuItem key={customer.id} value={customer.name}>
                  <Checkbox checked={selectedCustomers.includes(customer.name)} />
                  <ListItemText primary={customer.name} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Product Selection */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>製品選択</InputLabel>
            <Select
              multiple
              value={selectedProducts}
              onChange={(e) => setSelectedProducts(e.target.value as string[])}
              renderValue={(selected) => `${selected.length}製品選択済み`}
            >
              {csvData.products.map((product) => (
                <MenuItem key={product.name} value={product.name}>
                  <Checkbox checked={selectedProducts.includes(product.name)} />
                  <ListItemText primary={`${product.name} (重量: ${product.weight})`} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Quick Selection Buttons */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
            <Button
              variant="outlined"
              onClick={() => setSelectedCustomers(csvData.customers.slice(0, 5).map(c => c.name))}
            >
              主要都市5つ
            </Button>
            <Button
              variant="outlined"
              onClick={() => setSelectedCustomers(csvData.customers.slice(0, 10).map(c => c.name))}
            >
              主要都市10つ
            </Button>
            <Button
              variant="outlined"
              onClick={() => setSelectedProducts(['A', 'B', 'C'])}
            >
              製品A-C
            </Button>
            <Button
              variant="outlined"
              onClick={() => setSelectedProducts(csvData.products.map(p => p.name))}
            >
              全製品
            </Button>
          </Box>
        </Grid>

        {/* Summary Statistics */}
        {dataLoaded && (
          <Grid item xs={12}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  選択データサマリー
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Typography variant="h4" color="primary">
                      {selectedCustomers.length}
                    </Typography>
                    <Typography variant="body2">選択都市数</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="h4" color="secondary">
                      {selectedProducts.length}
                    </Typography>
                    <Typography variant="body2">選択製品数</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="h4" color="success.main">
                      {csvData.demands.filter(d => 
                        d.date === selectedDate && 
                        selectedCustomers.includes(d.cust) && 
                        selectedProducts.includes(d.prod)
                      ).reduce((sum, d) => sum + d.demand, 0)}
                    </Typography>
                    <Typography variant="body2">総需要量</Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="h4" color="warning.main">
                      {clients.length}
                    </Typography>
                    <Typography variant="body2">生成されたクライアント数</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Analysis Charts */}
        {dataLoaded && csvData.demands.length > 0 && (
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              需要分析
            </Typography>
            <Grid container spacing={2}>
              {/* Top Customers by Demand */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      需要上位都市 (選択製品)
                    </Typography>
                    <List dense>
                      {getTopCustomersByDemand().slice(0, 5).map((item, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Chip label={index + 1} size="small" color="primary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={item.customer}
                            secondary={`需要量: ${item.demand}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>

              {/* Top Products by Demand */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      需要上位製品 (選択都市)
                    </Typography>
                    <List dense>
                      {getTopProductsByDemand().slice(0, 5).map((item, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Chip label={index + 1} size="small" color="secondary" />
                          </ListItemIcon>
                          <ListItemText 
                            primary={`製品 ${item.product}`}
                            secondary={`需要量: ${item.demand} | 重量: ${getProductWeight(item.product)}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        )}

        {/* Preview Table */}
        {dataLoaded && csvData.demands.length > 0 && (
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              需要データプレビュー ({selectedDate})
            </Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>都市</TableCell>
                    <TableCell>製品</TableCell>
                    <TableCell align="right">需要量</TableCell>
                    <TableCell align="right">プロモ0</TableCell>
                    <TableCell align="right">プロモ1</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {csvData.demands
                    .filter(d => 
                      d.date === selectedDate && 
                      selectedCustomers.includes(d.cust) && 
                      selectedProducts.includes(d.prod)
                    )
                    .slice(0, 50) // Show first 50 rows
                    .map((demand, index) => (
                      <TableRow key={index}>
                        <TableCell>{demand.cust}</TableCell>
                        <TableCell>{demand.prod}</TableCell>
                        <TableCell align="right">{demand.demand}</TableCell>
                        <TableCell align="right">{demand.promo_0}</TableCell>
                        <TableCell align="right">{demand.promo_1}</TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
            {csvData.demands.filter(d => 
              d.date === selectedDate && 
              selectedCustomers.includes(d.cust) && 
              selectedProducts.includes(d.prod)
            ).length > 50 && (
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                最初の50行を表示中。総{csvData.demands.filter(d => 
                  d.date === selectedDate && 
                  selectedCustomers.includes(d.cust) && 
                  selectedProducts.includes(d.prod)
                ).length}行があります。
              </Typography>
            )}
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderClientManagement = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Client Management</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            onClick={() => generateSampleDataWithSize(5)}
          >
            5 CLIENTS
          </Button>
          <Button
            variant="outlined"
            onClick={() => generateSampleDataWithSize(15)}
          >
            15 CLIENTS
          </Button>
          <Button
            variant="outlined"
            onClick={() => generateSampleDataWithSize(30)}
          >
            30 CLIENTS
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={addClient}
          >
            Add Client
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Coordinates</TableCell>
              <TableCell>Delivery</TableCell>
              <TableCell>Service Time</TableCell>
              <TableCell>Time Window</TableCell>
              <TableCell>Group</TableCell>
              <TableCell>Priority</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {clients.map((client, index) => (
              <TableRow key={index}>
                <TableCell>{index}</TableCell>
                <TableCell>
                  <Typography variant="body2">
                    ({(client.x / 100).toFixed(4)}, {(client.y / 100).toFixed(4)})
                  </Typography>
                </TableCell>
                <TableCell>
                  <TextField
                    size="small"
                    type="number"
                    value={Array.isArray(client.delivery) ? client.delivery[0] : client.delivery}
                    onChange={(e) => updateClient(index, { delivery: Number(e.target.value) })}
                  />
                </TableCell>
                <TableCell>
                  <TextField
                    size="small"
                    type="number"
                    value={client.service_duration}
                    onChange={(e) => updateClient(index, { service_duration: Number(e.target.value) })}
                  />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                      size="small"
                      type="number"
                      placeholder="Early"
                      value={client.tw_early || 0}
                      onChange={(e) => updateClient(index, { tw_early: Number(e.target.value) })}
                      sx={{ width: 80 }}
                    />
                    <TextField
                      size="small"
                      type="number"
                      placeholder="Late"
                      value={client.tw_late || 1440}
                      onChange={(e) => updateClient(index, { tw_late: Number(e.target.value) })}
                      sx={{ width: 80 }}
                    />
                  </Box>
                </TableCell>
                <TableCell>
                  <Select
                    size="small"
                    value={client.group_id || ''}
                    onChange={(e) => updateClient(index, { group_id: e.target.value || undefined })}
                    displayEmpty
                  >
                    <MenuItem value="">None</MenuItem>
                    {clientGroups.map((group) => (
                      <MenuItem key={group.group_id} value={group.group_id}>
                        {group.group_id}
                      </MenuItem>
                    ))}
                  </Select>
                </TableCell>
                <TableCell>
                  <TextField
                    size="small"
                    type="number"
                    value={client.priority || 1}
                    onChange={(e) => updateClient(index, { priority: Number(e.target.value) })}
                    sx={{ width: 60 }}
                  />
                </TableCell>
                <TableCell>
                  <IconButton size="small" onClick={() => removeClient(index)} color="error">
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );

  const renderVehicleManagement = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Vehicle Type Management</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={addVehicleType}
        >
          Add Vehicle Type
        </Button>
      </Box>
      
      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="body2">
          車両台数は "Number Available" フィールドで設定できます。1台のみで配送したい場合は、このフィールドを 1 に設定してください。
        </Typography>
      </Alert>

      {vehicleTypes.map((vehicle, index) => (
        <Accordion key={index} sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <TruckIcon />
              <Typography variant="subtitle1">
                Vehicle Type {index + 1} ({vehicle.num_available} available)
              </Typography>
              {vehicle.can_reload && (
                <Chip
                  icon={<ReloadIcon />}
                  label="Reload"
                  size="small"
                  color="primary"
                />
              )}
            </Box>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  label="Number Available"
                  type="number"
                  value={vehicle.num_available}
                  onChange={(e) => updateVehicleType(index, { num_available: Number(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  label="Capacity"
                  type="number"
                  value={Array.isArray(vehicle.capacity) ? vehicle.capacity[0] : vehicle.capacity}
                  onChange={(e) => updateVehicleType(index, { capacity: Number(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  label="Fixed Cost"
                  type="number"
                  value={vehicle.fixed_cost}
                  onChange={(e) => updateVehicleType(index, { fixed_cost: Number(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  label="Distance Cost"
                  type="number"
                  inputProps={{ step: "0.001" }}
                  value={vehicle.unit_distance_cost || 0}
                  onChange={(e) => updateVehicleType(index, { unit_distance_cost: Number(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <TextField
                  fullWidth
                  label="Duration Cost"
                  type="number"
                  inputProps={{ step: "0.1" }}
                  value={vehicle.unit_duration_cost || 0}
                  onChange={(e) => updateVehicleType(index, { unit_duration_cost: Number(e.target.value) })}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Profile</InputLabel>
                  <Select
                    value={vehicle.profile || 'default'}
                    onChange={(e) => updateVehicleType(index, { profile: e.target.value })}
                  >
                    <MenuItem value="default">Default</MenuItem>
                    <MenuItem value="car">Car</MenuItem>
                    <MenuItem value="truck">Truck</MenuItem>
                    <MenuItem value="bicycle">Bicycle</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>Reload Options</Typography>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={vehicle.can_reload}
                      onChange={(e) => updateVehicleType(index, { can_reload: e.target.checked })}
                    />
                  }
                  label="Can Reload"
                />
                {vehicle.can_reload && (
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Max Reloads"
                        type="number"
                        value={vehicle.max_reloads || 1}
                        onChange={(e) => updateVehicleType(index, { max_reloads: Number(e.target.value) })}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="Reload Depots (comma-separated)"
                        value={vehicle.reload_depots?.join(',') || '0'}
                        onChange={(e) => updateVehicleType(index, { 
                          reload_depots: e.target.value.split(',').map(x => Number(x.trim())).filter(x => !isNaN(x))
                        })}
                      />
                    </Grid>
                  </Grid>
                )}
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    color="error"
                    startIcon={<DeleteIcon />}
                    onClick={() => removeVehicleType(index)}
                  >
                    Remove Vehicle Type
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );

  const renderClientGroups = () => (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Client Groups</Typography>
        <Button
          variant="contained"
          startIcon={<GroupIcon />}
          onClick={() => setGroupDialogOpen(true)}
        >
          Add Group
        </Button>
      </Box>

      {clientGroups.map((group, index) => (
        <Card key={index} sx={{ mb: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="subtitle1">{group.group_id}</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                {group.required && <Chip label="Required" color="warning" size="small" />}
                {group.mutually_exclusive && <Chip label="Exclusive" color="error" size="small" />}
                <IconButton size="small" onClick={() => {
                  setClientGroups(clientGroups.filter((_, i) => i !== index));
                }}>
                  <DeleteIcon />
                </IconButton>
              </Box>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Clients: {group.client_indices.join(', ')}
            </Typography>
            {group.penalty && (
              <Typography variant="body2" color="text.secondary">
                Penalty: {group.penalty}
              </Typography>
            )}
          </CardContent>
        </Card>
      ))}

      <Dialog open={groupDialogOpen} onClose={() => setGroupDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Client Group</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Group ID"
                value={newGroup.group_id || ''}
                onChange={(e) => setNewGroup({ ...newGroup, group_id: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Client Indices (comma-separated)"
                placeholder="0,1,2"
                value={newGroup.client_indices?.join(',') || ''}
                onChange={(e) => setNewGroup({ 
                  ...newGroup, 
                  client_indices: e.target.value.split(',').map(x => Number(x.trim())).filter(x => !isNaN(x))
                })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={newGroup.required || false}
                      onChange={(e) => setNewGroup({ ...newGroup, required: e.target.checked })}
                    />
                  }
                  label="Required (at least one client must be visited)"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={newGroup.mutually_exclusive || false}
                      onChange={(e) => setNewGroup({ ...newGroup, mutually_exclusive: e.target.checked })}
                    />
                  }
                  label="Mutually Exclusive (exactly one client must be visited)"
                />
              </FormGroup>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Penalty (optional)"
                type="number"
                value={newGroup.penalty || ''}
                onChange={(e) => setNewGroup({ ...newGroup, penalty: Number(e.target.value) })}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGroupDialogOpen(false)}>Cancel</Button>
          <Button onClick={addClientGroup} variant="contained">Add Group</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );

  const renderSolverConfig = () => (
    <Box>
      <Typography variant="h6" gutterBottom>Solver Configuration</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Max Runtime (seconds)"
            type="number"
            value={solverConfig.max_runtime}
            onChange={(e) => setSolverConfig({ ...solverConfig, max_runtime: Number(e.target.value) })}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Population Size"
            type="number"
            value={solverConfig.population_size || 25}
            onChange={(e) => setSolverConfig({ ...solverConfig, population_size: Number(e.target.value) })}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Random Seed"
            type="number"
            value={solverConfig.seed || ''}
            onChange={(e) => setSolverConfig({ ...solverConfig, seed: Number(e.target.value) })}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Max Iterations"
            type="number"
            value={solverConfig.max_iterations || ''}
            onChange={(e) => setSolverConfig({ ...solverConfig, max_iterations: Number(e.target.value) })}
          />
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>Penalty Configuration</Typography>
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Capacity Penalty"
            type="number"
            inputProps={{ step: "0.1" }}
            value={solverConfig.penalty_capacity || 100}
            onChange={(e) => setSolverConfig({ ...solverConfig, penalty_capacity: Number(e.target.value) })}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Time Window Penalty"
            type="number"
            inputProps={{ step: "0.1" }}
            value={solverConfig.penalty_time_window || 100}
            onChange={(e) => setSolverConfig({ ...solverConfig, penalty_time_window: Number(e.target.value) })}
          />
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
          <Button
            variant="outlined"
            size="medium"
            onClick={() => generateSampleDataWithSize(5)}
            disabled={loading}
          >
            5 Clients
          </Button>
          <Button
            variant="outlined"
            size="medium"
            onClick={() => generateSampleDataWithSize(15)}
            disabled={loading}
          >
            15 Clients
          </Button>
          <Button
            variant="outlined"
            size="medium"
            onClick={() => generateSampleDataWithSize(30)}
            disabled={loading}
          >
            30 Clients
          </Button>
          <Button
            variant="outlined"
            size="medium"
            onClick={() => generateSampleDataWithSize(50)}
            disabled={loading}
          >
            50 Clients
          </Button>
          <Button
            variant="outlined"
            size="medium"
            onClick={() => generateSampleDataWithSize(100)}
            disabled={loading}
          >
            100 Clients
          </Button>
        </Box>
        <Button
          variant="contained"
          size="large"
          startIcon={<PlayIcon />}
          onClick={solveFullSpecVRP}
          disabled={loading || clients.length === 0 || depots.length === 0 || vehicleTypes.length === 0}
        >
          {loading ? 'Optimizing...' : 'Solve VRP'}
        </Button>
      </Box>
    </Box>
  );

  const renderResults = () => {
    if (!solution) {
      return (
        <Box sx={{ textAlign: 'center', py: 5 }}>
          <Typography variant="body1" color="text.secondary">
            No optimization results yet. Configure parameters and run optimization.
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TruckIcon color="primary" sx={{ fontSize: 40 }} />
              <Typography variant="h6">{solution.routes?.length || 0}</Typography>
              <Typography variant="body2" color="text.secondary">Routes</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TimelineIcon color="secondary" sx={{ fontSize: 40 }} />
              <Typography variant="h6">
                {solution.statistics?.total_distance 
                  ? (solution.statistics.total_distance / 1000).toFixed(1) + ' km'
                  : 'N/A'
                }
              </Typography>
              <Typography variant="body2" color="text.secondary">Total Distance</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TimeIcon color="success" sx={{ fontSize: 40 }} />
              <Typography variant="h6">{solution.computation_time?.toFixed(2) || '0.00'}s</Typography>
              <Typography variant="body2" color="text.secondary">Computation Time</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Chip 
                label={solution.status?.toUpperCase() || 'UNKNOWN'}
                color={solution.status === 'optimal' ? 'success' : 'warning'}
                sx={{ mt: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Solution Status
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {solution.statistics && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>Solution Statistics</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">Clients Served</Typography>
                <Typography variant="h6">{solution.statistics.clients_served}</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">Vehicles Used</Typography>
                <Typography variant="h6">{solution.statistics.vehicles_used}</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">Total Duration</Typography>
                <Typography variant="h6">{Math.round(solution.statistics.total_duration / 60)}h</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">Avg. Capacity Utilization</Typography>
                <Typography variant="h6">{(solution.statistics.average_capacity_utilization * 100).toFixed(1)}%</Typography>
              </Grid>
            </Grid>
          </Paper>
        )}

        {/* Enhanced Map Visualization */}
        <VRPMapVisualization
          solution={solution}
          clients={clients}
          depots={depots}
          vehicleTypes={vehicleTypes}
          clientGroups={clientGroups}
        />

        {/* Gantt Chart for Schedule Visualization */}
        <Box sx={{ mt: 3 }}>
          <VRPGanttChart
            solution={solution}
            clients={clients}
            depots={depots}
            vehicleTypes={vehicleTypes}
          />
        </Box>

        <Typography variant="h6" gutterBottom>Route Details</Typography>
        
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell />
                <TableCell>Route</TableCell>
                <TableCell>Vehicle Type</TableCell>
                <TableCell>Clients</TableCell>
                <TableCell align="right">Distance</TableCell>
                <TableCell align="right">Duration</TableCell>
                <TableCell align="right">Load</TableCell>
                <TableCell>Features</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {solution.routes?.map((route: any, index: number) => (
                <React.Fragment key={index}>
                  <TableRow
                    hover
                    onClick={() => setExpandedRoute(expandedRoute === index ? null : index)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>
                      <IconButton size="small">
                        {expandedRoute === index ? <ArrowUpIcon /> : <ArrowDownIcon />}
                      </IconButton>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={`Route ${index + 1}`}
                        size="small"
                        sx={{ backgroundColor: getRouteColor(index), color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>{route.vehicle_type || 0}</TableCell>
                    <TableCell>{route.clients?.length || 0}</TableCell>
                    <TableCell align="right">{((route.distance || 0) / 1000).toFixed(1)} km</TableCell>
                    <TableCell align="right">{Math.round((route.duration || 0) / 60)}m</TableCell>
                    <TableCell align="right">{route.demand_served || 0}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        {route.reloads?.length > 0 && (
                          <Chip
                            icon={<ReloadIcon />}
                            label={route.reloads.length}
                            size="small"
                            color="primary"
                          />
                        )}
                        {route.breaks?.length > 0 && (
                          <Chip
                            icon={<ScheduleIcon />}
                            label={route.breaks.length}
                            size="small"
                            color="secondary"
                          />
                        )}
                      </Box>
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={8}>
                      <Collapse in={expandedRoute === index} timeout="auto" unmountOnExit>
                        <Box sx={{ margin: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Route Sequence:
                          </Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                            <Chip
                              icon={<StoreIcon />}
                              label="Depot"
                              size="small"
                              variant="filled"
                            />
                            {route.clients?.map((clientIdx: number, locIndex: number) => (
                              <React.Fragment key={locIndex}>
                                <Chip
                                  icon={<LocationIcon />}
                                  label={`Client ${clientIdx}`}
                                  size="small"
                                  variant="outlined"
                                />
                                {locIndex < route.clients.length - 1 && ' → '}
                              </React.Fragment>
                            ))}
                            <Chip
                              icon={<StoreIcon />}
                              label="Depot"
                              size="small"
                              variant="filled"
                            />
                          </Box>
                          
                          {route.reloads?.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                Reload Operations:
                              </Typography>
                              {route.reloads.map((reload: any, reloadIdx: number) => (
                                <Typography key={reloadIdx} variant="body2">
                                  Reload at depot {reload.depot_index} (position {reload.position_in_route}, {reload.reload_duration}min)
                                </Typography>
                              ))}
                            </Box>
                          )}
                          
                          {route.breaks?.length > 0 && (
                            <Box sx={{ mb: 2 }}>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                Breaks:
                              </Typography>
                              {route.breaks.map((breakInfo: any, breakIdx: number) => (
                                <Typography key={breakIdx} variant="body2">
                                  Break at position {breakInfo.position_in_route} ({breakInfo.break_duration}min)
                                </Typography>
                              ))}
                            </Box>
                          )}
                        </Box>
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Advanced PyVRP - Full Specification
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Complete PyVRP implementation with client groups, multiple routing profiles, reload functionality, and advanced solver configuration
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      <Card>
        <CardContent>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} variant="scrollable">
              <Tab 
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <StoreIcon />
                    Data Selection
                  </Box>
                } 
              />
              <Tab 
                label={
                  <Badge badgeContent={clients.length} color="primary">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LocationIcon />
                      Clients
                    </Box>
                  </Badge>
                } 
              />
              <Tab 
                label={
                  <Badge badgeContent={vehicleTypes.length} color="secondary">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TruckIcon />
                      Vehicles
                    </Box>
                  </Badge>
                } 
              />
              <Tab 
                label={
                  <Badge badgeContent={clientGroups.length} color="success">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <GroupIcon />
                      Groups
                    </Box>
                  </Badge>
                } 
              />
              <Tab 
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SettingsIcon />
                    Solver
                  </Box>
                } 
              />
              <Tab 
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TimelineIcon />
                    Results
                  </Box>
                } 
              />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            {renderDataSelection()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderClientManagement()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderVehicleManagement()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            {renderClientGroups()}
          </TabPanel>

          <TabPanel value={tabValue} index={4}>
            {renderSolverConfig()}
          </TabPanel>

          <TabPanel value={tabValue} index={5}>
            {renderResults()}
          </TabPanel>
        </CardContent>
      </Card>

      {loading && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
        >
          <Paper sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress />
            <Typography>Optimizing with PyVRP...</Typography>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default AdvancedVRPFullSpec;