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
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { 
  ApiService, LocationModel, TimeWindow, VRPSolution, VRPVariantInfo,
  VRPProblemData, ClientModel, UnifiedDepotModel, VehicleTypeModel, UnifiedVRPSolution
} from '../services/apiClient';

// Fix for default markers in React-Leaflet
// delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

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

const AdvancedVRP: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // VRP Variant selection
  const [selectedVariant, setSelectedVariant] = useState<string>('CVRP');
  const [variantInfo, setVariantInfo] = useState<{ [key: string]: VRPVariantInfo }>({});

  // Common parameters
  const [locations, setLocations] = useState<LocationModel[]>([]);
  const [vehicleCapacity, setVehicleCapacity] = useState<number>(1000);
  const [numVehicles, setNumVehicles] = useState<number>(5);
  const [maxRuntime, setMaxRuntime] = useState<number>(60);
  const [depotIndex, setDepotIndex] = useState<number>(0);

  // VRPTW specific
  const [timeWindows, setTimeWindows] = useState<TimeWindow[]>([]);
  const [serviceTimes, setServiceTimes] = useState<number[]>([]);

  // MDVRP specific
  const [depots, setDepots] = useState<any[]>([]);
  const [depotIndices, setDepotIndices] = useState<number[]>([]);

  // PDVRP specific
  const [pickupDeliveryPairs, setPickupDeliveryPairs] = useState<any[]>([]);

  // PC-VRP specific
  const [prizes, setPrizes] = useState<number[]>([]);
  const [minPrize, setMinPrize] = useState<number>(100);

  // Results
  const [solution, setSolution] = useState<VRPSolution | null>(null);
  const [solutions, setSolutions] = useState<VRPSolution[]>([]);
  const [unifiedSolution, setUnifiedSolution] = useState<UnifiedVRPSolution | null>(null);
  const [showMap, setShowMap] = useState(true);
  const [expandedRoute, setExpandedRoute] = useState<number | null>(null);

  // File upload
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadVRPVariants();
  }, []);

  const loadVRPVariants = async () => {
    try {
      const data = await ApiService.getVRPVariants();
      setVariantInfo(data.variants);
    } catch (err) {
      console.error('Failed to load VRP variants:', err);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.uploadVRPData(file, 'locations');
      
      if (result.validation.is_valid) {
        // Parse locations from CSV
        const parsedLocations: LocationModel[] = result.sample_data.map((row: any) => ({
          name: row.name || `Location_${row.index}`,
          lat: parseFloat(row.lat),
          lon: parseFloat(row.lon),
          demand: parseFloat(row.demand || 1),
        }));
        
        setLocations(parsedLocations);
        setSuccess(`Successfully loaded ${parsedLocations.length} locations`);
        
        // Initialize variant-specific arrays
        if (selectedVariant === 'VRPTW') {
          setTimeWindows(parsedLocations.map(() => ({ earliest: 0, latest: 24 })));
          setServiceTimes(parsedLocations.map(() => 0.5));
        } else if (selectedVariant === 'PC-VRP') {
          setPrizes(parsedLocations.map(() => Math.random() * 100));
        }
      } else {
        setError(`Data validation failed: ${result.validation.errors.join(', ')}`);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to upload file');
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = () => {
    // Generate sample locations
    const sampleLocations: LocationModel[] = [
      { name: 'Depot', lat: 35.6762, lon: 139.6503, demand: 0 },
      { name: 'Customer_1', lat: 35.6854, lon: 139.7531, demand: 20 },
      { name: 'Customer_2', lat: 35.6586, lon: 139.7454, demand: 30 },
      { name: 'Customer_3', lat: 35.6908, lon: 139.6909, demand: 25 },
      { name: 'Customer_4', lat: 35.6471, lon: 139.7103, demand: 15 },
      { name: 'Customer_5', lat: 35.6989, lon: 139.7746, demand: 35 },
      { name: 'Customer_6', lat: 35.6551, lon: 139.7556, demand: 40 },
      { name: 'Customer_7', lat: 35.6795, lon: 139.7656, demand: 20 },
      { name: 'Customer_8', lat: 35.6696, lon: 139.7081, demand: 25 },
      { name: 'Customer_9', lat: 35.6939, lon: 139.7037, demand: 30 },
      { name: 'Customer_10', lat: 35.6598, lon: 139.7008, demand: 15 },
    ];
    
    setLocations(sampleLocations);
    
    // Initialize variant-specific data
    if (selectedVariant === 'VRPTW') {
      setTimeWindows(sampleLocations.map((_, i) => 
        i === 0 ? { earliest: 0, latest: 24 } : { earliest: 8 + Math.random() * 4, latest: 16 + Math.random() * 4 }
      ));
      setServiceTimes(sampleLocations.map((_, i) => i === 0 ? 0 : 0.5));
    } else if (selectedVariant === 'MDVRP') {
      setDepotIndices([0, 5]);
      setDepots([
        { name: 'Depot_1', capacity: 1000, num_vehicles: 3 },
        { name: 'Depot_2', capacity: 1000, num_vehicles: 2 },
      ]);
    } else if (selectedVariant === 'PDVRP') {
      setPickupDeliveryPairs([
        { pickup_location_idx: 1, delivery_location_idx: 6, demand: 20 },
        { pickup_location_idx: 2, delivery_location_idx: 7, demand: 30 },
        { pickup_location_idx: 3, delivery_location_idx: 8, demand: 25 },
      ]);
    } else if (selectedVariant === 'PC-VRP') {
      setPrizes(sampleLocations.map((_, i) => i === 0 ? 0 : Math.round(Math.random() * 100 + 50)));
    }
    
    setSuccess('Sample data generated successfully');
  };

  const solveVRP = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      let result: VRPSolution;
      
      switch (selectedVariant) {
        case 'CVRP':
          result = await ApiService.solveCVRP({
            locations,
            depot_index: depotIndex,
            vehicle_capacity: vehicleCapacity,
            num_vehicles: numVehicles,
            max_runtime: maxRuntime,
          });
          break;
          
        case 'VRPTW':
          result = await ApiService.solveVRPTW({
            locations,
            time_windows: timeWindows,
            service_times: serviceTimes,
            depot_index: depotIndex,
            vehicle_capacity: vehicleCapacity,
            num_vehicles: numVehicles,
            max_runtime: maxRuntime,
          });
          break;
          
        case 'MDVRP':
          result = await ApiService.solveMDVRP({
            locations,
            depots: depots.map((d, i) => ({
              ...d,
              lat: locations[depotIndices[i]].lat,
              lon: locations[depotIndices[i]].lon,
            })),
            depot_indices: depotIndices,
            max_runtime: maxRuntime,
          });
          break;
          
        case 'PDVRP':
          result = await ApiService.solvePDVRP({
            locations,
            pickup_delivery_pairs: pickupDeliveryPairs,
            depot_index: depotIndex,
            vehicle_capacity: vehicleCapacity,
            max_runtime: maxRuntime,
          });
          break;
          
        case 'PC-VRP':
          result = await ApiService.solvePCVRP({
            locations,
            prizes,
            depot_index: depotIndex,
            vehicle_capacity: vehicleCapacity,
            min_prize: minPrize,
            max_runtime: maxRuntime,
          });
          break;
          
        default:
          throw new Error('Unknown VRP variant');
      }
      
      setSolution(result);
      setSolutions([...solutions, result]);
      setSuccess(`Optimization completed: ${result.status}`);
      setTabValue(2); // Switch to results tab
    } catch (err: any) {
      setError(err.message || 'Optimization failed');
    } finally {
      setLoading(false);
    }
  };

  const solveUnifiedVRP = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Convert locations to unified format
      const clients: ClientModel[] = locations
        .filter((_, idx) => idx !== depotIndex && !depotIndices.includes(idx))
        .map((loc, clientIdx) => {
          const originalIdx = locations.findIndex(l => l === loc);
          return {
            x: Math.round(loc.lon * 10000), // Scale and convert to integer
            y: Math.round(loc.lat * 10000),
            delivery: Math.round(loc.demand),
            pickup: 0,
            service_duration: selectedVariant === 'VRPTW' ? Math.round((serviceTimes[originalIdx] || 0.5) * 60) : 10,
            tw_early: selectedVariant === 'VRPTW' && timeWindows[originalIdx] 
              ? Math.round(timeWindows[originalIdx].earliest * 60) 
              : 0,
            tw_late: selectedVariant === 'VRPTW' && timeWindows[originalIdx] 
              ? Math.round(timeWindows[originalIdx].latest * 60) 
              : 1440,
            prize: selectedVariant === 'PC-VRP' && prizes[originalIdx] 
              ? Math.round(prizes[originalIdx]) 
              : 0,
            required: selectedVariant !== 'PC-VRP'
          };
        });

      // Create depots
      const unifiedDepots: UnifiedDepotModel[] = [];
      if (selectedVariant === 'MDVRP') {
        depotIndices.forEach(idx => {
          const depot = locations[idx];
          unifiedDepots.push({
            x: Math.round(depot.lon * 10000),
            y: Math.round(depot.lat * 10000)
          });
        });
      } else {
        const depot = locations[depotIndex];
        unifiedDepots.push({
          x: Math.round(depot.lon * 10000),
          y: Math.round(depot.lat * 10000)
        });
      }

      // Create vehicle types
      const vehicleTypes: VehicleTypeModel[] = [];
      if (selectedVariant === 'MDVRP') {
        depots.forEach((depot, idx) => {
          vehicleTypes.push({
            num_available: depot.num_vehicles,
            capacity: Math.round(depot.capacity),
            start_depot: idx,
            end_depot: idx,
            fixed_cost: 0,
            tw_early: 0,
            tw_late: 1440,
            max_duration: 480,
            max_distance: 200000
          });
        });
      } else {
        vehicleTypes.push({
          num_available: numVehicles,
          capacity: Math.round(vehicleCapacity),
          start_depot: 0,
          end_depot: 0,
          fixed_cost: 0,
          tw_early: 0,
          tw_late: 1440,
          max_duration: 480,
          max_distance: 200000
        });
      }

      const problemData: VRPProblemData = {
        clients,
        depots: unifiedDepots,
        vehicle_types: vehicleTypes,
        max_runtime: maxRuntime
      };

      const result = await ApiService.solveUnifiedVRP(problemData);
      
      setUnifiedSolution(result);
      setSuccess(`Unified optimization completed: ${result.status}`);
      setTabValue(2); // Switch to results tab
    } catch (err: any) {
      setError(err.message || 'Unified optimization failed');
    } finally {
      setLoading(false);
    }
  };

  const getRouteColor = (index: number) => {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48dbfb', '#ff9ff3', '#54a0ff'];
    return colors[index % colors.length];
  };

  const renderLocationSetup = () => (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <Button
          variant="contained"
          startIcon={<CloudUploadIcon />}
          onClick={() => fileInputRef.current?.click()}
        >
          Upload CSV
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />
        <Button
          variant="outlined"
          onClick={generateSampleData}
        >
          Generate Sample Data
        </Button>
        <Typography variant="body2" color="text.secondary">
          {locations.length} locations loaded
        </Typography>
      </Box>

      {locations.length > 0 && (
        <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell align="right">Latitude</TableCell>
                <TableCell align="right">Longitude</TableCell>
                <TableCell align="right">Demand</TableCell>
                {selectedVariant === 'VRPTW' && (
                  <>
                    <TableCell align="right">Time Window</TableCell>
                    <TableCell align="right">Service Time</TableCell>
                  </>
                )}
                {selectedVariant === 'PC-VRP' && (
                  <TableCell align="right">Prize</TableCell>
                )}
              </TableRow>
            </TableHead>
            <TableBody>
              {locations.map((loc, index) => (
                <TableRow key={index}>
                  <TableCell>{loc.name}</TableCell>
                  <TableCell align="right">{loc.lat.toFixed(4)}</TableCell>
                  <TableCell align="right">{loc.lon.toFixed(4)}</TableCell>
                  <TableCell align="right">{loc.demand}</TableCell>
                  {selectedVariant === 'VRPTW' && timeWindows[index] && (
                    <>
                      <TableCell align="right">
                        {timeWindows[index].earliest.toFixed(1)} - {timeWindows[index].latest.toFixed(1)}
                      </TableCell>
                      <TableCell align="right">{serviceTimes[index]?.toFixed(1)}h</TableCell>
                    </>
                  )}
                  {selectedVariant === 'PC-VRP' && prizes[index] !== undefined && (
                    <TableCell align="right">${prizes[index].toFixed(0)}</TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );

  const renderParameters = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Alert severity="info" icon={<InfoIcon />}>
          <Typography variant="subtitle2">
            {variantInfo[selectedVariant]?.name || selectedVariant}
          </Typography>
          <Typography variant="body2">
            {variantInfo[selectedVariant]?.description}
          </Typography>
        </Alert>
      </Grid>

      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          type="number"
          label="Vehicle Capacity"
          value={vehicleCapacity}
          onChange={(e) => setVehicleCapacity(Number(e.target.value))}
          InputProps={{
            endAdornment: <Typography variant="caption">kg</Typography>
          }}
        />
      </Grid>

      {selectedVariant !== 'MDVRP' && (
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            type="number"
            label="Number of Vehicles"
            value={numVehicles}
            onChange={(e) => setNumVehicles(Number(e.target.value))}
          />
        </Grid>
      )}

      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          type="number"
          label="Max Runtime (seconds)"
          value={maxRuntime}
          onChange={(e) => setMaxRuntime(Number(e.target.value))}
        />
      </Grid>

      {(selectedVariant === 'CVRP' || selectedVariant === 'VRPTW') && (
        <Grid item xs={12} sm={6} md={4}>
          <FormControl fullWidth>
            <InputLabel>Depot Location</InputLabel>
            <Select
              value={depotIndex}
              onChange={(e) => setDepotIndex(Number(e.target.value))}
            >
              {locations.map((loc, index) => (
                <MenuItem key={index} value={index}>
                  {loc.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      )}

      {selectedVariant === 'PC-VRP' && (
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            type="number"
            label="Minimum Prize to Collect"
            value={minPrize}
            onChange={(e) => setMinPrize(Number(e.target.value))}
            InputProps={{
              startAdornment: <Typography variant="caption">$</Typography>
            }}
          />
        </Grid>
      )}

      {selectedVariant === 'MDVRP' && (
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Multi-Depot Configuration
          </Typography>
          <Box sx={{ pl: 2 }}>
            {depots.map((depot, index) => (
              <Box key={index} sx={{ mb: 2 }}>
                <Typography variant="body2">
                  Depot {index + 1}: Location Index {depotIndices[index]}, 
                  Capacity: {depot.capacity}, Vehicles: {depot.num_vehicles}
                </Typography>
              </Box>
            ))}
          </Box>
        </Grid>
      )}

      {selectedVariant === 'PDVRP' && (
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom>
            Pickup-Delivery Pairs
          </Typography>
          <Box sx={{ pl: 2 }}>
            {pickupDeliveryPairs.map((pair, index) => (
              <Box key={index} sx={{ mb: 1 }}>
                <Typography variant="body2">
                  Pair {index + 1}: Pickup at {locations[pair.pickup_location_idx]?.name || pair.pickup_location_idx} → Delivery at {locations[pair.delivery_location_idx]?.name || pair.delivery_location_idx} 
                  (Demand: {pair.demand})
                </Typography>
              </Box>
            ))}
          </Box>
        </Grid>
      )}

      <Grid item xs={12}>
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            color="secondary"
            startIcon={<PlayIcon />}
            onClick={solveUnifiedVRP}
            disabled={loading || locations.length === 0}
          >
            {loading ? 'Optimizing...' : 'Test Unified API'}
          </Button>
          <Button
            variant="contained"
            color="primary"
            startIcon={<PlayIcon />}
            onClick={solveVRP}
            disabled={loading || locations.length === 0}
          >
            {loading ? 'Optimizing...' : 'Optimize Routes'}
          </Button>
        </Box>
      </Grid>
    </Grid>
  );

  const renderResults = () => {
    if (!solution && !unifiedSolution) {
      return (
        <Box sx={{ textAlign: 'center', py: 5 }}>
          <Typography variant="body1" color="text.secondary">
            No optimization results yet. Configure parameters and run optimization.
          </Typography>
        </Box>
      );
    }

    // Use unified solution if available, otherwise use regular solution
    const currentSolution = unifiedSolution || solution;
    const isUnified = !!unifiedSolution;
    
    if (!currentSolution) {
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
              <Typography variant="h6">
                {isUnified ? currentSolution.routes.length : (currentSolution as any).num_vehicles_used}
              </Typography>
              <Typography variant="body2" color="text.secondary">Vehicles Used</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TimelineIcon color="secondary" sx={{ fontSize: 40 }} />
              <Typography variant="h6">
                {isUnified 
                  ? ((currentSolution as any).routes.reduce((sum: number, r: any) => sum + (r.distance || 0), 0) / 1000).toFixed(1) + ' km'
                  : (currentSolution as any).total_distance.toFixed(1) + ' km'
                }
              </Typography>
              <Typography variant="body2" color="text.secondary">Total Distance</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <TimeIcon color="success" sx={{ fontSize: 40 }} />
              <Typography variant="h6">{(currentSolution as any).computation_time?.toFixed(2) || '0.00'}s</Typography>
              <Typography variant="body2" color="text.secondary">Computation Time</Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Chip 
                label={`${currentSolution.status.toUpperCase()} ${isUnified ? '(Unified)' : ''}`}
                color={currentSolution.status === 'optimal' ? 'success' : 'warning'}
                sx={{ mt: 1 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Solution Status
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        <FormControlLabel
          control={<Switch checked={showMap} onChange={(e) => setShowMap(e.target.checked)} />}
          label="Show Map Visualization"
        />

        {showMap && locations.length > 0 && (
          <Paper sx={{ mb: 3, height: 500 }}>
            <MapContainer
              center={[locations[0].lat, locations[0].lon]}
              zoom={11}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              
              {/* Draw locations */}
              {locations.map((loc, index) => (
                <Marker
                  key={index}
                  position={[loc.lat, loc.lon]}
                  icon={L.divIcon({
                    className: 'custom-div-icon',
                    html: `<div style="background-color: ${
                      depotIndices.includes(index) || index === depotIndex ? '#FF0000' : '#0066CC'
                    }; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; border: 2px solid white;">${
                      depotIndices.includes(index) || index === depotIndex ? 'D' : index
                    }</div>`,
                    iconSize: [30, 30],
                    iconAnchor: [15, 15],
                  })}
                >
                  <Popup>
                    <strong>{loc.name}</strong><br />
                    Demand: {loc.demand}<br />
                    {selectedVariant === 'PC-VRP' && prizes[index] && `Prize: $${prizes[index]}`}
                  </Popup>
                </Marker>
              ))}
              
              {/* Draw routes */}
              {currentSolution.routes.map((route, routeIndex) => {
                let routePositions: [number, number][] = [];
                
                if (isUnified) {
                  // For unified solution, map client indices to locations
                  const unifiedRoute = route as any; // Type assertion for unified route
                  const depot = locations.find((_, idx) => 
                    (selectedVariant === 'MDVRP' ? depotIndices.includes(idx) : idx === depotIndex)
                  );
                  
                  if (depot) {
                    routePositions.push([depot.lat, depot.lon]);
                    unifiedRoute.clients.forEach((clientIdx: number) => {
                      const clientLocation = locations.filter((_, idx) => 
                        idx !== depotIndex && !depotIndices.includes(idx)
                      )[clientIdx];
                      if (clientLocation) {
                        routePositions.push([clientLocation.lat, clientLocation.lon]);
                      }
                    });
                    routePositions.push([depot.lat, depot.lon]);
                  }
                } else {
                  // For regular solution
                  routePositions = (route as any).sequence.map((idx: number) => [locations[idx].lat, locations[idx].lon] as [number, number]);
                }
                
                return (
                  <Polyline
                    key={routeIndex}
                    positions={routePositions}
                    color={getRouteColor(routeIndex)}
                    weight={3}
                    opacity={0.8}
                  />
                );
              })}
            </MapContainer>
          </Paper>
        )}

        <Typography variant="h6" gutterBottom>
          Route Details
        </Typography>
        
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell />
                <TableCell>Route ID</TableCell>
                <TableCell>Stops</TableCell>
                <TableCell align="right">Distance (km)</TableCell>
                <TableCell align="right">Load</TableCell>
                <TableCell align="right">Utilization</TableCell>
                {selectedVariant === 'PC-VRP' && <TableCell align="right">Prize</TableCell>}
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {currentSolution.routes.map((route, index) => {
                const routeId = isUnified ? index + 1 : (route as any).route_id;
                const numStops = isUnified ? (route as any).clients?.length || 0 : (route as any).num_stops;
                const distance = isUnified ? ((route as any).distance / 1000) : (route as any).distance;
                const demand = isUnified ? (route as any).demand_served : (route as any).total_demand;
                
                return (
                  <React.Fragment key={routeId}>
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
                          label={`Route ${routeId}`}
                          size="small"
                          sx={{ backgroundColor: getRouteColor(index), color: 'white' }}
                        />
                      </TableCell>
                      <TableCell>{numStops}</TableCell>
                      <TableCell align="right">{distance.toFixed(1)}</TableCell>
                      <TableCell align="right">{demand}</TableCell>
                      <TableCell align="right">
                        {isUnified ? 'N/A' : (((route as any).capacity_utilization || 0) * 100).toFixed(1) + '%'}
                      </TableCell>
                      {selectedVariant === 'PC-VRP' && (
                        <TableCell align="right">${(route as any).total_prize?.toFixed(0) || 0}</TableCell>
                      )}
                      <TableCell>
                        <Chip label="Active" color="success" size="small" />
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
                              {isUnified ? (
                                <>
                                  <Chip
                                    icon={<StoreIcon />}
                                    label="Depot"
                                    size="small"
                                    variant="filled"
                                  />
                                  {(route as any).clients?.map((clientIdx: number, locIndex: number) => {
                                    const clientLocation = locations.find((_, idx) => 
                                      idx !== depotIndex && !depotIndices.includes(idx)
                                    );
                                    return (
                                      <React.Fragment key={locIndex}>
                                        <Chip
                                          icon={<LocationIcon />}
                                          label={clientLocation?.name || `Client ${clientIdx}`}
                                          size="small"
                                          variant="outlined"
                                        />
                                        {locIndex < ((route as any).clients?.length || 0) - 1 && ' → '}
                                      </React.Fragment>
                                    );
                                  })}
                                  <Chip
                                    icon={<StoreIcon />}
                                    label="Depot"
                                    size="small"
                                    variant="filled"
                                  />
                                </>
                              ) : (
                                (route as any).locations?.map((locName: string, locIndex: number) => (
                                  <React.Fragment key={locIndex}>
                                    <Chip
                                      icon={locIndex === 0 || locIndex === ((route as any).locations?.length || 0) - 1 ? <StoreIcon /> : <LocationIcon />}
                                      label={locName}
                                      size="small"
                                      variant={locIndex === 0 || locIndex === ((route as any).locations?.length || 0) - 1 ? "filled" : "outlined"}
                                    />
                                    {locIndex < ((route as any).locations?.length || 0) - 1 && ' → '}
                                  </React.Fragment>
                                ))
                              )}
                            </Box>
                            {(route as any).arrival_times && (
                              <Typography variant="body2" color="text.secondary">
                                Estimated arrival times available
                              </Typography>
                            )}
                          </Box>
                        </Collapse>
                      </TableCell>
                    </TableRow>
                  </React.Fragment>
                );
              })}
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
          Advanced Vehicle Routing (PyVRP)
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Solve complex vehicle routing problems using state-of-the-art algorithms
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
          <Box sx={{ mb: 3 }}>
            <FormControl fullWidth>
              <InputLabel>VRP Variant</InputLabel>
              <Select
                value={selectedVariant}
                onChange={(e) => setSelectedVariant(e.target.value)}
                startAdornment={
                  <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
                    {selectedVariant === 'CVRP' && <TruckIcon sx={{ mr: 1 }} />}
                    {selectedVariant === 'VRPTW' && <TimeIcon sx={{ mr: 1 }} />}
                    {selectedVariant === 'MDVRP' && <StoreIcon sx={{ mr: 1 }} />}
                    {selectedVariant === 'PDVRP' && <SwapIcon sx={{ mr: 1 }} />}
                    {selectedVariant === 'PC-VRP' && <PrizeIcon sx={{ mr: 1 }} />}
                  </Box>
                }
              >
                <MenuItem value="CVRP">Capacitated VRP (CVRP)</MenuItem>
                <MenuItem value="VRPTW">VRP with Time Windows (VRPTW)</MenuItem>
                <MenuItem value="MDVRP">Multi-Depot VRP (MDVRP)</MenuItem>
                <MenuItem value="PDVRP">Pickup & Delivery VRP (PDVRP)</MenuItem>
                <MenuItem value="PC-VRP">Prize-Collecting VRP (PC-VRP)</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
              <Tab label="Data Setup" />
              <Tab label="Parameters" />
              <Tab label="Results" />
              <Tab label="Compare" disabled={solutions.length < 2} />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            {renderLocationSetup()}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {renderParameters()}
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            {renderResults()}
          </TabPanel>

          <TabPanel value={tabValue} index={3}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Solution Comparison
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Compare multiple optimization runs to find the best solution
              </Typography>
              {/* Comparison functionality would go here */}
            </Box>
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
            <Typography>Optimizing routes...</Typography>
          </Paper>
        </Box>
      )}
    </Box>
  );
};

export default AdvancedVRP;