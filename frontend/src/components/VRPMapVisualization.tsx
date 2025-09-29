import React, { useEffect, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControlLabel,
  Switch,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  LocalShipping as TruckIcon,
  Store as StoreIcon,
  LocationOn as LocationIcon,
  RestoreFromTrash as ReloadIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Polyline, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';

// Fix for default markers in React-Leaflet
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface VRPMapVisualizationProps {
  solution: any;
  clients: any[];
  depots: any[];
  vehicleTypes: any[];
  clientGroups?: any[];
}

const VRPMapVisualization: React.FC<VRPMapVisualizationProps> = ({
  solution,
  clients,
  depots,
  vehicleTypes,
  clientGroups = []
}) => {
  const [showRoutes, setShowRoutes] = useState(true);
  const [showTimeWindows, setShowTimeWindows] = useState(false);
  const [showClientGroups, setShowClientGroups] = useState(false);
  const [showReloadDepots, setShowReloadDepots] = useState(true);
  const [selectedRouteIndex, setSelectedRouteIndex] = useState<number | null>(null);
  const [animationSpeed, setAnimationSpeed] = useState(50);
  const [currentAnimationStep, setCurrentAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Calculate center point for map
  const getMapCenter = (): [number, number] => {
    if (clients.length === 0 && depots.length === 0) return [35.6762, 139.6503];
    
    const allPoints = [...clients, ...depots];
    const avgLat = allPoints.reduce((sum, point) => sum + (point.y || point.lat || 0) / 10000, 0) / allPoints.length;
    const avgLon = allPoints.reduce((sum, point) => sum + (point.x || point.lon || 0) / 10000, 0) / allPoints.length;
    
    return [avgLat, avgLon];
  };

  const getRouteColor = (index: number) => {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48dbfb', '#ff9ff3', '#54a0ff'];
    return colors[index % colors.length];
  };

  const getClientGroupColor = (groupId: string) => {
    const colors = ['#FFE0E6', '#E0F7FA', '#F3E5F5', '#E8F5E8', '#FFF3E0'];
    const hash = groupId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[hash % colors.length];
  };

  // Convert PyVRP coordinates to lat/lon (10000x scale for Kanto region)
  const coordsToLatLon = (x: number, y: number): [number, number] => [
    y / 10000,
    x / 10000
  ];

  // Create custom icons
  const createCustomIcon = (color: string, symbol: string, size: number = 30) => {
    return L.divIcon({
      className: 'custom-div-icon',
      html: `<div style="
        background-color: ${color}; 
        width: ${size}px; 
        height: ${size}px; 
        border-radius: 50%; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        color: white; 
        font-weight: bold; 
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      ">${symbol}</div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  };

  // Animate route progression
  useEffect(() => {
    if (!isAnimating || !solution?.routes) return;

    const interval = setInterval(() => {
      setCurrentAnimationStep(prev => {
        const maxSteps = solution.routes.reduce((max: number, route: any) => 
          Math.max(max, (route.clients?.length || 0) + 2), 0);
        return prev >= maxSteps ? 0 : prev + 1;
      });
    }, 1000 / (animationSpeed / 10));

    return () => clearInterval(interval);
  }, [isAnimating, animationSpeed, solution]);

  const renderClientMarkers = () => {
    return clients.map((client, index) => {
      const [lat, lon] = coordsToLatLon(client.x, client.y);
      const clientGroup = clientGroups.find(group => group.client_indices.includes(index));
      
      return (
        <Marker
          key={`client-${index}`}
          position={[lat, lon]}
          icon={createCustomIcon(
            clientGroup ? getClientGroupColor(clientGroup.group_id) : '#0066CC',
            index.toString()
          )}
        >
          <Popup>
            <Box>
              <Typography variant="subtitle2">Client {index}</Typography>
              <Typography variant="body2">
                Delivery: {Array.isArray(client.delivery) ? client.delivery.join(', ') : client.delivery}
              </Typography>
              {client.tw_early !== undefined && client.tw_late !== undefined && (
                <Typography variant="body2">
                  Time Window: {Math.floor(client.tw_early / 60)}:00 - {Math.floor(client.tw_late / 60)}:00
                </Typography>
              )}
              {client.service_duration && (
                <Typography variant="body2">
                  Service Time: {client.service_duration} min
                </Typography>
              )}
              {clientGroup && (
                <Chip
                  label={clientGroup.group_id}
                  size="small"
                  color="primary"
                  style={{ marginTop: 4 }}
                />
              )}
            </Box>
          </Popup>
        </Marker>
      );
    });
  };

  const renderDepotMarkers = () => {
    return depots.map((depot, index) => {
      const [lat, lon] = coordsToLatLon(depot.x, depot.y);
      const isReloadDepot = depot.is_reload_depot;
      
      return (
        <Marker
          key={`depot-${index}`}
          position={[lat, lon]}
          icon={createCustomIcon(
            isReloadDepot ? '#FF4444' : '#FF0000',
            'D',
            isReloadDepot ? 35 : 30
          )}
        >
          <Popup>
            <Box>
              <Typography variant="subtitle2">Depot {index}</Typography>
              <Typography variant="body2">Type: {depot.depot_type || 'main'}</Typography>
              {depot.tw_early !== undefined && depot.tw_late !== undefined && (
                <Typography variant="body2">
                  Hours: {Math.floor(depot.tw_early / 60)}:00 - {Math.floor(depot.tw_late / 60)}:00
                </Typography>
              )}
              {isReloadDepot && (
                <Box>
                  <Chip
                    icon={<ReloadIcon />}
                    label="Reload Available"
                    size="small"
                    color="primary"
                    style={{ marginTop: 4 }}
                  />
                  {depot.reload_time && (
                    <Typography variant="body2">
                      Reload Time: {depot.reload_time} min
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
          </Popup>
        </Marker>
      );
    });
  };

  const renderTimeWindowCircles = () => {
    if (!showTimeWindows) return null;

    return clients.map((client, index) => {
      if (!client.tw_early || !client.tw_late) return null;
      
      const [lat, lon] = coordsToLatLon(client.x, client.y);
      const timeWindowDuration = client.tw_late - client.tw_early;
      const radius = Math.max(50, timeWindowDuration / 10); // Scale radius based on time window

      return (
        <Circle
          key={`timewindow-${index}`}
          center={[lat, lon]}
          radius={radius}
          pathOptions={{
            color: '#FFA500',
            fillColor: '#FFA500',
            fillOpacity: 0.2,
            weight: 2,
            dashArray: '5, 5'
          }}
        />
      );
    });
  };

  const renderRoutePolylines = () => {
    if (!showRoutes || !solution?.routes) return null;

    return solution.routes.map((route: any, routeIndex: number) => {
      if (selectedRouteIndex !== null && selectedRouteIndex !== routeIndex) {
        return null; // Hide non-selected routes if one is selected
      }

      const routePositions: [number, number][] = [];
      
      // Add depot start
      if (route.start_depot !== undefined && depots[route.start_depot]) {
        const depot = depots[route.start_depot];
        routePositions.push(coordsToLatLon(depot.x, depot.y));
      }

      // Add client positions
      if (route.clients) {
        route.clients.forEach((clientIdx: number) => {
          if (clients[clientIdx]) {
            const client = clients[clientIdx];
            routePositions.push(coordsToLatLon(client.x, client.y));
          }
        });
      }

      // Add depot end
      if (route.end_depot !== undefined && depots[route.end_depot]) {
        const depot = depots[route.end_depot];
        routePositions.push(coordsToLatLon(depot.x, depot.y));
      }

      // Animation: only show up to current step
      let animatedPositions = routePositions;
      if (isAnimating && selectedRouteIndex === routeIndex) {
        animatedPositions = routePositions.slice(0, currentAnimationStep + 1);
      }

      return (
        <Polyline
          key={`route-${routeIndex}`}
          positions={animatedPositions}
          color={getRouteColor(routeIndex)}
          weight={selectedRouteIndex === routeIndex ? 5 : 3}
          opacity={selectedRouteIndex === routeIndex ? 1 : 0.7}
        />
      );
    });
  };

  const renderClientGroupAreas = () => {
    if (!showClientGroups || clientGroups.length === 0) return null;

    return clientGroups.map((group) => {
      const groupPositions: [number, number][] = group.client_indices
        .filter((idx: number) => clients[idx])
        .map((idx: number) => {
          const client = clients[idx];
          return coordsToLatLon(client.x, client.y);
        });

      if (groupPositions.length < 2) return null;

      // Create a convex hull or simple polygon around group clients
      return (
        <Polyline
          key={`group-${group.group_id}`}
          positions={[...groupPositions, groupPositions[0]]} // Close the polygon
          color={getClientGroupColor(group.group_id)}
          weight={2}
          dashArray="10, 5"
          opacity={0.8}
          fillColor={getClientGroupColor(group.group_id)}
          fillOpacity={0.1}
        />
      );
    });
  };

  return (
    <Box>
      {/* Map Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>Map Controls</Typography>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
          <FormControlLabel
            control={<Switch checked={showRoutes} onChange={(e) => setShowRoutes(e.target.checked)} />}
            label="Show Routes"
          />
          <FormControlLabel
            control={<Switch checked={showTimeWindows} onChange={(e) => setShowTimeWindows(e.target.checked)} />}
            label="Show Time Windows"
          />
          <FormControlLabel
            control={<Switch checked={showClientGroups} onChange={(e) => setShowClientGroups(e.target.checked)} />}
            label="Show Client Groups"
          />
          <FormControlLabel
            control={<Switch checked={showReloadDepots} onChange={(e) => setShowReloadDepots(e.target.checked)} />}
            label="Highlight Reload Depots"
          />
        </Box>

        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Focus Route</InputLabel>
            <Select
              value={selectedRouteIndex ?? ''}
              onChange={(e) => setSelectedRouteIndex(e.target.value === '' ? null : Number(e.target.value))}
              displayEmpty
            >
              <MenuItem value="">All Routes</MenuItem>
              {solution?.routes?.map((_: any, index: number) => (
                <MenuItem key={index} value={index}>Route {index + 1}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Switch 
                checked={isAnimating} 
                onChange={(e) => {
                  setIsAnimating(e.target.checked);
                  if (!e.target.checked) setCurrentAnimationStep(0);
                }} 
              />
            }
            label="Animate"
          />

          {isAnimating && (
            <Box sx={{ width: 200 }}>
              <Typography variant="caption">Animation Speed</Typography>
              <Slider
                value={animationSpeed}
                onChange={(_, value) => setAnimationSpeed(value as number)}
                min={10}
                max={100}
                size="small"
              />
            </Box>
          )}
        </Box>
      </Paper>

      {/* Map */}
      <Paper sx={{ height: 600, mb: 2 }}>
        <MapContainer
          center={getMapCenter()}
          zoom={10}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {renderDepotMarkers()}
          {renderClientMarkers()}
          {renderTimeWindowCircles()}
          {renderClientGroupAreas()}
          {renderRoutePolylines()}
        </MapContainer>
      </Paper>

      {/* Route Details */}
      {solution?.routes && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Route Details</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Route</TableCell>
                  <TableCell>Vehicle</TableCell>
                  <TableCell>Clients</TableCell>
                  <TableCell>Distance</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Features</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {solution.routes.map((route: any, index: number) => (
                  <TableRow 
                    key={index}
                    onClick={() => setSelectedRouteIndex(selectedRouteIndex === index ? null : index)}
                    sx={{ 
                      cursor: 'pointer',
                      backgroundColor: selectedRouteIndex === index ? 'action.selected' : 'inherit'
                    }}
                  >
                    <TableCell>
                      <Chip
                        label={`Route ${index + 1}`}
                        size="small"
                        sx={{ backgroundColor: getRouteColor(index), color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>Type {route.vehicle_type || 0}</TableCell>
                    <TableCell>{route.clients?.length || 0}</TableCell>
                    <TableCell>{((route.distance || 0) / 1000).toFixed(1)} km</TableCell>
                    <TableCell>{Math.round((route.duration || 0) / 60)} min</TableCell>
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
                ))}
              </TableBody>
            </Table>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
};

export default VRPMapVisualization;