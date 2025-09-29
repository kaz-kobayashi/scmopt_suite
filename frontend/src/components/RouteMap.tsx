import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Box, Typography, FormControl, InputLabel, Select, MenuItem, Chip, OutlinedInput } from '@mui/material';

// Fix Leaflet icon issues
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface Location {
  name: string;
  lat: number;
  lon: number;
  demand?: number;
}

interface Route {
  route_id: number;
  locations: string[];
  total_distance: number;
  total_demand?: number;
}

interface RouteMapProps {
  locations: Location[];
  routes?: Route[];
  depot?: string;
}

const RouteMap: React.FC<RouteMapProps> = ({ locations, routes, depot = 'Tokyo_DC' }) => {
  const mapRef = useRef<L.Map | null>(null);
  const [selectedRoutes, setSelectedRoutes] = useState<number[]>([]);

  // Define colors for different routes
  const routeColors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

  // Find depot location
  const depotLocation = locations.find(loc => loc.name === depot);
  const centerLat = depotLocation?.lat || 35.6812;
  const centerLon = depotLocation?.lon || 139.7671;

  // Create location lookup map
  const locationMap = new Map<string, Location>();
  locations.forEach(loc => {
    locationMap.set(loc.name, loc);
  });

  // Custom depot icon
  const depotIcon = L.divIcon({
    html: `<div style="background-color: #ff0000; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    className: 'depot-marker'
  });

  // Initialize with all routes selected
  useEffect(() => {
    if (routes && routes.length > 0) {
      setSelectedRoutes(routes.map(route => route.route_id));
    }
  }, [routes]);

  // Fit map to show all locations
  useEffect(() => {
    if (mapRef.current && locations.length > 0) {
      const bounds = L.latLngBounds(locations.map(loc => [loc.lat, loc.lon]));
      mapRef.current.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [locations]);

  // Handle route selection change
  const handleRouteSelectionChange = (event: any) => {
    const value = event.target.value;
    setSelectedRoutes(typeof value === 'string' ? value.split(',').map(Number) : value);
  };

  // Filter routes based on selection
  const visibleRoutes = routes?.filter(route => selectedRoutes.includes(route.route_id)) || [];
  
  // Calculate statistics for visible routes
  const visibleStats = visibleRoutes.length > 0 ? {
    totalDistance: visibleRoutes.reduce((sum, route) => sum + route.total_distance, 0),
    totalCustomers: visibleRoutes.reduce((sum, route) => sum + (route.locations.length - 2), 0),
    averageDistance: visibleRoutes.reduce((sum, route) => sum + route.total_distance, 0) / visibleRoutes.length
  } : null;

  return (
    <Box sx={{ width: '100%' }}>
      {/* Route Selection Controls */}
      {routes && routes.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <FormControl fullWidth size="small">
            <InputLabel id="route-select-label">表示するルート</InputLabel>
            <Select
              labelId="route-select-label"
              multiple
              value={selectedRoutes}
              onChange={handleRouteSelectionChange}
              input={<OutlinedInput id="select-multiple-chip" label="表示するルート" />}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => {
                    const route = routes.find(r => r.route_id === value);
                    const routeIndex = routes.findIndex(r => r.route_id === value);
                    const color = routeColors[routeIndex % routeColors.length];
                    return (
                      <Chip 
                        key={value} 
                        label={`ルート ${value}`}
                        size="small"
                        style={{ backgroundColor: color, color: 'white' }}
                      />
                    );
                  })}
                </Box>
              )}
              MenuProps={{
                PaperProps: {
                  style: {
                    maxHeight: 224,
                    width: 250,
                  },
                },
              }}
            >
              {routes.map((route, index) => (
                <MenuItem key={route.route_id} value={route.route_id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    <Box
                      sx={{
                        width: 16,
                        height: 16,
                        backgroundColor: routeColors[index % routeColors.length],
                        borderRadius: '50%',
                        mr: 1,
                      }}
                    />
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="body2">
                        ルート {route.route_id}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {route.total_distance.toFixed(1)} km, {route.locations.length - 2} 顧客
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
            <Typography 
              variant="caption" 
              component="span" 
              sx={{ cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
              onClick={() => setSelectedRoutes(routes.map(r => r.route_id))}
            >
              すべて選択
            </Typography>
            <Typography 
              variant="caption" 
              component="span" 
              sx={{ cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
              onClick={() => setSelectedRoutes([])}
            >
              すべて解除
            </Typography>
            <Typography 
              variant="caption" 
              component="span" 
              sx={{ cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
              onClick={() => {
                // Select top 5 routes by distance
                const top5Routes = routes
                  .sort((a, b) => b.total_distance - a.total_distance)
                  .slice(0, 5)
                  .map(r => r.route_id);
                setSelectedRoutes(top5Routes);
              }}
            >
              長距離5ルート
            </Typography>
            <Typography 
              variant="caption" 
              component="span" 
              sx={{ cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
              onClick={() => {
                // Select routes with most customers
                const busyRoutes = routes
                  .sort((a, b) => (b.locations.length - 2) - (a.locations.length - 2))
                  .slice(0, 5)
                  .map(r => r.route_id);
                setSelectedRoutes(busyRoutes);
              }}
            >
              多顧客5ルート
            </Typography>
          </Box>
          
          {/* Statistics for selected routes */}
          {visibleStats && (
            <Box sx={{ mt: 2, p: 1, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
              <Typography variant="caption" color="text.secondary" display="block">
                選択ルート統計:
              </Typography>
              <Typography variant="caption" display="block">
                総距離: {visibleStats.totalDistance.toFixed(1)} km | 
                総顧客数: {visibleStats.totalCustomers} | 
                平均距離: {visibleStats.averageDistance.toFixed(1)} km
              </Typography>
            </Box>
          )}
        </Box>
      )}

      <Box sx={{ height: 500, width: '100%', position: 'relative' }}>
        <MapContainer
          center={[centerLat, centerLon]}
          zoom={10}
          style={{ height: '100%', width: '100%' }}
          ref={mapRef}
        >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Draw depot */}
        {depotLocation && (
          <Marker position={[depotLocation.lat, depotLocation.lon]} icon={depotIcon}>
            <Popup>
              <Typography variant="subtitle2">{depotLocation.name}</Typography>
              <Typography variant="body2">Distribution Center</Typography>
            </Popup>
          </Marker>
        )}

        {/* Draw route polylines if routes are provided */}
        {visibleRoutes && visibleRoutes.map((route) => {
          const routeIndex = routes?.findIndex(r => r.route_id === route.route_id) || 0;
          const routeColor = routeColors[routeIndex % routeColors.length];
          const routeCoordinates: [number, number][] = [];

          // Build route coordinates
          route.locations.forEach(locName => {
            const loc = locationMap.get(locName);
            if (loc) {
              routeCoordinates.push([loc.lat, loc.lon]);
            }
          });

          return (
            <React.Fragment key={route.route_id}>
              {/* Draw route line */}
              {routeCoordinates.length > 1 && (
                <Polyline
                  positions={routeCoordinates}
                  color={routeColor}
                  weight={3}
                  opacity={0.7}
                  dashArray="10, 10"
                />
              )}

              {/* Draw customer markers for this route */}
              {route.locations.map((locName, locIndex) => {
                if (locName === depot) return null;
                const loc = locationMap.get(locName);
                if (!loc) return null;

                return (
                  <CircleMarker
                    key={`${route.route_id}-${locIndex}`}
                    center={[loc.lat, loc.lon]}
                    radius={8}
                    fillColor={routeColor}
                    fillOpacity={0.8}
                    color="white"
                    weight={2}
                  >
                    <Popup>
                      <Typography variant="subtitle2">{loc.name}</Typography>
                      <Typography variant="body2">Demand: {loc.demand || 0} kg</Typography>
                      <Typography variant="body2" color={routeColor}>
                        Route {route.route_id}
                      </Typography>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </React.Fragment>
          );
        })}

        {/* Draw all locations if no routes are provided or no routes are selected */}
        {(!routes || selectedRoutes.length === 0) && locations.map((loc) => {
          if (loc.name === depot) return null;
          return (
            <CircleMarker
              key={loc.name}
              center={[loc.lat, loc.lon]}
              radius={6}
              fillColor="#3388ff"
              fillOpacity={0.6}
              color="white"
              weight={2}
            >
              <Popup>
                <Typography variant="subtitle2">{loc.name}</Typography>
                <Typography variant="body2">Demand: {loc.demand || 0} kg</Typography>
              </Popup>
            </CircleMarker>
          );
        })}
        </MapContainer>

        {/* Legend for visible routes */}
        {visibleRoutes && visibleRoutes.length > 0 && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            backgroundColor: 'white',
            padding: 2,
            borderRadius: 1,
            boxShadow: 2,
            zIndex: 1000,
            maxWidth: 200,
            maxHeight: 300,
            overflow: 'auto',
          }}
        >
          <Typography variant="subtitle2" gutterBottom>
            表示中のルート ({visibleRoutes.length}/{routes?.length || 0})
          </Typography>
          {visibleRoutes.map((route) => {
            const routeIndex = routes?.findIndex(r => r.route_id === route.route_id) || 0;
            return (
              <Box key={route.route_id} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                <Box
                  sx={{
                    width: 20,
                    height: 3,
                    backgroundColor: routeColors[routeIndex % routeColors.length],
                    mr: 1,
                  }}
                />
                <Typography variant="caption">
                  ルート {route.route_id}: {route.total_distance.toFixed(1)} km
                </Typography>
              </Box>
            );
          })}
        </Box>
        )}
      </Box>
    </Box>
  );
};

export default RouteMap;