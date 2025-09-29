import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Box, Typography, Card, CardContent, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';

// Fix for default markers in React Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Custom icons for different node types
const dcIcon = new L.Icon({
  iconUrl: 'data:image/svg+xml;base64,' + btoa(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#2196f3" width="24" height="24">
      <circle cx="12" cy="12" r="8" fill="#2196f3" stroke="#1976d2" stroke-width="2"/>
      <text x="12" y="16" text-anchor="middle" fill="white" font-size="10" font-weight="bold">DC</text>
    </svg>
  `),
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12],
});

// Colors for different edge types
const edgeColors = {
  vehicle: '#ff9800', // Orange for vehicle routes
  path: '#f44336',    // Red for selected paths
};

interface SNDMapVisualizationProps {
  visualizationData: {
    nodes: Array<{
      id: string;
      name: string;
      lat: number;
      lon: number;
      node_type: string;
    }>;
    edges: Array<{
      from_node: string;
      to_node: string;
      edge_type: string;
      weight: number;
      color: string;
      width: number;
    }>;
    center_lat: number;
    center_lon: number;
    zoom_level: number;
  };
  onDestinationFilter?: (destination: string | null) => void;
}

// Component to handle map bounds fitting
const FitMapBounds: React.FC<{ nodes: any[] }> = ({ nodes }) => {
  const map = useMap();
  
  useEffect(() => {
    if (nodes.length > 0) {
      const bounds = L.latLngBounds(nodes.map(node => [node.lat, node.lon]));
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [map, nodes]);
  
  return null;
};

const SNDMapVisualization: React.FC<SNDMapVisualizationProps> = ({ 
  visualizationData, 
  onDestinationFilter 
}) => {
  const [selectedDestination, setSelectedDestination] = React.useState<string>('');

  if (!visualizationData || !visualizationData.nodes || visualizationData.nodes.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            ネットワーク地図
          </Typography>
          <Box sx={{ height: 400, bgcolor: 'grey.100', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography variant="body1" color="text.secondary">
              可視化データがありません。先に可視化を生成してください。
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  const handleDestinationChange = (event: SelectChangeEvent) => {
    const destination = event.target.value;
    setSelectedDestination(destination);
    if (onDestinationFilter) {
      onDestinationFilter(destination === '' ? null : destination);
    }
  };

  // Get unique destinations for filter
  const destinations = Array.from(new Set(visualizationData.nodes.map(node => node.name)));

  // Create node lookup for edge rendering
  const nodeMap = new Map(visualizationData.nodes.map(node => [node.id, node]));

  // Group edges by type for rendering
  const vehicleEdges = visualizationData.edges.filter(edge => edge.edge_type === 'vehicle');
  const pathEdges = visualizationData.edges.filter(edge => edge.edge_type === 'path');

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            サービスネットワーク地図
          </Typography>
          
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>目的地フィルター</InputLabel>
            <Select
              value={selectedDestination}
              onChange={handleDestinationChange}
              label="目的地フィルター"
            >
              <MenuItem value="">すべて表示</MenuItem>
              {destinations.map((destination) => (
                <MenuItem key={destination} value={destination}>
                  {destination}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        <Box sx={{ height: 500, border: 1, borderColor: 'divider', borderRadius: 1 }}>
          <MapContainer
            center={[visualizationData.center_lat, visualizationData.center_lon]}
            zoom={visualizationData.zoom_level}
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom={true}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            
            <FitMapBounds nodes={visualizationData.nodes} />

            {/* Render vehicle routes (base network) */}
            {vehicleEdges.map((edge, index) => {
              const fromNode = nodeMap.get(edge.from_node);
              const toNode = nodeMap.get(edge.to_node);
              
              if (!fromNode || !toNode) return null;
              
              return (
                <Polyline
                  key={`vehicle-${index}`}
                  positions={[
                    [fromNode.lat, fromNode.lon],
                    [toNode.lat, toNode.lon]
                  ]}
                  color={edge.color}
                  weight={edge.width}
                  opacity={0.7}
                />
              );
            })}

            {/* Render selected paths */}
            {pathEdges.map((edge, index) => {
              const fromNode = nodeMap.get(edge.from_node);
              const toNode = nodeMap.get(edge.to_node);
              
              if (!fromNode || !toNode) return null;
              
              return (
                <Polyline
                  key={`path-${index}`}
                  positions={[
                    [fromNode.lat, fromNode.lon],
                    [toNode.lat, toNode.lon]
                  ]}
                  color={edge.color}
                  weight={edge.width}
                  opacity={0.9}
                  dashArray="5, 5"
                />
              );
            })}

            {/* Render DC nodes */}
            {visualizationData.nodes.map((node) => (
              <Marker
                key={node.id}
                position={[node.lat, node.lon]}
                icon={dcIcon}
              >
                <Popup>
                  <Box>
                    <Typography variant="subtitle2" fontWeight="bold">
                      {node.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      配送センター (DC)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      座標: ({node.lat.toFixed(4)}, {node.lon.toFixed(4)})
                    </Typography>
                  </Box>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 20, height: 3, bgcolor: edgeColors.vehicle, opacity: 0.7 }} />
            <Typography variant="body2">車両ルート (ベースネットワーク)</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ 
              width: 20, 
              height: 3, 
              bgcolor: edgeColors.path, 
              opacity: 0.9,
              borderStyle: 'dashed',
              borderWidth: '1px 0',
              borderColor: edgeColors.path
            }} />
            <Typography variant="body2">選択されたパス</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ 
              width: 16, 
              height: 16, 
              bgcolor: '#2196f3', 
              borderRadius: '50%',
              border: '2px solid #1976d2'
            }} />
            <Typography variant="body2">配送センター (DC)</Typography>
          </Box>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            ノード数: {visualizationData.nodes.length} | 
            エッジ数: {visualizationData.edges.length} | 
            中心座標: ({visualizationData.center_lat.toFixed(4)}, {visualizationData.center_lon.toFixed(4)})
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SNDMapVisualization;