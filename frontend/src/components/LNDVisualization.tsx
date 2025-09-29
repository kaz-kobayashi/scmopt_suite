import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Card, CardContent, Typography, Grid, Alert, Divider } from '@mui/material';

interface LNDVisualizationProps {
  title: string;
  data: any;
  layout?: any;
  config?: any;
}

const LNDVisualization: React.FC<LNDVisualizationProps> = ({
  title,
  data,
  layout = {},
  config = { responsive: true, displayModeBar: true },
}) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" component="h3" gutterBottom>
          {title}
        </Typography>
        <Box>
          <Plot
            data={data}
            layout={{
              autosize: true,
              ...layout,
            }}
            config={config}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export interface WeiszfeldLocationData {
  optimal_location: {
    latitude: number;
    longitude: number;
  };
  service_area_statistics: {
    facility_location: [number, number];
    total_customers: number;
    total_demand: number;
    average_distance: number;
    weighted_average_distance: number;
    max_distance: number;
    min_distance: number;
    median_distance: number;
    service_levels: {
      [key: string]: {
        customers_served: number;
        customers_percentage: number;
        demand_served: number;
        demand_percentage: number;
      };
    };
  };
  algorithm: string;
  parameters: {
    max_iterations: number;
    tolerance: number;
    weighted: boolean;
  };
}

export interface MultiFacilityWeiszfeldData {
  facility_locations: [number, number][];
  assignments: number[];
  total_cost: number;
  facility_stats: Array<{
    facility_index: number;
    location: [number, number];
    customers_assigned: number;
    total_demand_served: number;
    average_distance: number;
  }>;
  algorithm: string;
  iterations: number;
  converged: boolean;
  parameters: {
    num_facilities: number;
    max_iterations: number;
    tolerance: number;
    random_state: number;
  };
}

export const MultiFacilityWeiszfeldChart: React.FC<{ 
  data: MultiFacilityWeiszfeldData; 
  customerData: any[] 
}> = ({ data, customerData = [] }) => {
  const { facility_locations, assignments, facility_stats } = data;

  // Color palette for facilities
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];

  // Create scatter traces for each facility's assigned customers
  const customerTraces = facility_locations.map((_, facilityIndex) => {
    const assignedCustomers = customerData.filter((_, customerIndex) => 
      assignments[customerIndex] === facilityIndex
    );

    return {
      x: assignedCustomers.map(customer => customer.lon || customer.longitude),
      y: assignedCustomers.map(customer => customer.lat || customer.latitude),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: `Facility ${facilityIndex + 1} Customers`,
      marker: {
        color: colors[facilityIndex % colors.length],
        size: assignedCustomers.map(customer => Math.max((customer.demand || 1) / 10, 5)),
        sizemode: 'diameter' as const,
        opacity: 0.7,
        line: {
          color: '#2c3e50',
          width: 1
        }
      },
      hovertemplate: '<b>Customer (Facility %{fullData.name})</b><br>' +
                    'Lat: %{y:.4f}<br>' +
                    'Lon: %{x:.4f}<br>' +
                    'Demand: %{customdata}<br>' +
                    '<extra></extra>',
      customdata: assignedCustomers.map(customer => customer.demand || 1)
    };
  });

  // Create facility locations trace
  const facilityTrace = {
    x: facility_locations.map(location => location[1]), // longitude
    y: facility_locations.map(location => location[0]), // latitude
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Facilities',
    marker: {
      color: '#2c3e50',
      size: 25,
      symbol: 'star' as const,
      line: {
        color: 'white',
        width: 3
      }
    },
    hovertemplate: '<b>Facility %{pointIndex}</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Customers: %{customdata.customers}<br>' +
                  'Demand: %{customdata.demand:.1f}<br>' +
                  'Avg Distance: %{customdata.distance:.1f} km<br>' +
                  '<extra></extra>',
    customdata: facility_stats.map(stat => ({
      customers: stat.customers_assigned,
      demand: stat.total_demand_served,
      distance: stat.average_distance
    }))
  };

  const layout = {
    title: 'Multi-Facility Weiszfeld Algorithm Results',
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Multi-Facility Weiszfeld Results
          </Typography>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Facilities</Typography>
              <Typography variant="h6">{facility_locations.length}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Cost</Typography>
              <Typography variant="h6">{data.total_cost.toFixed(1)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Iterations</Typography>
              <Typography variant="h6">{data.iterations}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Converged</Typography>
              <Typography variant="h6">{data.converged ? 'Yes' : 'No'}</Typography>
            </Grid>
          </Grid>

          <Typography variant="subtitle2" gutterBottom>Facility Details:</Typography>
          <Grid container spacing={2}>
            {facility_stats.map((stat, index) => (
              <Grid item xs={12/Math.min(facility_stats.length, 3)} key={index}>
                <Box sx={{ p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Facility {index + 1}
                  </Typography>
                  <Typography variant="body2">
                    Location: ({stat.location[0].toFixed(4)}, {stat.location[1].toFixed(4)})
                  </Typography>
                  <Typography variant="body2">
                    Customers: {stat.customers_assigned}
                  </Typography>
                  <Typography variant="body2">
                    Demand: {stat.total_demand_served.toFixed(1)}
                  </Typography>
                  <Typography variant="body2">
                    Avg Distance: {stat.average_distance.toFixed(1)} km
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="Multi-Facility Locations" 
        data={[...customerTraces, facilityTrace]} 
        layout={layout} 
      />
    </Box>
  );
};

export const WeiszfeldLocationChart: React.FC<{ data: WeiszfeldLocationData; customerData: any[] }> = ({ 
  data, 
  customerData = [] 
}) => {
  const { optimal_location, service_area_statistics } = data;

  // Create scatter plot with customer locations and optimal facility
  const customerTrace = {
    x: customerData.map(customer => customer.lon || customer.longitude),
    y: customerData.map(customer => customer.lat || customer.latitude),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Customers',
    marker: {
      color: '#3498db',
      size: customerData.map(customer => Math.max((customer.demand || 1) / 10, 5)),
      sizemode: 'diameter' as const,
      line: {
        color: '#2980b9',
        width: 1
      }
    },
    hovertemplate: '<b>Customer</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Demand: %{customdata}<br>' +
                  '<extra></extra>',
    customdata: customerData.map(customer => customer.demand || 1)
  };

  const facilityTrace = {
    x: [optimal_location.longitude],
    y: [optimal_location.latitude],
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Optimal Facility',
    marker: {
      color: '#e74c3c',
      size: 20,
      symbol: 'star' as const,
      line: {
        color: '#c0392b',
        width: 2
      }
    },
    hovertemplate: '<b>Optimal Facility (Weiszfeld)</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  '<extra></extra>'
  };

  const layout = {
    title: 'Optimal Facility Location (Weiszfeld Algorithm)',
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Weiszfeld Algorithm Results
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Optimal Location</Typography>
              <Typography variant="h6">
                {optimal_location.latitude.toFixed(4)}, {optimal_location.longitude.toFixed(4)}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Customers</Typography>
              <Typography variant="h6">{service_area_statistics.total_customers}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Demand</Typography>
              <Typography variant="h6">{service_area_statistics.total_demand.toFixed(0)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Avg Distance</Typography>
              <Typography variant="h6">{service_area_statistics.average_distance.toFixed(1)} km</Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Service Level Analysis:</Typography>
            {Object.entries(service_area_statistics.service_levels).map(([threshold, stats]) => (
              <Typography key={threshold} variant="body2" sx={{ ml: 1 }}>
                {threshold}: {stats.customers_percentage.toFixed(1)}% customers, {stats.demand_percentage.toFixed(1)}% demand
              </Typography>
            ))}
          </Box>
        </CardContent>
      </Card>
      
      <LNDVisualization title="Optimal Facility Location" data={[customerTrace, facilityTrace]} layout={layout} />
    </Box>
  );
};

export interface CustomerClusteringData {
  clustered_data: Array<{
    [key: string]: any;
    cluster: number;
    lat: number;
    lon: number;
    demand?: number;
  }>;
  cluster_statistics: Array<{
    cluster_id: number;
    num_customers: number;
    total_demand: number;
    center_lat: number;
    center_lon: number;
    avg_distance_to_center: number;
  }>;
  algorithm: string;
  parameters: any;
}

export const CustomerClusteringChart: React.FC<{ data: CustomerClusteringData }> = ({ data }) => {
  const { clustered_data, cluster_statistics } = data;

  // Color palette for clusters
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];

  // Create scatter traces for each cluster
  const clusterTraces = cluster_statistics.map(cluster => {
    const clusterCustomers = clustered_data.filter(customer => customer.cluster === cluster.cluster_id);
    
    return {
      x: clusterCustomers.map(customer => customer.lon),
      y: clusterCustomers.map(customer => customer.lat),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: `Cluster ${cluster.cluster_id}`,
      marker: {
        color: colors[cluster.cluster_id % colors.length],
        size: clusterCustomers.map(customer => Math.max((customer.demand || 1) / 10, 5)),
        sizemode: 'diameter' as const,
        opacity: 0.7
      },
      hovertemplate: '<b>Cluster %{fullData.name}</b><br>' +
                    'Lat: %{y:.4f}<br>' +
                    'Lon: %{x:.4f}<br>' +
                    'Demand: %{customdata}<br>' +
                    '<extra></extra>',
      customdata: clusterCustomers.map(customer => customer.demand || 1)
    };
  });

  // Add cluster centers
  const centerTrace = {
    x: cluster_statistics.map(cluster => cluster.center_lon),
    y: cluster_statistics.map(cluster => cluster.center_lat),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Cluster Centers',
    marker: {
      color: 'black',
      size: 15,
      symbol: 'x' as const,
      line: {
        color: 'white',
        width: 2
      }
    },
    hovertemplate: '<b>Cluster Center</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Customers: %{customdata.customers}<br>' +
                  'Total Demand: %{customdata.demand}<br>' +
                  '<extra></extra>',
    customdata: cluster_statistics.map(cluster => ({
      customers: cluster.num_customers,
      demand: cluster.total_demand
    }))
  };

  const layout = {
    title: `Customer Clustering (${data.algorithm.toUpperCase()})`,
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Customer Clustering Results ({data.algorithm.toUpperCase()})
          </Typography>
          <Grid container spacing={2}>
            {cluster_statistics.map(cluster => (
              <Grid item xs={12/Math.min(cluster_statistics.length, 4)} key={cluster.cluster_id}>
                <Typography variant="body2" color="text.secondary">Cluster {cluster.cluster_id}</Typography>
                <Typography variant="body1">{cluster.num_customers} customers</Typography>
                <Typography variant="body2">Demand: {cluster.total_demand.toFixed(0)}</Typography>
                <Typography variant="body2">Avg Dist: {cluster.avg_distance_to_center.toFixed(1)} km</Typography>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="Customer Clustering Visualization" 
        data={[...clusterTraces, centerTrace]} 
        layout={layout} 
      />
    </Box>
  );
};

export interface KMedianSolutionData {
  selected_facilities: number[];
  assignments?: number[];
  total_cost: number;
  facility_locations?: [number, number][];
  facility_stats?: Array<{
    facility_index: number;
    location: [number, number];
    customers_assigned: number;
    total_demand_served: number;
    average_distance: number;
  }>;
  candidate_facilities: [number, number][];
  algorithm: string;
  iterations: number;
  parameters: any;
  error?: string;
}

export const KMedianSolutionChart: React.FC<{ 
  data: KMedianSolutionData; 
  customerData: any[] 
}> = ({ data, customerData = [] }) => {
  const { facility_locations, facility_stats, candidate_facilities, assignments } = data;

  // Handle case where data might be incomplete (e.g., placeholder/error responses)
  if (!assignments || !facility_locations || !customerData.length) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="h6" color="text.secondary">
          {data.error || 'No data available for visualization'}
        </Typography>
        {data.error && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Please ensure the algorithm implementation is complete.
          </Typography>
        )}
      </Box>
    );
  }

  // Create customer traces colored by assignment
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
  
  const customerTrace = {
    x: customerData.map(customer => customer.lon || customer.longitude),
    y: customerData.map(customer => customer.lat || customer.latitude),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Customers',
    marker: {
      color: assignments.map(assignment => colors[assignment % colors.length]),
      size: customerData.map(customer => Math.max((customer.demand || 1) / 10, 5)),
      sizemode: 'diameter' as const,
      line: {
        color: 'rgba(0,0,0,0.3)',
        width: 1
      }
    },
    hovertemplate: '<b>Customer</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Demand: %{customdata.demand}<br>' +
                  'Assigned to Facility: %{customdata.assignment}<br>' +
                  '<extra></extra>',
    customdata: customerData.map((customer, i) => ({
      demand: customer.demand || 1,
      assignment: assignments[i]
    }))
  };

  // Selected facilities
  const selectedFacilitiesTrace = {
    x: facility_locations.map(loc => loc[1]),
    y: facility_locations.map(loc => loc[0]),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Selected Facilities',
    marker: {
      color: '#e74c3c',
      size: 20,
      symbol: 'square' as const,
      line: {
        color: '#c0392b',
        width: 2
      }
    },
    hovertemplate: '<b>Selected Facility</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Customers Served: %{customdata.customers}<br>' +
                  'Total Demand: %{customdata.demand}<br>' +
                  '<extra></extra>',
    customdata: facility_stats?.map(stat => ({
      customers: stat.customers_assigned,
      demand: stat.total_demand_served
    })) || []
  };

  // Candidate facilities (not selected)
  const candidateTrace = {
    x: candidate_facilities.map(loc => loc[1]),
    y: candidate_facilities.map(loc => loc[0]),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Candidate Facilities',
    marker: {
      color: '#bdc3c7',
      size: 8,
      symbol: 'circle' as const,
      opacity: 0.5
    },
    hovertemplate: '<b>Candidate Facility</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  '<extra></extra>'
  };

  const layout = {
    title: 'K-Median Facility Location Solution',
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            K-Median Solution Results
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Cost</Typography>
              <Typography variant="h6">{data.total_cost.toFixed(2)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Facilities Selected</Typography>
              <Typography variant="h6">{facility_locations.length}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Customers Served</Typography>
              <Typography variant="h6">{customerData.length}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Iterations</Typography>
              <Typography variant="h6">{data.iterations}</Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Facility Details:</Typography>
            {facility_stats?.map((stat, index) => (
              <Typography key={index} variant="body2" sx={{ ml: 1 }}>
                Facility {index + 1}: {stat.customers_assigned} customers, {stat.total_demand_served.toFixed(0)} demand, {stat.average_distance.toFixed(1)} km avg distance
              </Typography>
            ))}
          </Box>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="K-Median Facility Location Solution" 
        data={[candidateTrace, customerTrace, selectedFacilitiesTrace]} 
        layout={layout} 
      />
    </Box>
  );
};

export interface ServiceAreaData {
  service_area_analysis: {
    facility_location: [number, number];
    total_customers: number;
    total_demand: number;
    average_distance: number;
    weighted_average_distance: number;
    max_distance: number;
    min_distance: number;
    median_distance: number;
    service_levels: {
      [key: string]: {
        customers_served: number;
        customers_percentage: number;
        demand_served: number;
        demand_percentage: number;
      };
    };
  };
  facility_location: {
    latitude: number;
    longitude: number;
  };
}

export const ServiceAreaChart: React.FC<{ 
  data: ServiceAreaData; 
  customerData: any[] 
}> = ({ data, customerData = [] }) => {
  const { service_area_analysis, facility_location } = data;

  // Create distance-based color mapping for customers
  const getDistanceColor = (distance: number): string => {
    if (distance <= 10) return '#2ecc71';      // Green - very close
    if (distance <= 25) return '#f39c12';     // Orange - close
    if (distance <= 50) return '#e67e22';     // Dark orange - medium
    if (distance <= 100) return '#e74c3c';    // Red - far
    return '#8e44ad';                          // Purple - very far
  };

  // Get service area classification based on distance
  const getServiceAreaClassification = (distance: number): string => {
    if (distance <= 10) return 'Service Area - Very Close';
    if (distance <= 25) return 'Service Area - Close';
    if (distance <= 50) return 'Service Area - Medium Distance';
    if (distance <= 100) return 'Service Area - Far';
    return 'Service Area - Very Far';
  };

  // Calculate distances and assign colors
  const enhancedCustomerData = customerData.map(customer => {
    const distance = Math.sqrt(
      Math.pow((customer.lat - facility_location.latitude) * 111, 2) + 
      Math.pow((customer.lon - facility_location.longitude) * 111 * Math.cos(customer.lat * Math.PI / 180), 2)
    );
    return {
      ...customer,
      distance: distance,
      color: getDistanceColor(distance),
      serviceAreaClassification: getServiceAreaClassification(distance)
    };
  });

  const customerTrace = {
    x: enhancedCustomerData.map(customer => customer.lon),
    y: enhancedCustomerData.map(customer => customer.lat),
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Customers',
    marker: {
      color: enhancedCustomerData.map(customer => customer.color),
      size: enhancedCustomerData.map(customer => Math.max((customer.demand || 1) / 10, 5)),
      sizemode: 'diameter' as const,
      line: {
        color: 'rgba(0,0,0,0.3)',
        width: 1
      }
    },
    hovertemplate: '<b>%{customdata.classification}</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  'Demand: %{customdata.demand}<br>' +
                  'Distance: %{customdata.distance:.1f} km<br>' +
                  '<extra></extra>',
    customdata: enhancedCustomerData.map(customer => ({
      demand: customer.demand || 1,
      distance: customer.distance,
      classification: customer.serviceAreaClassification
    }))
  };

  const facilityTrace = {
    x: [facility_location.longitude],
    y: [facility_location.latitude],
    mode: 'markers' as const,
    type: 'scatter' as const,
    name: 'Facility',
    marker: {
      color: '#2c3e50',
      size: 25,
      symbol: 'diamond' as const,
      line: {
        color: 'white',
        width: 3
      }
    },
    hovertemplate: '<b>Facility</b><br>' +
                  'Lat: %{y:.4f}<br>' +
                  'Lon: %{x:.4f}<br>' +
                  '<extra></extra>'
  };

  const layout = {
    title: 'Facility Service Area Analysis',
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Service Area Analysis
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Customers</Typography>
              <Typography variant="h6">{service_area_analysis.total_customers}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Demand</Typography>
              <Typography variant="h6">{service_area_analysis.total_demand.toFixed(0)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Avg Distance</Typography>
              <Typography variant="h6">{service_area_analysis.average_distance.toFixed(1)} km</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Max Distance</Typography>
              <Typography variant="h6">{service_area_analysis.max_distance.toFixed(1)} km</Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>Service Level Coverage:</Typography>
            <Grid container spacing={1}>
              {Object.entries(service_area_analysis.service_levels).map(([threshold, stats]) => (
                <Grid item xs={6} key={threshold}>
                  <Typography variant="body2">
                    {threshold}: {stats.customers_percentage.toFixed(1)}% customers, {stats.demand_percentage.toFixed(1)}% demand
                  </Typography>
                </Grid>
              ))}
            </Grid>
          </Box>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="Service Area Analysis" 
        data={[customerTrace, facilityTrace]} 
        layout={layout} 
      />
    </Box>
  );
};

export interface SingleSourceLNDData {
  status: string;
  runtime: number;
  total_cost: number;
  selected_warehouses: Array<{
    warehouse_index: string;
    warehouse_name: string;
    location: [number, number];
    customers_assigned: number;
    total_demand_served: number;
  }>;
  customer_assignments: Array<{
    customer_id: string;
    customer_name: string;
    customer_location: [number, number];
    warehouse_index: string;
    warehouse_name: string;
    warehouse_location: [number, number];
    product_id: string;
    product_name: string;
    demand: number;
    transportation_cost: number;
    delivery_cost: number;
  }>;
  cost_breakdown: {
    fixed_cost: number;
    transportation_cost: number;
    delivery_cost: number;
    variable_cost: number;
  };
  num_warehouses_opened: number;
  num_customer_assignments: number;
  single_sourcing: boolean;
  message: string;
}

export const SingleSourceLNDChart: React.FC<{ 
  data: SingleSourceLNDData;
}> = ({ data }) => {
  if (!data || !data.selected_warehouses || data.selected_warehouses.length === 0) {
    return (
      <Alert severity="warning">
        No optimal solution found. Please check your data and parameters.
      </Alert>
    );
  }

  // Color palette for warehouses
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];

  // Group customer assignments by warehouse
  const warehouseToCustomers = data.customer_assignments.reduce((acc, assignment) => {
    if (!acc[assignment.warehouse_index]) {
      acc[assignment.warehouse_index] = [];
    }
    acc[assignment.warehouse_index].push(assignment);
    return acc;
  }, {} as Record<string, typeof data.customer_assignments>);

  // Create customer traces for each warehouse
  const customerTraces = data.selected_warehouses.map((warehouse, index) => {
    const warehouseCustomers = warehouseToCustomers[warehouse.warehouse_index] || [];
    
    return {
      x: warehouseCustomers.map(customer => customer.customer_location[1]),
      y: warehouseCustomers.map(customer => customer.customer_location[0]),
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: `${warehouse.warehouse_name} Customers`,
      marker: {
        color: colors[index % colors.length],
        size: warehouseCustomers.map(customer => Math.max(customer.demand / 100, 5)),
        sizemode: 'diameter' as const,
        opacity: 0.7,
        line: {
          color: '#2c3e50',
          width: 1
        }
      },
      hovertemplate: '<b>Customer: %{customdata.customerName}</b><br>' +
                    'Location: (%{y:.4f}, %{x:.4f})<br>' +
                    'Product: %{customdata.productName}<br>' +
                    'Demand: %{customdata.demand}<br>' +
                    'Assigned to: %{customdata.warehouseName}<br>' +
                    'Transportation Cost: ¥%{customdata.transportCost:.2f}<br>' +
                    'Delivery Cost: ¥%{customdata.deliveryCost:.2f}<br>' +
                    '<extra></extra>',
      customdata: warehouseCustomers.map(customer => ({
        customerName: customer.customer_name,
        productName: customer.product_name,
        demand: customer.demand,
        warehouseName: customer.warehouse_name,
        transportCost: customer.transportation_cost,
        deliveryCost: customer.delivery_cost
      }))
    };
  });

  // Create warehouse trace
  const warehouseTrace = {
    x: data.selected_warehouses.map(warehouse => warehouse.location[1]),
    y: data.selected_warehouses.map(warehouse => warehouse.location[0]),
    mode: 'markers+text' as const,
    type: 'scatter' as const,
    name: 'Selected Warehouses',
    text: data.selected_warehouses.map(warehouse => warehouse.warehouse_name),
    textposition: 'top center' as const,
    marker: {
      color: data.selected_warehouses.map((_, index) => colors[index % colors.length]),
      size: 25,
      symbol: 'square' as const,
      line: {
        color: 'white',
        width: 3
      }
    },
    hovertemplate: '<b>Warehouse: %{customdata.name}</b><br>' +
                  'Location: (%{y:.4f}, %{x:.4f})<br>' +
                  'Customers Assigned: %{customdata.customers}<br>' +
                  'Total Demand Served: %{customdata.demand}<br>' +
                  '<extra></extra>',
    customdata: data.selected_warehouses.map(warehouse => ({
      name: warehouse.warehouse_name,
      customers: warehouse.customers_assigned,
      demand: warehouse.total_demand_served
    }))
  };

  // Create connection lines between warehouses and customers
  const connectionLines: any[] = [];
  data.selected_warehouses.forEach((warehouse, warehouseIndex) => {
    const warehouseCustomers = warehouseToCustomers[warehouse.warehouse_index] || [];
    warehouseCustomers.forEach(customer => {
      connectionLines.push({
        x: [warehouse.location[1], customer.customer_location[1]],
        y: [warehouse.location[0], customer.customer_location[0]],
        mode: 'lines' as const,
        type: 'scatter' as const,
        line: {
          color: colors[warehouseIndex % colors.length],
          width: 0.5
        },
        showlegend: false,
        hoverinfo: 'skip' as const
      });
    });
  });

  const layout = {
    title: 'Single Source Logistics Network Design Results',
    xaxis: { 
      title: 'Longitude',
      scaleanchor: 'y',
      scaleratio: 1
    },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 600,
    margin: { l: 50, r: 50, t: 80, b: 50 }
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Single Source LND Optimization Results
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Cost</Typography>
              <Typography variant="h6">¥{(data.total_cost || 0).toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Warehouses Opened</Typography>
              <Typography variant="h6">{data.num_warehouses_opened || 0}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Runtime</Typography>
              <Typography variant="h6">{(data.runtime || 0).toFixed(2)}s</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Status</Typography>
              <Typography variant="h6" color={data.status === 'Optimal' ? 'success.main' : 'warning.main'}>
                {data.status}
              </Typography>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom>Cost Breakdown:</Typography>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Fixed Costs</Typography>
              <Typography variant="body1">¥{(data.cost_breakdown?.fixed_cost || 0).toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Transportation Costs</Typography>
              <Typography variant="body1">¥{(data.cost_breakdown?.transportation_cost || 0).toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Delivery Costs</Typography>
              <Typography variant="body1">¥{(data.cost_breakdown?.delivery_cost || 0).toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Variable Costs</Typography>
              <Typography variant="body1">¥{(data.cost_breakdown?.variable_cost || 0).toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom>Selected Warehouses:</Typography>
          <Grid container spacing={2}>
            {data.selected_warehouses.map((warehouse, index) => (
              <Grid item xs={12/Math.min(data.selected_warehouses.length, 3)} key={warehouse.warehouse_index}>
                <Box sx={{ 
                  p: 1.5, 
                  border: '2px solid', 
                  borderColor: colors[index % colors.length],
                  borderRadius: 1,
                  backgroundColor: `${colors[index % colors.length]}10`
                }}>
                  <Typography variant="body2" fontWeight="bold" color={colors[index % colors.length]}>
                    {warehouse.warehouse_name}
                  </Typography>
                  <Typography variant="body2">
                    Location: ({warehouse.location?.[0]?.toFixed(4) || 'N/A'}, {warehouse.location?.[1]?.toFixed(4) || 'N/A'})
                  </Typography>
                  <Typography variant="body2">
                    Customers: {warehouse.customers_assigned || 0}
                  </Typography>
                  <Typography variant="body2">
                    Total Demand: {warehouse.total_demand_served?.toFixed(0) || 'N/A'}
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="Single Source Network Configuration" 
        data={[...connectionLines, ...customerTraces, warehouseTrace]} 
        layout={layout} 
      />
    </Box>
  );
};

export interface ElbowMethodData {
  analysis_results: Array<{
    num_facilities: number;
    total_cost: number;
    average_distance: number;
    wcss: number;
    facility_locations: [number, number][];
    assignments: number[];
  }>;
  costs: number[];
  cost_changes: number[];
  improvements: number[];
  optimal_num_facilities: number;
  algorithm: string;
  parameters: {
    min_facilities: number;
    max_facilities: number;
    max_iterations: number;
    tolerance: number;
    random_state: number;
  };
  num_customers: number;
  total_demand: number;
}

export const ElbowMethodChart: React.FC<{ data: ElbowMethodData }> = ({ data }) => {
  if (!data || !data.analysis_results) {
    return (
      <Alert severity="warning">
        No analysis results available.
      </Alert>
    );
  }

  // Prepare data for cost curve chart
  const costTrace = {
    x: data.analysis_results.map(r => r.num_facilities),
    y: data.costs,
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: 'Total Cost',
    line: {
      color: '#3498db',
      width: 3
    },
    marker: {
      size: 8,
      color: '#3498db'
    }
  };

  // Mark the optimal point
  const optimalIndex = data.analysis_results.findIndex(r => r.num_facilities === data.optimal_num_facilities);
  const optimalTrace = {
    x: [data.optimal_num_facilities],
    y: [data.costs[optimalIndex]],
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Optimal Number',
    marker: {
      size: 15,
      color: '#e74c3c',
      symbol: 'star' as const,
      line: {
        color: '#c0392b',
        width: 2
      }
    }
  };

  const costLayout = {
    title: 'Elbow Method Analysis - Cost vs Number of Facilities',
    xaxis: { 
      title: 'Number of Facilities',
      dtick: 1
    },
    yaxis: { title: 'Total Cost' },
    showlegend: true,
    hovermode: 'closest' as const,
    annotations: [{
      x: data.optimal_num_facilities,
      y: data.costs[optimalIndex],
      xref: 'x',
      yref: 'y',
      text: `Optimal: ${data.optimal_num_facilities} facilities`,
      showarrow: true,
      arrowhead: 7,
      ax: 0,
      ay: -40
    }]
  };

  // Prepare data for improvement chart
  const improvementTrace = {
    x: data.analysis_results.slice(1).map(r => r.num_facilities),
    y: data.improvements,
    type: 'bar' as const,
    name: 'Cost Improvement %',
    marker: {
      color: data.improvements.map((_, i) => 
        i + 2 === data.optimal_num_facilities ? '#e74c3c' : '#2ecc71'
      )
    }
  };

  const improvementLayout = {
    title: 'Cost Improvement Percentage',
    xaxis: { 
      title: 'Number of Facilities',
      dtick: 1
    },
    yaxis: { 
      title: 'Improvement %',
      tickformat: '.1f'
    },
    showlegend: false,
    hovermode: 'closest' as const
  };

  // Prepare data for WCSS chart (Within-Cluster Sum of Squares)
  const wcssTrace = {
    x: data.analysis_results.map(r => r.num_facilities),
    y: data.analysis_results.map(r => r.wcss),
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: 'WCSS',
    line: {
      color: '#9b59b6',
      width: 3
    },
    marker: {
      size: 8,
      color: '#9b59b6'
    }
  };

  const wcssLayout = {
    title: 'Within-Cluster Sum of Squares',
    xaxis: { 
      title: 'Number of Facilities',
      dtick: 1
    },
    yaxis: { title: 'WCSS' },
    showlegend: false,
    hovermode: 'closest' as const
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Elbow Method Analysis Results
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Optimal Facilities</Typography>
              <Typography variant="h4" color="primary">{data.optimal_num_facilities}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Algorithm Used</Typography>
              <Typography variant="h6">{data.algorithm.toUpperCase()}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Customers</Typography>
              <Typography variant="h6">{data.num_customers}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Demand</Typography>
              <Typography variant="h6">{data.total_demand.toFixed(0)}</Typography>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom>Analysis Summary:</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">
                Facility Range Tested: {data.parameters.min_facilities} - {data.parameters.max_facilities}
              </Typography>
              <Typography variant="body2">
                Algorithm: {data.algorithm} (max iterations: {data.parameters.max_iterations})
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">
                Cost at Optimal: ¥{data.costs[optimalIndex]?.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}
              </Typography>
              <Typography variant="body2">
                Average Distance at Optimal: {data.analysis_results[optimalIndex]?.average_distance.toFixed(2)} km
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <LNDVisualization 
            title="Cost vs Number of Facilities" 
            data={[costTrace, optimalTrace]} 
            layout={costLayout} 
          />
        </Grid>
        <Grid item xs={6}>
          <LNDVisualization 
            title="Cost Improvement by Additional Facility" 
            data={[improvementTrace]} 
            layout={improvementLayout} 
          />
        </Grid>
        <Grid item xs={6}>
          <LNDVisualization 
            title="Within-Cluster Sum of Squares" 
            data={[wcssTrace]} 
            layout={wcssLayout} 
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export interface MultipleSourceLNDData {
  status: string;
  runtime: number;
  total_cost: number;
  selected_warehouses: Array<{
    warehouse_id: string;
    location: {
      lat: number;
      lon: number;
    };
  }>;
  cost_breakdown: {
    fixed_cost: number;
    transportation_cost: number;
    delivery_cost: number;
    variable_cost: number;
  };
  num_warehouses_opened: number;
  message: string;
}

export const MultipleSourceLNDChart: React.FC<{ 
  data: MultipleSourceLNDData;
  customerData?: any[];
  warehouseData?: any[];
  factoryData?: any[];
}> = ({ data, customerData = [], warehouseData = [], factoryData = [] }) => {
  if (!data || data.status !== 'Optimal') {
    return (
      <Alert severity="error">
        {data?.message || 'Optimization failed or no data available'}
      </Alert>
    );
  }

  const { selected_warehouses, cost_breakdown, total_cost } = data;

  // Color palette
  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

  // Create customer trace
  const customerTrace = customerData.length > 0 ? {
    x: customerData.map(c => typeof c.lon === 'string' ? parseFloat(c.lon) : c.lon),
    y: customerData.map(c => typeof c.lat === 'string' ? parseFloat(c.lat) : c.lat),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Customers',
    marker: {
      size: 8,
      color: '#34495e',
      symbol: 'circle' as const,
      line: {
        color: '#2c3e50',
        width: 1
      }
    },
    text: customerData.map(c => `Customer: ${c.customer_id || c.name || 'Unknown'}`),
    hovertemplate: '%{text}<br>Location: (%{x:.4f}, %{y:.4f})<extra></extra>'
  } : null;

  // Create factory trace
  const factoryTrace = factoryData.length > 0 ? {
    x: factoryData.map(f => typeof f.lon === 'string' ? parseFloat(f.lon) : f.lon),
    y: factoryData.map(f => typeof f.lat === 'string' ? parseFloat(f.lat) : f.lat),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Factories',
    marker: {
      size: 12,
      color: '#8e44ad',
      symbol: 'square' as const,
      line: {
        color: '#6c3483',
        width: 2
      }
    },
    text: factoryData.map(f => `Factory: ${f.factory_id || f.name || 'Unknown'}`),
    hovertemplate: '%{text}<br>Location: (%{x:.4f}, %{y:.4f})<extra></extra>'
  } : null;

  // Create warehouse traces (available vs selected)
  const availableWarehouses = warehouseData.filter(w => 
    !selected_warehouses.some(sw => sw.warehouse_id === (w.warehouse_id || w.name))
  );

  const availableWarehouseTrace = availableWarehouses.length > 0 ? {
    x: availableWarehouses.map(w => typeof w.lon === 'string' ? parseFloat(w.lon) : w.lon),
    y: availableWarehouses.map(w => typeof w.lat === 'string' ? parseFloat(w.lat) : w.lat),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Available Warehouses',
    marker: {
      size: 10,
      color: '#bdc3c7',
      symbol: 'diamond' as const,
      line: {
        color: '#95a5a6',
        width: 1
      }
    },
    text: availableWarehouses.map(w => `Warehouse: ${w.warehouse_id || w.name || 'Unknown'}<br>Status: Available`),
    hovertemplate: '%{text}<br>Location: (%{x:.4f}, %{y:.4f})<extra></extra>'
  } : null;

  // Create selected warehouse trace
  const selectedWarehouseTrace = {
    x: selected_warehouses.map(w => w.location.lon),
    y: selected_warehouses.map(w => w.location.lat),
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: 'Selected Warehouses',
    marker: {
      size: 15,
      color: '#e74c3c',
      symbol: 'diamond' as const,
      line: {
        color: '#c0392b',
        width: 2
      }
    },
    text: selected_warehouses.map(w => `Warehouse: ${w.warehouse_id}<br>Status: SELECTED`),
    hovertemplate: '%{text}<br>Location: (%{x:.4f}, %{y:.4f})<extra></extra>'
  };

  const traces = [customerTrace, factoryTrace, availableWarehouseTrace, selectedWarehouseTrace].filter(Boolean);

  const layout = {
    title: 'Multiple Source LND - Network Configuration',
    xaxis: { title: 'Longitude' },
    yaxis: { title: 'Latitude' },
    showlegend: true,
    hovermode: 'closest' as const,
    height: 500
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Multiple Source LND Optimization Results
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Total Cost</Typography>
              <Typography variant="h5" color="primary">¥{total_cost.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Warehouses Opened</Typography>
              <Typography variant="h5">{data.num_warehouses_opened}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Runtime</Typography>
              <Typography variant="h6">{data.runtime.toFixed(2)}s</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">Status</Typography>
              <Typography variant="h6" color="success.main">{data.status}</Typography>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom>Cost Breakdown:</Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2">Fixed Cost:</Typography>
              <Typography variant="body1" fontWeight="bold">
                ¥{cost_breakdown.fixed_cost.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2">Transportation Cost:</Typography>
              <Typography variant="body1" fontWeight="bold">
                ¥{cost_breakdown.transportation_cost.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2">Delivery Cost:</Typography>
              <Typography variant="body1" fontWeight="bold">
                ¥{cost_breakdown.delivery_cost.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}
              </Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2">Variable Cost:</Typography>
              <Typography variant="body1" fontWeight="bold">
                ¥{cost_breakdown.variable_cost.toLocaleString('ja-JP', { minimumFractionDigits: 2 })}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Selected Warehouses
          </Typography>
          <Grid container spacing={2}>
            {selected_warehouses.map((warehouse, index) => (
              <Grid item xs={12/Math.min(selected_warehouses.length, 4)} key={warehouse.warehouse_id}>
                <Box sx={{ 
                  p: 1.5, 
                  border: '2px solid', 
                  borderColor: colors[index % colors.length],
                  borderRadius: 1,
                  backgroundColor: `${colors[index % colors.length]}10`
                }}>
                  <Typography variant="body2" fontWeight="bold" color={colors[index % colors.length]}>
                    {warehouse.warehouse_id}
                  </Typography>
                  <Typography variant="body2">
                    Location: ({warehouse.location.lat?.toFixed(4) || 'N/A'}, {warehouse.location.lon?.toFixed(4) || 'N/A'})
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
      
      <LNDVisualization 
        title="Multiple Source Network Configuration" 
        data={traces} 
        layout={layout} 
      />
    </Box>
  );
};

export default LNDVisualization;