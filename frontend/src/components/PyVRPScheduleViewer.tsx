import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  LocalShipping as TruckIcon,
  LocationOn as LocationIcon,
  Schedule as TimeIcon,
  Store as DepotIcon,
} from '@mui/icons-material';

interface ScheduleEvent {
  time: number;
  type: 'departure' | 'travel' | 'arrival' | 'service' | 'return';
  location: string;
  duration: number;
  details: string;
  clientIndex?: number;
}

interface RouteSchedule {
  routeIndex: number;
  vehicleType: number;
  startTime: number;
  endTime: number;
  events: ScheduleEvent[];
  totalDistance: number;
  totalDuration: number;
  clientsServed: number[];
}

interface PyVRPScheduleViewerProps {
  solution: any;
  clients: any[];
  depots: any[];
  vehicleTypes: any[];
  getClientName?: (index: number) => string;
}

const PyVRPScheduleViewer: React.FC<PyVRPScheduleViewerProps> = ({
  solution,
  clients,
  depots,
  vehicleTypes,
  getClientName
}) => {
  // Helper function to get client name if available
  const defaultGetClientName = (index: number): string => {
    return `Client ${index}`;  // Default client naming
  };
  
  const clientNameFunction = getClientName || defaultGetClientName;
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

  // Calculate haversine distance between two points
  const calculateDistance = (loc1: any, loc2: any): number => {
    // Check if coordinates are already scaled or in lat/lon format
    const lon1 = (loc1.x || 0) > 1000 ? (loc1.x || 0) / 10000 : (loc1.x || 0);
    const lat1 = (loc1.y || 0) > 1000 ? (loc1.y || 0) / 10000 : (loc1.y || 0);
    const lon2 = (loc2.x || 0) > 1000 ? (loc2.x || 0) / 10000 : (loc2.x || 0);
    const lat2 = (loc2.y || 0) > 1000 ? (loc2.y || 0) / 10000 : (loc2.y || 0);
    
    const toRadians = (degrees: number) => degrees * (Math.PI / 180);
    
    const dlon = toRadians(lon2 - lon1);
    const dlat = toRadians(lat2 - lat1);
    const a = Math.sin(dlat/2) * Math.sin(dlat/2) + 
              Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) * 
              Math.sin(dlon/2) * Math.sin(dlon/2);
    const c = 2 * Math.asin(Math.sqrt(a));
    const r = 6371000; // Earth's radius in meters
    
    return r * c;
  };

  // Generate detailed schedule for each route
  const routeSchedules: RouteSchedule[] = useMemo(() => {
    if (!solution?.routes || !clients || !depots) return [];

    // Debug: Log the actual solution data
    console.log('PyVRP Solution:', solution);
    console.log('Routes data:', solution.routes);
    
    return solution.routes.map((route: any, routeIndex: number) => {
      console.log(`Processing Route ${routeIndex}:`, route);
      console.log(`Route clients:`, route.clients);
      console.log(`Route has arrival_times:`, route.arrival_times);
      console.log(`Route has departure_times:`, route.departure_times);
      
      const events: ScheduleEvent[] = [];
      const vehicleType = vehicleTypes[route.vehicle_type || 0];
      // Use backend start_time if available and valid, otherwise use vehicle type time window or default
      const backendStartTime = route.start_time;
      // If backend start_time is 0, treat it as invalid and use vehicle type time window
      const startTime = (backendStartTime !== undefined && backendStartTime !== null && backendStartTime > 0) ? 
        backendStartTime : 
        (vehicleType?.tw_early || 480); // Default 8:00 AM
      let currentTime = startTime;
      
      console.log(`Route ${routeIndex} start time calculation:`, {
        backendStartTime,
        vehicleTypeTwEarly: vehicleType?.tw_early,
        finalStartTime: startTime
      });
      
      // Check if we have pre-calculated timing data from PyVRP backend
      const useBackendTiming = route.arrival_times && route.departure_times && 
                               route.arrival_times.length === route.clients?.length &&
                               route.departure_times.length === route.clients?.length;
      
      // Starting depot
      const startDepot = depots[route.start_depot || 0];
      let currentLocation = startDepot;
      
      // Departure from depot
      events.push({
        time: currentTime,
        type: 'departure',
        location: `Depot ${route.start_depot || 0}`,
        duration: 0,
        details: 'Departure from depot'
      });

      let totalDistance = 0;

      // Process each client visit
      if (route.clients && Array.isArray(route.clients)) {
        if (useBackendTiming) {
          console.log('Using backend timing data');
          // Use pre-calculated timing from PyVRP backend
          route.clients.forEach((clientIndex: number, visitIndex: number) => {
            const client = clients[clientIndex];
            if (!client) {
              console.warn(`Client ${clientIndex} not found in clients array`);
              return;
            }

            const rawArrivalTime = route.arrival_times[visitIndex];
            const rawDepartureTime = route.departure_times[visitIndex];
            
            // If backend times start from 0, add the start time offset
            // Otherwise use backend times directly as they are already absolute times
            const isRelativeTime = route.start_time === 0 || route.start_time === undefined;
            
            const arrivalTime = (rawArrivalTime !== undefined && rawArrivalTime !== null) ? 
              (isRelativeTime ? startTime + rawArrivalTime : rawArrivalTime) : 
              startTime + visitIndex * 30;
            const departureTime = (rawDepartureTime !== undefined && rawDepartureTime !== null) ? 
              (isRelativeTime ? startTime + rawDepartureTime : rawDepartureTime) : 
              arrivalTime + (client.service_duration || 10);
            
            console.log(`Visit ${visitIndex}: Raw arrival=${rawArrivalTime}, Raw departure=${rawDepartureTime}`);
            console.log(`Visit ${visitIndex}: Fixed arrival=${arrivalTime}, Fixed departure=${departureTime}`);
            
            // Travel to client (from previous location)
            const travelStartTime = visitIndex === 0 ? 
              startTime : 
              (route.departure_times[visitIndex - 1] || startTime);
            const travelTime = Math.max(0, arrivalTime - travelStartTime);
            
            if (travelTime > 0) {
              events.push({
                time: travelStartTime,
                type: 'travel',
                location: visitIndex === 0 
                  ? `Depot ${route.start_depot || 0} → Client ${clientIndex}`
                  : `Client ${route.clients[visitIndex - 1]} → Client ${clientIndex}`,
                duration: travelTime,
                details: `Travel time`
              });
            }

            // Arrival and Service events
            events.push({
              time: arrivalTime,
              type: 'arrival',
              location: clientNameFunction(clientIndex),
              duration: 0,
              details: `Arrival at client`,
              clientIndex: clientIndex
            });

            const serviceTime = departureTime - arrivalTime;
            events.push({
              time: arrivalTime,
              type: 'service',
              location: clientNameFunction(clientIndex),
              duration: serviceTime,
              details: `Service - Delivery: ${client.delivery}${client.pickup ? `, Pickup: ${client.pickup}` : ''}${client.prize ? `, Prize: ${client.prize}` : ''}`,
              clientIndex: clientIndex
            });
            
            currentTime = departureTime;
          });
        } else {
          console.log('Calculating timing manually');
          // Calculate timing manually
          route.clients.forEach((clientIndex: number, visitIndex: number) => {
            const client = clients[clientIndex];
            if (!client) {
              console.warn(`Client ${clientIndex} not found in clients array`);
              return;
            }

            // Calculate travel time to client
            const travelDistance = calculateDistance(currentLocation, client);
            const travelTime = Math.max(1, Math.round(travelDistance / 1000 / 20 * 60)); // 20km/h, minimum 1 minute
            
            console.log(`Route ${routeIndex}, Visit ${visitIndex}: Client ${clientIndex}`);
            console.log(`  From:`, currentLocation);
            console.log(`  To:`, client);
            console.log(`  Distance: ${(travelDistance/1000).toFixed(2)}km, Travel time: ${travelTime}min`);
            console.log(`  Current time before travel: ${formatTime(currentTime)}`);
            
            totalDistance += travelDistance;

            // Travel event
            events.push({
              time: currentTime,
              type: 'travel',
              location: visitIndex === 0 
                ? `Depot ${route.start_depot || 0} → ${clientNameFunction(clientIndex)}`
                : `${clientNameFunction(route.clients[visitIndex - 1])} → ${clientNameFunction(clientIndex)}`,
              duration: travelTime,
              details: `Travel ${(travelDistance / 1000).toFixed(1)} km`
            });
            
            currentTime += travelTime;

            // Arrival event
            events.push({
              time: currentTime,
              type: 'arrival',
              location: clientNameFunction(clientIndex),
              duration: 0,
              details: `Arrival at client`,
              clientIndex: clientIndex
            });

            // Service event
            const serviceTime = client.service_duration || 10;
            events.push({
              time: currentTime,
              type: 'service',
              location: clientNameFunction(clientIndex),
              duration: serviceTime,
              details: `Service - Delivery: ${client.delivery}${client.pickup ? `, Pickup: ${client.pickup}` : ''}${client.prize ? `, Prize: ${client.prize}` : ''}`,
              clientIndex: clientIndex
            });
            
            currentTime += serviceTime;
            currentLocation = client;
            
            console.log(`  Service completed at: ${formatTime(currentTime)}`);
            console.log(`  Current location updated to: Client ${clientIndex}`);
            console.log(`  ---`);
          });
        }
      }

      // Return to depot
      const endDepot = depots[route.end_depot || route.start_depot || 0];
      const returnDistance = calculateDistance(currentLocation, endDepot);
      const returnTime = Math.max(1, Math.round(returnDistance / 1000 / 20 * 60)); // 20km/h, minimum 1 minute
      
      totalDistance += returnDistance;

      // Travel back to depot
      events.push({
        time: currentTime,
        type: 'travel',
        location: route.clients && route.clients.length > 0
          ? `${clientNameFunction(route.clients[route.clients.length - 1])} → Depot ${route.end_depot || route.start_depot || 0}`
          : `Depot ${route.start_depot || 0} → Depot ${route.end_depot || route.start_depot || 0}`,
        duration: returnTime,
        details: `Return ${(returnDistance / 1000).toFixed(1)} km`
      });
      
      currentTime += returnTime;

      // Return to depot
      events.push({
        time: currentTime,
        type: 'return',
        location: `Depot ${route.end_depot || route.start_depot || 0}`,
        duration: 0,
        details: 'Return to depot'
      });

      // Use backend calculated values if available, otherwise use manual calculations
      const finalDistance = route.distance || Math.round(totalDistance);
      const finalDuration = route.duration || (currentTime - startTime);
      
      console.log(`Route ${routeIndex} final stats:`, {
        backendDistance: route.distance,
        calculatedDistance: Math.round(totalDistance),
        finalDistance,
        backendDuration: route.duration,
        calculatedDuration: currentTime - startTime,
        finalDuration
      });

      return {
        routeIndex,
        vehicleType: route.vehicle_type || 0,
        startTime,
        endTime: currentTime,
        events,
        totalDistance: finalDistance,
        totalDuration: finalDuration,
        clientsServed: route.clients || []
      };
    });
  }, [solution, clients, depots, vehicleTypes]);

  // Get event icon and color
  const getEventIcon = (type: ScheduleEvent['type']) => {
    switch (type) {
      case 'departure':
        return <DepotIcon />;
      case 'travel':
        return <TruckIcon />;
      case 'arrival':
        return <LocationIcon />;
      case 'service':
        return <TimeIcon />;
      case 'return':
        return <DepotIcon />;
      default:
        return <LocationIcon />;
    }
  };

  const getEventColor = (type: ScheduleEvent['type']): "success" | "warning" | "info" | "primary" | "error" => {
    switch (type) {
      case 'departure':
        return 'success';
      case 'travel':
        return 'warning';
      case 'arrival':
        return 'info';
      case 'service':
        return 'primary';
      case 'return':
        return 'error';
      default:
        return 'info';
    }
  };

  const getEventLabel = (type: ScheduleEvent['type']): string => {
    switch (type) {
      case 'departure':
        return '出発';
      case 'travel':
        return '移動';
      case 'arrival':
        return '到着';
      case 'service':
        return 'サービス';
      case 'return':
        return '帰着';
      default:
        return '活動';
    }
  };

  if (!solution?.routes || routeSchedules.length === 0) {
    return (
      <Alert severity="info">
        No route data available for schedule display.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Detailed Route Schedules
      </Typography>

      {routeSchedules.map((routeSchedule) => (
        <Accordion key={routeSchedule.routeIndex} sx={{ mb: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                icon={<TruckIcon />}
                label={`Route ${routeSchedule.routeIndex + 1}`}
                color="primary"
                size="small"
              />
              <Typography variant="body2">
                Vehicle Type {routeSchedule.vehicleType} |
                Duration: {formatTime(routeSchedule.totalDuration)} |
                Distance: {(routeSchedule.totalDistance / 1000).toFixed(1)} km |
                Clients: {routeSchedule.clientsServed.length}
              </Typography>
            </Box>
          </AccordionSummary>
          
          <AccordionDetails>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Start Time</TableCell>
                    <TableCell>Activity</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>End Time</TableCell>
                    <TableCell>Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {routeSchedule.events.map((event, eventIndex) => (
                    <TableRow key={eventIndex} sx={{ 
                      '&:hover': { backgroundColor: 'grey.50' },
                      backgroundColor: event.type === 'service' ? 'action.hover' : 'inherit'
                    }}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {formatTime(event.time)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getEventIcon(event.type)}
                          label={getEventLabel(event.type)}
                          color={getEventColor(event.type)}
                          size="small"
                          sx={{ minWidth: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {event.location}
                        </Typography>
                        {event.clientIndex !== undefined && clients[event.clientIndex] && (
                          <Typography variant="caption" color="textSecondary" display="block">
                            Coords: ({clients[event.clientIndex].x}, {clients[event.clientIndex].y})
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {event.duration > 0 ? `${event.duration} min` : '-'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {event.duration > 0 ? formatTime(event.time + event.duration) : '-'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {event.details}
                        </Typography>
                        {event.clientIndex !== undefined && clients[event.clientIndex] && (
                          <Box sx={{ mt: 0.5 }}>
                            {clients[event.clientIndex].tw_early && (
                              <Typography variant="caption" color="textSecondary" display="block">
                                Time Window: {formatTime(clients[event.clientIndex].tw_early)} - {formatTime(clients[event.clientIndex].tw_late)}
                              </Typography>
                            )}
                          </Box>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      ))}

      {/* Summary Statistics */}
      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Solution Summary
        </Typography>
        <Typography variant="body2">
          Total Routes: {routeSchedules.length}
        </Typography>
        <Typography variant="body2">
          Total Distance: {(routeSchedules.reduce((sum, rs) => sum + rs.totalDistance, 0) / 1000).toFixed(1)} km
        </Typography>
        <Typography variant="body2">
          Total Duration: {formatTime(routeSchedules.reduce((sum, rs) => sum + rs.totalDuration, 0))}
        </Typography>
        <Typography variant="body2">
          Total Clients Served: {routeSchedules.reduce((sum, rs) => sum + rs.clientsServed.length, 0)}
        </Typography>
        <Typography variant="body2">
          Earliest Start: {formatTime(Math.min(...routeSchedules.map(rs => rs.startTime)))}
        </Typography>
        <Typography variant="body2">
          Latest End: {formatTime(Math.max(...routeSchedules.map(rs => rs.endTime)))}
        </Typography>
      </Paper>
    </Box>
  );
};

export default PyVRPScheduleViewer;