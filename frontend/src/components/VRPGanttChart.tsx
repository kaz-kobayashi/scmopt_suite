import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControlLabel,
  Switch,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
} from '@mui/material';
import {
  LocalShipping as TruckIcon,
  Schedule as TimeIcon,
  LocationOn as LocationIcon,
  RestoreFromTrash as ReloadIcon,
  Coffee as BreakIcon,
  AccessTime as ServiceIcon,
} from '@mui/icons-material';

interface GanttTask {
  id: string;
  type: 'travel' | 'service' | 'wait' | 'reload' | 'break';
  label: string;
  startTime: number; // minutes from start
  endTime: number; // minutes from start
  duration: number; // minutes
  location?: string;
  clientIndex?: number;
  details?: string;
}

interface GanttRoute {
  routeId: number;
  vehicleType: number;
  vehicleId: number;
  startDepot: number;
  endDepot: number;
  totalDuration: number;
  tasks: GanttTask[];
  color: string;
}

interface VRPGanttChartProps {
  solution: any;
  clients: any[];
  depots: any[];
  vehicleTypes: any[];
}

const VRPGanttChart: React.FC<VRPGanttChartProps> = ({
  solution,
  clients,
  depots,
  vehicleTypes
}) => {
  const [showWaitTimes, setShowWaitTimes] = useState(true);
  const [showServiceTimes, setShowServiceTimes] = useState(true);
  const [showTravelTimes, setShowTravelTimes] = useState(true);
  const [showReloads, setShowReloads] = useState(true);
  const [showBreaks, setShowBreaks] = useState(true);
  const [timeScale, setTimeScale] = useState(1); // 1 = 1 hour per 60px
  const [selectedRoute, setSelectedRoute] = useState<number | null>(null);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [showTaskDetails, setShowTaskDetails] = useState(false);
  const [viewMode, setViewMode] = useState<'gantt' | 'timeline' | 'table'>('timeline');

  // Color scheme for different task types
  const taskColors = {
    travel: '#FF6B6B',
    service: '#4ECDC4',
    wait: '#FFE066',
    reload: '#45B7D1',
    break: '#96CEB4'
  };

  const routeColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#48dbfb', '#ff9ff3', '#54a0ff'];

  // Helper function to calculate haversine distance between two locations
  const calculateDistance = (loc1: any, loc2: any): number => {
    const lon1 = (loc1.x || 0) / 100;
    const lat1 = (loc1.y || 0) / 100;
    const lon2 = (loc2.x || 0) / 100;
    const lat2 = (loc2.y || 0) / 100;
    
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

  // Format time for display
  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  // Test function to validate task sequence
  const validateTaskSequence = (tasks: GanttTask[], routeId: number) => {
    let prevEndTime: number | null = null;
    let errors: string[] = [];
    
    tasks.forEach((task, index) => {
      // Check if task starts after previous task ends
      if (prevEndTime !== null && task.startTime !== prevEndTime) {
        errors.push(`Route ${routeId}, Task ${index} (${task.label}): Gap detected. Expected start: ${formatTime(prevEndTime)}, Actual start: ${formatTime(task.startTime)}`);
      }
      
      // Check if duration matches start/end time
      if (task.endTime - task.startTime !== task.duration) {
        errors.push(`Route ${routeId}, Task ${index} (${task.label}): Duration mismatch. Duration: ${task.duration}, Calculated: ${task.endTime - task.startTime}`);
      }
      
      prevEndTime = task.endTime;
    });
    
    if (errors.length > 0) {
      console.error('Schedule validation errors:', errors);
    }
    
    return errors.length === 0;
  };

  // Calculate detailed timing for each route
  const ganttRoutes: GanttRoute[] = useMemo(() => {
    if (!solution?.routes) return [];

    return solution.routes.map((route: any, routeIndex: number) => {
      const tasks: GanttTask[] = [];
      // Start at realistic working hours (8:00 AM = 480 minutes)
      const vehicleType = vehicleTypes[route.vehicle_type || 0];
      let currentTime = vehicleType?.tw_early || 480; // Default to 8:00 AM

      // Start at depot
      const startDepot = depots[route.start_depot] || depots[0];
      let currentLocation = startDepot;
      let currentLocationLabel = `Depot ${route.start_depot || 0}`;
      let previousClientIndex: number | null = null;

      // Add each client visit
      route.clients?.forEach((clientIndex: number, visitIndex: number) => {
        const client = clients[clientIndex];
        if (!client) return;

        // Calculate travel time to client
        const travelDistance = calculateDistance(currentLocation, client);
        const travelTime = Math.round(travelDistance / 1000 / 20 * 60); // Assume 20km/h speed for Tokyo
        
        // Debug: Log for all movements to diagnose issues
        if (routeIndex === 0) {
          console.log(`Route ${routeIndex}, visit ${visitIndex}: From ${currentLocationLabel} to Client ${clientIndex}, Distance ${(travelDistance/1000).toFixed(2)}km, Time ${travelTime}min, CurrentTime: ${formatTime(currentTime)}`);
        }

        // Add travel task (always add if there's actual movement, even if time rounds to 0)
        // Only skip if it's the exact same location (distance = 0)
        if (travelDistance > 0) {
          tasks.push({
            id: `route-${routeIndex}-travel-${visitIndex}`,
            type: 'travel',
            label: `${currentLocationLabel} → Client ${clientIndex}`,
            startTime: currentTime,
            endTime: currentTime + Math.max(travelTime, 1), // At least 1 minute for any movement
            duration: Math.max(travelTime, 1),
            location: `${currentLocationLabel} → Client ${clientIndex}`,
            details: `${(travelDistance / 1000).toFixed(1)} km`
          });
          currentTime += Math.max(travelTime, 1);
        }

        // Check for waiting time (if arriving before time window)
        const arrivalTime = currentTime;
        const timeWindowStart = client.tw_early || 0;
        const waitTime = Math.max(0, timeWindowStart - arrivalTime);

        if (waitTime > 0) {
          tasks.push({
            id: `route-${routeIndex}-wait-${visitIndex}`,
            type: 'wait',
            label: `Wait at Client ${clientIndex}`,
            startTime: currentTime,
            endTime: currentTime + waitTime,
            duration: waitTime,
            location: `Client ${clientIndex}`,
            clientIndex,
            details: `Wait for time window`
          });
          currentTime += waitTime;
        }

        // Add service task
        const serviceTime = client.service_duration || 10;
        tasks.push({
          id: `route-${routeIndex}-service-${visitIndex}`,
          type: 'service',
          label: `Service Client ${clientIndex}`,
          startTime: currentTime,
          endTime: currentTime + serviceTime,
          duration: serviceTime,
          location: `Client ${clientIndex}`,
          clientIndex,
          details: `Delivery: ${Array.isArray(client.delivery) ? client.delivery.join(', ') : client.delivery}`
        });
        currentTime += serviceTime;

        // Update current location immediately after service completion
        currentLocation = client;
        currentLocationLabel = `Client ${clientIndex}`;
        previousClientIndex = clientIndex;

        // Add reload if specified in route (simplified logic)
        if (route.reloads && visitIndex < route.reloads.length) {
          const reloadTime = 30; // Assume 30 min reload
          tasks.push({
            id: `route-${routeIndex}-reload-${visitIndex}`,
            type: 'reload',
            label: `Reload at Depot`,
            startTime: currentTime,
            endTime: currentTime + reloadTime,
            duration: reloadTime,
            location: `Depot ${route.start_depot}`,
            details: `Vehicle reload`
          });
          currentTime += reloadTime;
        }

        // Add break if needed (every 4 hours of work)
        if (currentTime > 0 && currentTime % 240 < serviceTime) {
          const breakTime = 30;
          tasks.push({
            id: `route-${routeIndex}-break-${visitIndex}`,
            type: 'break',
            label: `Driver Break`,
            startTime: currentTime,
            endTime: currentTime + breakTime,
            duration: breakTime,
            location: `Client ${clientIndex}`,
            details: `Required break`
          });
          currentTime += breakTime;
        }
      });

      // Return to depot
      const endDepot = depots[route.end_depot] || depots[route.start_depot] || depots[0];
      const returnDistance = calculateDistance(currentLocation, endDepot);
      const returnTime = Math.round(returnDistance / 1000 / 20 * 60); // Assume 20km/h speed for Tokyo

      if (returnDistance > 0) {
        tasks.push({
          id: `route-${routeIndex}-return`,
          type: 'travel',
          label: `${currentLocationLabel} → Depot`,
          startTime: currentTime,
          endTime: currentTime + Math.max(returnTime, 1),
          duration: Math.max(returnTime, 1),
          location: `${currentLocationLabel} → Depot ${route.end_depot || route.start_depot}`,
          details: `${(returnDistance / 1000).toFixed(1)} km`
        });
        currentTime += Math.max(returnTime, 1);
      }

      // Validate task sequence
      validateTaskSequence(tasks, routeIndex);
      
      return {
        routeId: routeIndex,
        vehicleType: route.vehicle_type || 0,
        vehicleId: route.vehicle_id || routeIndex,
        startDepot: route.start_depot || 0,
        endDepot: route.end_depot || route.start_depot || 0,
        totalDuration: currentTime - (vehicleType?.tw_early || 480), // Duration from start time
        tasks: tasks,
        color: routeColors[routeIndex % routeColors.length]
      };
    });
  }, [solution, clients, depots]);

  // Calculate chart dimensions
  const maxDuration = Math.max(...ganttRoutes.map(r => r.totalDuration), 60);
  const chartWidth = Math.max(800, maxDuration * timeScale);
  const rowHeight = viewMode === 'timeline' ? 120 : 60; // More space for timeline view
  const chartHeight = ganttRoutes.length * rowHeight;

  // Filter tasks based on display options
  const shouldShowTask = (task: GanttTask): boolean => {
    switch (task.type) {
      case 'travel': return showTravelTimes;
      case 'service': return showServiceTimes;
      case 'wait': return showWaitTimes;
      case 'reload': return showReloads;
      case 'break': return showBreaks;
      default: return true;
    }
  };

  // Get task icon
  const getTaskIcon = (taskType: string) => {
    switch (taskType) {
      case 'travel': return <TruckIcon fontSize="small" />;
      case 'service': return <ServiceIcon fontSize="small" />;
      case 'wait': return <TimeIcon fontSize="small" />;
      case 'reload': return <ReloadIcon fontSize="small" />;
      case 'break': return <BreakIcon fontSize="small" />;
      default: return <LocationIcon fontSize="small" />;
    }
  };

  if (!solution?.routes || ganttRoutes.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="textSecondary">
          No route data available for Gantt chart
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Solve a VRP problem first to see the schedule visualization
        </Typography>
      </Paper>
    );
  }

  return (
    <Box>
      {/* Controls */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Route Schedule Visualization
        </Typography>

        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
              <Typography variant="body2">View Mode:</Typography>
              <Button
                variant={viewMode === 'table' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewMode('table')}
              >
                詳細テーブル
              </Button>
              <Button
                variant={viewMode === 'timeline' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewMode('timeline')}
              >
                タイムライン
              </Button>
              <Button
                variant={viewMode === 'gantt' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewMode('gantt')}
              >
                ガントチャート
              </Button>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <FormControlLabel
                control={<Switch size="small" checked={showTravelTimes} onChange={(e) => setShowTravelTimes(e.target.checked)} />}
                label="Travel"
              />
              <FormControlLabel
                control={<Switch size="small" checked={showServiceTimes} onChange={(e) => setShowServiceTimes(e.target.checked)} />}
                label="Service"
              />
              <FormControlLabel
                control={<Switch size="small" checked={showWaitTimes} onChange={(e) => setShowWaitTimes(e.target.checked)} />}
                label="Wait"
              />
              <FormControlLabel
                control={<Switch size="small" checked={showReloads} onChange={(e) => setShowReloads(e.target.checked)} />}
                label="Reload"
              />
              <FormControlLabel
                control={<Switch size="small" checked={showBreaks} onChange={(e) => setShowBreaks(e.target.checked)} />}
                label="Breaks"
              />
            </Box>
          </Grid>

          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Focus Route</InputLabel>
                <Select
                  value={selectedRoute ?? ''}
                  onChange={(e) => setSelectedRoute(e.target.value === '' ? null : Number(e.target.value))}
                  displayEmpty
                >
                  <MenuItem value="">All Routes</MenuItem>
                  {ganttRoutes.map((route, index) => (
                    <MenuItem key={index} value={index}>Route {index + 1}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ width: 120 }}>
                <Typography variant="caption">Time Scale</Typography>
                <Slider
                  value={timeScale}
                  onChange={(_, value) => setTimeScale(value as number)}
                  min={0.5}
                  max={3}
                  step={0.5}
                  size="small"
                  marks={[
                    { value: 0.5, label: '0.5x' },
                    { value: 1, label: '1x' },
                    { value: 2, label: '2x' },
                    { value: 3, label: '3x' }
                  ]}
                />
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Table View - Detailed Schedule */}
      {viewMode === 'table' && (
        <Paper sx={{ mb: 2 }}>
          <Typography variant="h6" sx={{ p: 2, pb: 1 }}>
            詳細スケジュール - 到着・出発時刻一覧
          </Typography>
          {ganttRoutes.map((route, routeIndex) => {
            if (selectedRoute !== null && selectedRoute !== routeIndex) return null;
            
            return (
              <Box key={route.routeId} sx={{ mb: 3 }}>
                <Box sx={{ px: 2, py: 1, backgroundColor: 'grey.50' }}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    Route {routeIndex + 1} - 車両タイプ {route.vehicleType}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    総所要時間: {formatTime(route.totalDuration)} | 顧客数: {route.tasks.filter(t => t.type === 'service').length}件
                  </Typography>
                </Box>
                
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>時刻</TableCell>
                        <TableCell>活動</TableCell>
                        <TableCell>場所</TableCell>
                        <TableCell>所要時間</TableCell>
                        <TableCell>累積時間</TableCell>
                        <TableCell>詳細</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>{formatTime(vehicleTypes[route.vehicleType]?.tw_early || 480)}</TableCell>
                        <TableCell>
                          <Chip
                            icon={<TruckIcon />}
                            label="出発"
                            size="small"
                            sx={{ backgroundColor: '#4CAF50', color: 'white' }}
                          />
                        </TableCell>
                        <TableCell>Depot {route.startDepot}</TableCell>
                        <TableCell>-</TableCell>
                        <TableCell>{formatTime(vehicleTypes[route.vehicleType]?.tw_early || 480)}</TableCell>
                        <TableCell>拠点から出発</TableCell>
                      </TableRow>
                      
                      {route.tasks.map((task, taskIndex) => (
                        <TableRow key={task.id} sx={{ '&:hover': { backgroundColor: 'grey.50' } }}>
                          <TableCell>
                            <Typography variant="body2" fontWeight="bold">
                              {formatTime(task.startTime)}
                            </Typography>
                            {task.endTime > task.startTime && (
                              <Typography variant="caption" color="textSecondary">
                                ↓ {formatTime(task.endTime)}
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            <Chip
                              icon={getTaskIcon(task.type)}
                              label={task.type === 'travel' ? '移動' : 
                                     task.type === 'service' ? 'サービス' :
                                     task.type === 'wait' ? '待機' :
                                     task.type === 'reload' ? '補給' : '休憩'}
                              size="small"
                              sx={{ 
                                backgroundColor: taskColors[task.type], 
                                color: 'white',
                                minWidth: 80
                              }}
                            />
                          </TableCell>
                          <TableCell>
                            {task.type === 'travel' ? (
                              <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                                {task.location}
                              </Typography>
                            ) : (
                              <Typography variant="body2">
                                {task.location}
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {task.duration}分
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="textSecondary">
                              {formatTime(task.endTime)}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {task.details}
                            </Typography>
                            {task.clientIndex !== undefined && clients[task.clientIndex] && (
                              <Typography variant="caption" color="textSecondary" display="block">
                                配送量: {Array.isArray(clients[task.clientIndex].delivery) 
                                  ? clients[task.clientIndex].delivery.join(', ') 
                                  : clients[task.clientIndex].delivery}
                              </Typography>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                      
                      <TableRow sx={{ backgroundColor: 'grey.100' }}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {formatTime(route.totalDuration)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            icon={<TruckIcon />}
                            label="帰着"
                            size="small"
                            sx={{ backgroundColor: '#FF9800', color: 'white' }}
                          />
                        </TableCell>
                        <TableCell>Depot {route.endDepot}</TableCell>
                        <TableCell>-</TableCell>
                        <TableCell>{formatTime(route.totalDuration)}</TableCell>
                        <TableCell>拠点に帰着</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            );
          })}
        </Paper>
      )}

      {/* Timeline/Gantt Chart */}
      {(viewMode === 'timeline' || viewMode === 'gantt') && (
        <Paper sx={{ p: 2, overflow: 'auto' }}>
        <Typography variant="h6" gutterBottom>
          Route Schedule Timeline
        </Typography>

        <Box sx={{ position: 'relative', minWidth: chartWidth, minHeight: chartHeight + (viewMode === 'timeline' ? 70 : 50) }}>
          {/* Time axis */}
          <Box sx={{ position: 'absolute', top: 0, left: 200, right: 0, height: viewMode === 'timeline' ? 50 : 30, borderBottom: 1, borderColor: 'divider' }}>
            {Array.from({ length: Math.ceil(maxDuration / (viewMode === 'timeline' ? 30 : 60)) + 1 }, (_, i) => (
              <Box
                key={i}
                sx={{
                  position: 'absolute',
                  left: i * (viewMode === 'timeline' ? 30 : 60) * timeScale,
                  top: 0,
                  height: viewMode === 'timeline' ? 50 : 30,
                  borderLeft: 1,
                  borderColor: 'divider',
                  pl: 1
                }}
              >
                <Typography variant="caption" color="textSecondary">
                  {formatTime(i * (viewMode === 'timeline' ? 30 : 60))}
                </Typography>
                {viewMode === 'timeline' && (
                  <Typography variant="caption" display="block" color="textSecondary" sx={{ fontSize: '0.6rem' }}>
                    {i * 30}min
                  </Typography>
                )}
              </Box>
            ))}
          </Box>

          {/* Routes */}
          {ganttRoutes.map((route, routeIndex) => {
            if (selectedRoute !== null && selectedRoute !== routeIndex) return null;

            const yPosition = routeIndex * rowHeight + (viewMode === 'timeline' ? 60 : 40);

            return (
              <Box key={route.routeId}>
                {/* Route label */}
                <Box
                  sx={{
                    position: 'absolute',
                    left: 0,
                    top: yPosition,
                    width: 190,
                    height: rowHeight - 10,
                    display: 'flex',
                    alignItems: 'center',
                    pr: 2
                  }}
                >
                  <Chip
                    icon={<TruckIcon />}
                    label={`Route ${routeIndex + 1}`}
                    size="small"
                    sx={{ 
                      backgroundColor: route.color, 
                      color: 'white',
                      mr: 1
                    }}
                  />
                  <Box>
                    <Typography variant="caption" color="textSecondary">
                      {formatTime(route.totalDuration)}
                    </Typography>
                    {viewMode === 'timeline' && (
                      <Typography variant="caption" display="block" color="textSecondary">
                        {route.tasks.filter(t => t.type === 'service').length}件配送
                      </Typography>
                    )}
                  </Box>
                </Box>

                {/* Tasks */}
                {route.tasks.map((task) => {
                  if (!shouldShowTask(task)) return null;

                  const taskWidth = Math.max(task.duration * timeScale, viewMode === 'timeline' ? 30 : 20);
                  const taskLeft = 200 + task.startTime * timeScale;
                  const taskHeight = viewMode === 'timeline' ? 35 : rowHeight - 20;

                  return (
                    <Tooltip
                      key={task.id}
                      title={
                        <Box>
                          <Typography variant="subtitle2">{task.label}</Typography>
                          <Typography variant="body2">
                            {formatTime(task.startTime)} - {formatTime(task.endTime)} ({task.duration} min)
                          </Typography>
                          {task.details && (
                            <Typography variant="body2">{task.details}</Typography>
                          )}
                          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                            Click for more details
                          </Typography>
                        </Box>
                      }
                    >
                      <Box
                        onClick={() => {
                          setSelectedTask(task.id);
                          setShowTaskDetails(true);
                        }}
                        sx={{
                          position: 'absolute',
                          left: taskLeft,
                          top: yPosition + (viewMode === 'timeline' ? 15 : 5),
                          width: taskWidth,
                          height: taskHeight,
                          backgroundColor: taskColors[task.type],
                          borderRadius: 1,
                          border: selectedTask === task.id ? 3 : 1,
                          borderColor: selectedTask === task.id ? 'primary.main' : 'rgba(0,0,0,0.1)',
                          display: 'flex',
                          flexDirection: viewMode === 'timeline' ? 'column' : 'row',
                          alignItems: 'center',
                          justifyContent: 'center',
                          px: viewMode === 'timeline' ? 0.5 : 1,
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            opacity: 0.8,
                            transform: 'scale(1.02)',
                            zIndex: 10,
                            boxShadow: 3
                          }
                        }}
                      >
                        {getTaskIcon(task.type)}
                        {viewMode === 'timeline' ? (
                          taskWidth > 30 && (
                            <Typography variant="caption" sx={{ color: 'white', fontWeight: 'bold', fontSize: '0.6rem', textAlign: 'center' }}>
                              {task.type === 'travel' ? '移動' : 
                               task.type === 'service' ? `C${task.clientIndex}` :
                               task.type === 'wait' ? '待機' :
                               task.type === 'reload' ? '補給' : '休憩'}
                            </Typography>
                          )
                        ) : (
                          taskWidth > 60 && (
                            <Typography variant="caption" sx={{ ml: 0.5, color: 'white', fontWeight: 'bold' }}>
                              {task.type}
                            </Typography>
                          )
                        )}
                      </Box>
                    </Tooltip>
                  );
                })}
              </Box>
            );
          })}
        </Box>
      </Paper>
      )}

      {/* Summary Table */}
      <Paper sx={{ mt: 2 }}>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Route</TableCell>
                <TableCell>Vehicle</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Clients</TableCell>
                <TableCell>Travel Time</TableCell>
                <TableCell>Service Time</TableCell>
                <TableCell>Wait Time</TableCell>
                <TableCell>Other</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {ganttRoutes.map((route, index) => {
                const travelTime = route.tasks.filter(t => t.type === 'travel').reduce((sum, t) => sum + t.duration, 0);
                const serviceTime = route.tasks.filter(t => t.type === 'service').reduce((sum, t) => sum + t.duration, 0);
                const waitTime = route.tasks.filter(t => t.type === 'wait').reduce((sum, t) => sum + t.duration, 0);
                const otherTime = route.totalDuration - travelTime - serviceTime - waitTime;

                return (
                  <TableRow 
                    key={index}
                    onClick={() => setSelectedRoute(selectedRoute === index ? null : index)}
                    sx={{ 
                      cursor: 'pointer',
                      backgroundColor: selectedRoute === index ? 'action.selected' : 'inherit'
                    }}
                  >
                    <TableCell>
                      <Chip
                        label={`Route ${index + 1}`}
                        size="small"
                        sx={{ backgroundColor: route.color, color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>Type {route.vehicleType}</TableCell>
                    <TableCell>{formatTime(route.totalDuration)}</TableCell>
                    <TableCell>{route.tasks.filter(t => t.type === 'service').length}</TableCell>
                    <TableCell>{formatTime(travelTime)}</TableCell>
                    <TableCell>{formatTime(serviceTime)}</TableCell>
                    <TableCell>{formatTime(waitTime)}</TableCell>
                    <TableCell>{formatTime(otherTime)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Task Details Dialog */}
      <Dialog
        open={showTaskDetails}
        onClose={() => {
          setShowTaskDetails(false);
          setSelectedTask(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Task Details
        </DialogTitle>
        <DialogContent>
          {selectedTask && (() => {
            const task = ganttRoutes
              .flatMap(route => route.tasks)
              .find(t => t.id === selectedTask);
            
            if (!task) return null;

            return (
              <Box>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      {task.label}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Type
                    </Typography>
                    <Chip
                      icon={getTaskIcon(task.type)}
                      label={task.type.charAt(0).toUpperCase() + task.type.slice(1)}
                      sx={{ backgroundColor: taskColors[task.type], color: 'white' }}
                    />
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Duration
                    </Typography>
                    <Typography variant="body1">
                      {task.duration} minutes
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      Start Time
                    </Typography>
                    <Typography variant="body1">
                      {formatTime(task.startTime)}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">
                      End Time
                    </Typography>
                    <Typography variant="body1">
                      {formatTime(task.endTime)}
                    </Typography>
                  </Grid>
                  
                  {task.location && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="textSecondary">
                        Location
                      </Typography>
                      <Typography variant="body1">
                        {task.location}
                      </Typography>
                    </Grid>
                  )}
                  
                  {task.details && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="textSecondary">
                        Details
                      </Typography>
                      <Typography variant="body1">
                        {task.details}
                      </Typography>
                    </Grid>
                  )}
                  
                  {task.clientIndex !== undefined && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="textSecondary">
                        Client Information
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2">
                          Client Index: {task.clientIndex}
                        </Typography>
                        {clients[task.clientIndex] && (
                          <>
                            <Typography variant="body2">
                              Coordinates: ({clients[task.clientIndex].x}, {clients[task.clientIndex].y})
                            </Typography>
                            <Typography variant="body2">
                              Delivery: {Array.isArray(clients[task.clientIndex].delivery) 
                                ? clients[task.clientIndex].delivery.join(', ') 
                                : clients[task.clientIndex].delivery}
                            </Typography>
                            <Typography variant="body2">
                              Service Duration: {clients[task.clientIndex].service_duration} min
                            </Typography>
                          </>
                        )}
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </Box>
            );
          })()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setShowTaskDetails(false);
            setSelectedTask(null);
          }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VRPGanttChart;