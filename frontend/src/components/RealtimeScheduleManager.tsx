import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  LinearProgress,
  Divider,
  Paper,
  Badge,
  Snackbar
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
  Info as InfoIcon,
  Add as AddIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Timeline as TimelineIcon,
  Build as BuildIcon,
  Schedule as ScheduleIcon,
  Speed as SpeedIcon,
  Notifications as NotificationIcon,
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon
} from '@mui/icons-material';
import { useWebSocket } from '../hooks/useWebSocket';
import ReoptimizationManager from './ReoptimizationManager';

interface RealtimeScheduleManagerProps {
  solutionData: any;
  onUpdate?: () => void;
}

interface ScheduleEvent {
  id: string;
  event_type: string;
  timestamp: string;
  target_id: string;
  description: string;
  event_data: any;
  impact_level: string;
  auto_reoptimize: boolean;
  processed: boolean;
}

interface RealtimeStats {
  schedule_id: string;
  current_time: string;
  completed_jobs: number;
  active_jobs: number;
  delayed_jobs: number;
  machine_utilization: Record<string, number>;
  critical_path_status: string;
  estimated_completion: string;
  kpi_metrics: Record<string, number>;
}

const RealtimeScheduleManager: React.FC<RealtimeScheduleManagerProps> = ({
  solutionData,
  onUpdate
}) => {
  const [scheduleId, setScheduleId] = useState<string>('');
  const [isActive, setIsActive] = useState(false);
  const [events, setEvents] = useState<ScheduleEvent[]>([]);
  const [realtimeStats, setRealtimeStats] = useState<RealtimeStats | null>(null);
  const [showEventDialog, setShowEventDialog] = useState(false);
  const [loading, setLoading] = useState(false);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [notifications, setNotifications] = useState<string[]>([]);
  const [showNotification, setShowNotification] = useState(false);

  // Event form state
  const [eventType, setEventType] = useState('job_delay');
  const [targetId, setTargetId] = useState('');
  const [description, setDescription] = useState('');
  const [impactLevel, setImpactLevel] = useState('medium');
  const [eventData, setEventData] = useState<any>({});

  // Generate unique client ID
  const clientId = useMemo(() => `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`, []);

  // WebSocket connection
  const {
    isConnected,
    lastMessage,
    subscribe,
    unsubscribe,
    requestStats
  } = useWebSocket(`ws://localhost:8000/ws/${clientId}`, {
    onMessage: (message) => {
      handleWebSocketMessage(message);
    },
    onOpen: () => {
      console.log('WebSocket connected');
      // Subscribe to current schedule if active
      if (scheduleId && isActive) {
        subscribe(scheduleId);
      }
    },
    onClose: () => {
      console.log('WebSocket disconnected');
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
      setErrorMessage('WebSocket接続エラーが発生しました');
    }
  });

  // Handle WebSocket messages
  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'event_added':
        // 新しいイベントを追加
        fetchEvents();
        addNotification(`新しいイベント: ${message.event.description}`);
        break;
      
      case 'stats_update':
        // 統計情報を更新
        if (message.stats) {
          setRealtimeStats(message.stats);
        }
        break;
      
      case 'alert':
        // アラートを表示
        addNotification(`アラート (${message.alert.severity}): ${message.alert.message}`);
        break;
      
      case 'reoptimization_started':
        setLoading(true);
        addNotification('再最適化を開始しました...');
        break;
      
      case 'reoptimization_completed':
        setLoading(false);
        if (message.success) {
          addNotification('再最適化が完了しました');
          if (onUpdate) onUpdate();
        } else {
          addNotification('再最適化に失敗しました');
        }
        break;
    }
  };

  const addNotification = (message: string) => {
    setNotifications(prev => [...prev, message]);
    setShowNotification(true);
  };

  // Auto-refresh interval (WebSocketが無効な場合のフォールバック)
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isActive && autoUpdate && scheduleId && !isConnected) {
      interval = setInterval(() => {
        fetchScheduleStats();
        fetchEvents();
      }, 5000); // Update every 5 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isActive, autoUpdate, scheduleId, isConnected]);

  // Subscribe to schedule when it becomes active
  useEffect(() => {
    if (scheduleId && isActive && isConnected) {
      subscribe(scheduleId);
    }
    
    return () => {
      if (scheduleId && isConnected) {
        unsubscribe(scheduleId);
      }
    };
  }, [scheduleId, isActive, isConnected]);

  const startRealtimeSchedule = async () => {
    if (!solutionData) {
      setErrorMessage('ソリューションデータがありません。まず最適化を実行してください。');
      return;
    }

    setLoading(true);
    try {
      console.log('Sending solution data:', solutionData);
      
      const response = await fetch('http://localhost:8000/api/realtime/schedules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(solutionData)
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API Error Response:', errorText);
        throw new Error(`リアルタイムスケジュール作成に失敗しました (${response.status}): ${errorText}`);
      }

      const newScheduleId = await response.text();
      setScheduleId(newScheduleId.replace(/"/g, '')); // Remove quotes
      setIsActive(true);
      setErrorMessage('');
      
      // Initial data fetch
      await fetchScheduleStats();
      await fetchEvents();
    } catch (error) {
      console.error('Error starting realtime schedule:', error);
      setErrorMessage(`リアルタイムスケジュール開始エラー: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  const stopRealtimeSchedule = async () => {
    if (!scheduleId) return;

    try {
      await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}`, {
        method: 'DELETE'
      });

      setIsActive(false);
      setScheduleId('');
      setEvents([]);
      setRealtimeStats(null);
    } catch (error) {
      setErrorMessage(`Error stopping schedule: ${error}`);
    }
  };

  const fetchScheduleStats = async () => {
    if (!scheduleId) return;

    try {
      const response = await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}/stats`);
      if (response.ok) {
        const stats = await response.json();
        setRealtimeStats(stats);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchEvents = async () => {
    if (!scheduleId) return;

    try {
      const response = await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}/events?include_history=true&limit=50`);
      if (response.ok) {
        const eventList = await response.json();
        setEvents(eventList);
      }
    } catch (error) {
      console.error('Error fetching events:', error);
    }
  };

  const addEvent = async () => {
    if (!scheduleId || !targetId) return;

    setLoading(true);
    try {
      const eventPayload = {
        event_type: eventType,
        target_id: targetId,
        description: description,
        event_data: eventData,
        impact_level: impactLevel,
        auto_reoptimize: impactLevel === 'critical' || impactLevel === 'high'
      };

      const response = await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}/events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(eventPayload)
      });

      if (!response.ok) {
        throw new Error('Failed to add event');
      }

      setShowEventDialog(false);
      resetEventForm();
      await fetchEvents();
      await fetchScheduleStats();
    } catch (error) {
      setErrorMessage(`Error adding event: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const addJobDelayEvent = async (jobId: string, operationId: string, originalDuration: number, newDuration: number, reason: string) => {
    if (!scheduleId) return;

    try {
      const response = await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}/events/job-delay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          operation_id: operationId,
          original_duration: originalDuration,
          new_duration: newDuration,
          delay_reason: reason,
          impact_level: newDuration > originalDuration * 1.5 ? 'high' : 'medium'
        })
      });

      if (response.ok) {
        await fetchEvents();
        await fetchScheduleStats();
      }
    } catch (error) {
      setErrorMessage(`Error adding job delay: ${error}`);
    }
  };

  const triggerManualReoptimization = async () => {
    if (!scheduleId) return;

    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/realtime/schedules/${scheduleId}/reoptimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reoptimization_type: 'incremental',
          time_limit: 60,
          preserve_completed_jobs: true
        })
      });

      if (!response.ok) {
        throw new Error('Failed to trigger reoptimization');
      }

      const result = await response.json();
      setErrorMessage('');
      
      // Refresh data
      await fetchScheduleStats();
      await fetchEvents();
      
      if (onUpdate) {
        onUpdate();
      }
    } catch (error) {
      setErrorMessage(`Error during reoptimization: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const resetEventForm = () => {
    setEventType('job_delay');
    setTargetId('');
    setDescription('');
    setImpactLevel('medium');
    setEventData({});
  };

  const getImpactLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getEventTypeIcon = (type: string) => {
    switch (type) {
      case 'job_delay': return <ScheduleIcon />;
      case 'machine_breakdown': return <BuildIcon />;
      case 'urgent_job_add': return <SpeedIcon />;
      case 'job_completion': return <SuccessIcon />;
      case 'priority_change': return <TimelineIcon />;
      default: return <InfoIcon />;
    }
  };

  const formatEventType = (type: string) => {
    const typeMap: Record<string, string> = {
      'job_delay': '作業遅延',
      'machine_breakdown': '機械故障',
      'urgent_job_add': '緊急ジョブ追加',
      'job_completion': '作業完了',
      'priority_change': '優先度変更',
      'resource_change': 'リソース変更'
    };
    return typeMap[type] || type;
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        リアルタイムスケジュール管理
      </Typography>

      {errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errorMessage}
        </Alert>
      )}

      {/* WebSocket接続状態 */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {isConnected ? (
          <Chip
            icon={<WifiIcon />}
            label="WebSocket接続中"
            color="success"
            size="small"
            sx={{ mr: 2 }}
          />
        ) : (
          <Chip
            icon={<WifiOffIcon />}
            label="WebSocket未接続"
            color="error"
            size="small"
            sx={{ mr: 2 }}
          />
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Control Panel */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                スケジュール制御
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                {!isActive ? (
                  <Button
                    variant="contained"
                    startIcon={<StartIcon />}
                    onClick={startRealtimeSchedule}
                    disabled={loading || !solutionData}
                  >
                    リアルタイム開始
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    color="secondary"
                    startIcon={<StopIcon />}
                    onClick={stopRealtimeSchedule}
                    disabled={loading}
                  >
                    停止
                  </Button>
                )}
                
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => setShowEventDialog(true)}
                  disabled={!isActive}
                >
                  イベント追加
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={triggerManualReoptimization}
                  disabled={!isActive || loading}
                >
                  再最適化
                </Button>
              </Box>

              <FormControlLabel
                control={
                  <Switch
                    checked={autoUpdate}
                    onChange={(e) => setAutoUpdate(e.target.checked)}
                  />
                }
                label="自動更新"
                disabled={!isActive}
              />

              {loading && <LinearProgress sx={{ mt: 2 }} />}
              
              {scheduleId && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Schedule ID: {scheduleId}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Real-time Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                リアルタイム統計
                {isActive && <Chip label="ACTIVE" color="success" size="small" sx={{ ml: 1 }} />}
              </Typography>

              {realtimeStats ? (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">完了ジョブ</Typography>
                    <Typography variant="h6">{realtimeStats.completed_jobs}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">アクティブジョブ</Typography>
                    <Typography variant="h6">{realtimeStats.active_jobs}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">遅延ジョブ</Typography>
                    <Typography variant="h6" color="error.main">{realtimeStats.delayed_jobs}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">クリティカルパス</Typography>
                    <Typography variant="h6">{realtimeStats.critical_path_status}</Typography>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      機械稼働率
                    </Typography>
                    {Object.entries(realtimeStats.machine_utilization).map(([machineId, utilization]) => (
                      <Box key={machineId} sx={{ mb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2">{machineId}</Typography>
                          <Typography variant="body2">{(utilization * 100).toFixed(1)}%</Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={utilization * 100} 
                          color={utilization > 0.9 ? 'error' : utilization > 0.7 ? 'warning' : 'primary'}
                        />
                      </Box>
                    ))}
                  </Grid>
                </Grid>
              ) : (
                <Typography color="text.secondary">
                  {isActive ? 'Loading stats...' : 'スケジュールを開始してください'}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Event Timeline */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  イベントタイムライン
                  <Badge badgeContent={events.filter(e => !e.processed).length} color="error" sx={{ ml: 2 }}>
                    <NotificationIcon />
                  </Badge>
                </Typography>
                
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={fetchEvents}
                  disabled={!isActive}
                >
                  更新
                </Button>
              </Box>

              {events.length === 0 ? (
                <Typography color="text.secondary">
                  イベントはありません
                </Typography>
              ) : (
                <List>
                  {events.slice(0, 10).map((event) => (
                    <Paper key={event.id} elevation={1} sx={{ mb: 1 }}>
                      <ListItem>
                        <Box sx={{ mr: 2 }}>
                          {getEventTypeIcon(event.event_type)}
                        </Box>
                        
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2">
                                {formatEventType(event.event_type)}
                              </Typography>
                              <Chip 
                                label={event.impact_level} 
                                size="small" 
                                color={getImpactLevelColor(event.impact_level) as any}
                              />
                              {!event.processed && (
                                <Chip label="未処理" size="small" color="warning" />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                {event.description}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Target: {event.target_id} | {new Date(event.timestamp).toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    </Paper>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Reoptimization Manager */}
        {isActive && scheduleId && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <ReoptimizationManager
                  scheduleId={scheduleId}
                  currentSolution={solutionData}
                  onReoptimizationComplete={(result) => {
                    if (result.success) {
                      addNotification('再最適化が完了し、スケジュールが更新されました');
                      fetchScheduleStats();
                      if (onUpdate) onUpdate();
                    }
                  }}
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Event Dialog */}
      <Dialog open={showEventDialog} onClose={() => setShowEventDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>イベント追加</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>イベントタイプ</InputLabel>
                <Select
                  value={eventType}
                  onChange={(e) => setEventType(e.target.value)}
                >
                  <MenuItem value="job_delay">作業遅延</MenuItem>
                  <MenuItem value="machine_breakdown">機械故障</MenuItem>
                  <MenuItem value="urgent_job_add">緊急ジョブ追加</MenuItem>
                  <MenuItem value="job_completion">作業完了</MenuItem>
                  <MenuItem value="priority_change">優先度変更</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="対象ID"
                value={targetId}
                onChange={(e) => setTargetId(e.target.value)}
                required
              />
            </Grid>

            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>影響レベル</InputLabel>
                <Select
                  value={impactLevel}
                  onChange={(e) => setImpactLevel(e.target.value)}
                >
                  <MenuItem value="low">低</MenuItem>
                  <MenuItem value="medium">中</MenuItem>
                  <MenuItem value="high">高</MenuItem>
                  <MenuItem value="critical">緊急</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="説明"
                multiline
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </Grid>
          </Grid>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setShowEventDialog(false)}>
            キャンセル
          </Button>
          <Button 
            onClick={addEvent} 
            variant="contained"
            disabled={!targetId || loading}
          >
            追加
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notifications Snackbar */}
      <Snackbar
        open={showNotification}
        autoHideDuration={6000}
        onClose={() => setShowNotification(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setShowNotification(false)} 
          severity="info" 
          variant="filled"
        >
          {notifications[notifications.length - 1]}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default RealtimeScheduleManager;