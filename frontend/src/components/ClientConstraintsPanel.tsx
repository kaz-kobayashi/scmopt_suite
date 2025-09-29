import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  IconButton,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
  Slider,
  Tooltip,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Autocomplete,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Person as ClientIcon,
  AccessTime as TimeIcon,
  Star as PriorityIcon,
  Warning as WarningIcon,
  LocalShipping as VehicleIcon,
} from '@mui/icons-material';

// Time window interface
interface TimeWindow {
  id: string;
  start: number;  // minutes from midnight
  end: number;    // minutes from midnight
  preference: 'preferred' | 'acceptable' | 'restricted';
}

// Service time details
interface ServiceTimeDetails {
  base_time: number;          // minutes
  time_per_unit: number;      // minutes per unit of delivery/pickup
  setup_time: number;         // one-time setup minutes
  cleanup_time: number;       // cleanup minutes after service
  time_variability: number;   // percentage variability (0-50%)
}

// Advanced client interface
interface AdvancedClient {
  id: string;
  name: string;
  x: number;
  y: number;
  
  // Basic demands
  delivery: number;
  pickup?: number;
  
  // Multiple time windows
  time_windows: TimeWindow[];
  
  // Priority and classification
  priority: 'high' | 'medium' | 'low';
  client_type: 'regular' | 'vip' | 'new' | 'problematic';
  
  // Service requirements
  service_details: ServiceTimeDetails;
  
  // Vehicle constraints
  required_vehicle_skills: string[];      // skills that vehicle must have
  prohibited_vehicle_types: string[];     // vehicle types that cannot serve
  preferred_vehicle_types: string[];      // preferred vehicle types (soft constraint)
  
  // Compatibility constraints
  incompatible_clients: string[];         // client IDs that cannot be in same route
  required_sequence_before: string[];     // must visit these clients before
  required_sequence_after: string[];      // must visit these clients after
  
  // Additional constraints
  max_wait_time?: number;                 // maximum wait time at client (minutes)
  requires_appointment: boolean;
  appointment_flexibility?: number;       // minutes of flexibility around appointment
  
  // Notes and metadata
  notes?: string;
  contact_info?: string;
  special_instructions?: string;
}

interface ClientConstraintsPanelProps {
  clients: AdvancedClient[];
  onClientsChange: (clients: AdvancedClient[]) => void;
  availableVehicleTypes?: string[];
  availableSkills?: { id: string; name: string }[];
}

const ClientConstraintsPanel: React.FC<ClientConstraintsPanelProps> = ({
  clients,
  onClientsChange,
  availableVehicleTypes = [],
  availableSkills = []
}) => {
  const [editingClient, setEditingClient] = useState<AdvancedClient | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);

  // Priority colors
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  // Time formatting
  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  // Create new client
  const createNewClient = (): AdvancedClient => ({
    id: `client_${Date.now()}`,
    name: `顧客 ${clients.length + 1}`,
    x: Math.round(1394000 + Math.random() * 4000),  // Ensure integer coordinates
    y: Math.round(356000 + Math.random() * 3000),   // Ensure integer coordinates
    delivery: 5,
    time_windows: [{
      id: 'tw_1',
      start: 480,
      end: 1080,
      preference: 'preferred'
    }],
    priority: 'medium',
    client_type: 'regular',
    service_details: {
      base_time: 10,
      time_per_unit: 1,
      setup_time: 5,
      cleanup_time: 2,
      time_variability: 10
    },
    required_vehicle_skills: [],
    prohibited_vehicle_types: [],
    preferred_vehicle_types: [],
    incompatible_clients: [],
    required_sequence_before: [],
    required_sequence_after: [],
    requires_appointment: false
  });

  const handleAddClient = () => {
    setEditingClient(createNewClient());
    setDialogOpen(true);
  };

  const handleEditClient = (client: AdvancedClient) => {
    setEditingClient({ ...client });
    setDialogOpen(true);
  };

  const handleSaveClient = () => {
    if (!editingClient) return;

    const existingIndex = clients.findIndex(c => c.id === editingClient.id);
    let newClients;

    if (existingIndex >= 0) {
      newClients = [...clients];
      newClients[existingIndex] = editingClient;
    } else {
      newClients = [...clients, editingClient];
    }

    onClientsChange(newClients);
    setDialogOpen(false);
    setEditingClient(null);
  };

  const handleDeleteClient = (id: string) => {
    onClientsChange(clients.filter(c => c.id !== id));
  };

  // Time window management
  const handleAddTimeWindow = () => {
    if (!editingClient) return;
    
    const newTimeWindow: TimeWindow = {
      id: `tw_${Date.now()}`,
      start: 540,
      end: 720,
      preference: 'acceptable'
    };

    setEditingClient({
      ...editingClient,
      time_windows: [...editingClient.time_windows, newTimeWindow]
    });
  };

  const handleUpdateTimeWindow = (index: number, field: keyof TimeWindow, value: any) => {
    if (!editingClient) return;

    const newTimeWindows = [...editingClient.time_windows];
    newTimeWindows[index] = { ...newTimeWindows[index], [field]: value };

    setEditingClient({
      ...editingClient,
      time_windows: newTimeWindows
    });
  };

  const handleDeleteTimeWindow = (index: number) => {
    if (!editingClient) return;

    setEditingClient({
      ...editingClient,
      time_windows: editingClient.time_windows.filter((_, i) => i !== index)
    });
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ClientIcon color="primary" />
          <Typography variant="h6">顧客制約拡張</Typography>
        </Box>
        <Button
          variant="contained"
          onClick={handleAddClient}
          startIcon={<AddIcon />}
        >
          顧客追加
        </Button>
      </Box>

      {/* Clients Summary Table */}
      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>顧客名</TableCell>
              <TableCell>優先度</TableCell>
              <TableCell>時間窓</TableCell>
              <TableCell>配送量</TableCell>
              <TableCell>必要スキル</TableCell>
              <TableCell>制約</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {clients.map((client) => (
              <TableRow key={client.id}>
                <TableCell>
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {client.name}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {client.client_type}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip
                    icon={<PriorityIcon />}
                    label={client.priority}
                    color={getPriorityColor(client.priority) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box>
                    {client.time_windows.map((tw, index) => (
                      <Typography key={index} variant="caption" display="block">
                        {formatTime(tw.start)} - {formatTime(tw.end)}
                        {tw.preference !== 'preferred' && ` (${tw.preference})`}
                      </Typography>
                    ))}
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    配送: {client.delivery}
                    {client.pickup && `, 集荷: ${client.pickup}`}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    サービス時間: {client.service_details.base_time}分
                  </Typography>
                </TableCell>
                <TableCell>
                  {client.required_vehicle_skills.length > 0 ? (
                    client.required_vehicle_skills.map(skillId => (
                      <Chip
                        key={skillId}
                        label={availableSkills.find(s => s.id === skillId)?.name || skillId}
                        size="small"
                        sx={{ mr: 0.5 }}
                      />
                    ))
                  ) : (
                    <Typography variant="caption" color="textSecondary">なし</Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    {client.requires_appointment && (
                      <Tooltip title="要予約">
                        <TimeIcon fontSize="small" color="warning" />
                      </Tooltip>
                    )}
                    {client.incompatible_clients.length > 0 && (
                      <Tooltip title="非互換顧客あり">
                        <WarningIcon fontSize="small" color="error" />
                      </Tooltip>
                    )}
                    {client.prohibited_vehicle_types.length > 0 && (
                      <Tooltip title="車両制限あり">
                        <VehicleIcon fontSize="small" color="info" />
                      </Tooltip>
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  <IconButton
                    onClick={() => handleEditClient(client)}
                    size="small"
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeleteClient(client.id)}
                    size="small"
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Client Edit Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingClient?.id.startsWith('client_') ? '顧客追加' : '顧客編集'}
        </DialogTitle>
        <DialogContent>
          {editingClient && (
            <Box sx={{ mt: 1 }}>
              {/* Basic Information */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>基本情報</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="顧客名"
                        value={editingClient.name}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          name: e.target.value
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <FormControl fullWidth>
                        <InputLabel>優先度</InputLabel>
                        <Select
                          value={editingClient.priority}
                          onChange={(e) => setEditingClient({
                            ...editingClient,
                            priority: e.target.value as any
                          })}
                        >
                          <MenuItem value="high">高</MenuItem>
                          <MenuItem value="medium">中</MenuItem>
                          <MenuItem value="low">低</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <FormControl fullWidth>
                        <InputLabel>顧客タイプ</InputLabel>
                        <Select
                          value={editingClient.client_type}
                          onChange={(e) => setEditingClient({
                            ...editingClient,
                            client_type: e.target.value as any
                          })}
                        >
                          <MenuItem value="regular">通常</MenuItem>
                          <MenuItem value="vip">VIP</MenuItem>
                          <MenuItem value="new">新規</MenuItem>
                          <MenuItem value="problematic">要注意</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="配送量"
                        type="number"
                        value={editingClient.delivery}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          delivery: parseInt(e.target.value)
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="集荷量"
                        type="number"
                        value={editingClient.pickup || 0}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          pickup: parseInt(e.target.value) || undefined
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={editingClient.requires_appointment}
                            onChange={(e) => setEditingClient({
                              ...editingClient,
                              requires_appointment: e.target.checked
                            })}
                          />
                        }
                        label="要予約"
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Time Windows */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>時間窓設定</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mb: 2 }}>
                    <Button
                      onClick={handleAddTimeWindow}
                      startIcon={<AddIcon />}
                      size="small"
                    >
                      時間窓追加
                    </Button>
                  </Box>
                  {editingClient.time_windows.map((tw, index) => (
                    <Grid container spacing={2} key={tw.id} sx={{ mb: 2 }}>
                      <Grid item xs={3}>
                        <TextField
                          label="開始時刻"
                          type="time"
                          value={`${Math.floor(tw.start / 60).toString().padStart(2, '0')}:${(tw.start % 60).toString().padStart(2, '0')}`}
                          onChange={(e) => {
                            const [hours, minutes] = e.target.value.split(':').map(Number);
                            handleUpdateTimeWindow(index, 'start', hours * 60 + minutes);
                          }}
                          fullWidth
                          InputLabelProps={{ shrink: true }}
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <TextField
                          label="終了時刻"
                          type="time"
                          value={`${Math.floor(tw.end / 60).toString().padStart(2, '0')}:${(tw.end % 60).toString().padStart(2, '0')}`}
                          onChange={(e) => {
                            const [hours, minutes] = e.target.value.split(':').map(Number);
                            handleUpdateTimeWindow(index, 'end', hours * 60 + minutes);
                          }}
                          fullWidth
                          InputLabelProps={{ shrink: true }}
                        />
                      </Grid>
                      <Grid item xs={4}>
                        <FormControl fullWidth>
                          <InputLabel>優先度</InputLabel>
                          <Select
                            value={tw.preference}
                            onChange={(e) => handleUpdateTimeWindow(index, 'preference', e.target.value)}
                          >
                            <MenuItem value="preferred">推奨</MenuItem>
                            <MenuItem value="acceptable">許容</MenuItem>
                            <MenuItem value="restricted">制限付き</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={2}>
                        <IconButton
                          onClick={() => handleDeleteTimeWindow(index)}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Grid>
                    </Grid>
                  ))}
                </AccordionDetails>
              </Accordion>

              {/* Service Time Details */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>サービス時間詳細</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="基本サービス時間 (分)"
                        type="number"
                        value={editingClient.service_details.base_time}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          service_details: {
                            ...editingClient.service_details,
                            base_time: parseInt(e.target.value)
                          }
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="単位あたり時間 (分/個)"
                        type="number"
                        value={editingClient.service_details.time_per_unit}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          service_details: {
                            ...editingClient.service_details,
                            time_per_unit: parseFloat(e.target.value)
                          }
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="準備時間 (分)"
                        type="number"
                        value={editingClient.service_details.setup_time}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          service_details: {
                            ...editingClient.service_details,
                            setup_time: parseInt(e.target.value)
                          }
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="片付け時間 (分)"
                        type="number"
                        value={editingClient.service_details.cleanup_time}
                        onChange={(e) => setEditingClient({
                          ...editingClient,
                          service_details: {
                            ...editingClient.service_details,
                            cleanup_time: parseInt(e.target.value)
                          }
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography gutterBottom>
                        時間変動性: {editingClient.service_details.time_variability}%
                      </Typography>
                      <Slider
                        value={editingClient.service_details.time_variability}
                        onChange={(_, value) => setEditingClient({
                          ...editingClient,
                          service_details: {
                            ...editingClient.service_details,
                            time_variability: value as number
                          }
                        })}
                        min={0}
                        max={50}
                        marks={[
                          { value: 0, label: '0%' },
                          { value: 25, label: '25%' },
                          { value: 50, label: '50%' }
                        ]}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Vehicle Requirements */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>車両要件</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" gutterBottom>
                        必要な車両スキル
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                        {availableSkills.map((skill) => (
                          <Chip
                            key={skill.id}
                            label={skill.name}
                            onClick={() => {
                              const currentSkills = editingClient.required_vehicle_skills || [];
                              if (currentSkills.includes(skill.id)) {
                                setEditingClient({
                                  ...editingClient,
                                  required_vehicle_skills: currentSkills.filter(s => s !== skill.id)
                                });
                              } else {
                                setEditingClient({
                                  ...editingClient,
                                  required_vehicle_skills: [...currentSkills, skill.id]
                                });
                              }
                            }}
                            color={editingClient.required_vehicle_skills?.includes(skill.id) ? 'primary' : 'default'}
                            variant={editingClient.required_vehicle_skills?.includes(skill.id) ? 'filled' : 'outlined'}
                          />
                        ))}
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Autocomplete
                        multiple
                        options={availableVehicleTypes}
                        value={editingClient.prohibited_vehicle_types}
                        onChange={(_, newValue) => setEditingClient({
                          ...editingClient,
                          prohibited_vehicle_types: newValue
                        })}
                        renderInput={(params) => (
                          <TextField {...params} label="禁止車両タイプ" />
                        )}
                        renderTags={(value, getTagProps) =>
                          value.map((option, index) => (
                            <Chip
                              {...getTagProps({ index })}
                              label={option}
                              color="error"
                              size="small"
                            />
                          ))
                        }
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Autocomplete
                        multiple
                        options={availableVehicleTypes}
                        value={editingClient.preferred_vehicle_types}
                        onChange={(_, newValue) => setEditingClient({
                          ...editingClient,
                          preferred_vehicle_types: newValue
                        })}
                        renderInput={(params) => (
                          <TextField {...params} label="推奨車両タイプ" />
                        )}
                        renderTags={(value, getTagProps) =>
                          value.map((option, index) => (
                            <Chip
                              {...getTagProps({ index })}
                              label={option}
                              color="success"
                              size="small"
                            />
                          ))
                        }
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Compatibility Constraints */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>互換性制約</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Autocomplete
                        multiple
                        options={clients.filter(c => c.id !== editingClient.id).map(c => ({ id: c.id, label: c.name }))}
                        value={editingClient.incompatible_clients.map(id => {
                          const client = clients.find(c => c.id === id);
                          return client ? { id: client.id, label: client.name } : null;
                        }).filter(Boolean) as any[]}
                        onChange={(_, newValue) => setEditingClient({
                          ...editingClient,
                          incompatible_clients: newValue.map(v => v.id)
                        })}
                        getOptionLabel={(option) => option.label}
                        renderInput={(params) => (
                          <TextField {...params} label="同一ルートNG顧客" />
                        )}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>キャンセル</Button>
          <Button onClick={handleSaveClient} variant="contained">
            保存
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default ClientConstraintsPanel;