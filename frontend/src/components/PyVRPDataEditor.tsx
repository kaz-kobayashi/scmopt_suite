import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper,
  Divider,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  RestoreFromTrash as ResetIcon,
  Save as SaveIcon,
} from '@mui/icons-material';

interface Client {
  x: number;
  y: number;
  delivery: number;
  pickup?: number;
  service_duration?: number;
  tw_early?: number;
  tw_late?: number;
  required?: boolean;
  prize?: number;
}

interface Depot {
  x: number;
  y: number;
  tw_early?: number;
  tw_late?: number;
}

interface VehicleType {
  num_available: number;
  capacity: number;
  start_depot: number;
  end_depot?: number;
  fixed_cost?: number;
  tw_early?: number;
  tw_late?: number;
  max_duration?: number;
  max_distance?: number;
}

interface ProblemData {
  clients: Client[];
  depots: Depot[];
  vehicle_types: VehicleType[];
}

interface PyVRPDataEditorProps {
  problemData: ProblemData;
  onDataChange: (newData: ProblemData) => void;
  onSave?: () => void;
}

const PyVRPDataEditor: React.FC<PyVRPDataEditorProps> = ({
  problemData,
  onDataChange,
  onSave
}) => {
  const [expandedPanel, setExpandedPanel] = useState<string | false>('clients');

  // Format time for input (minutes to HH:MM)
  const formatTimeForInput = (minutes: number | undefined): string => {
    if (minutes === undefined) return '';
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  // Parse time input (HH:MM to minutes)
  const parseTimeInput = (timeStr: string): number | undefined => {
    if (!timeStr) return undefined;
    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 60 + minutes;
  };

  // Update client data
  const updateClient = (index: number, field: keyof Client, value: any) => {
    const newClients = [...problemData.clients];
    newClients[index] = { ...newClients[index], [field]: value };
    onDataChange({ ...problemData, clients: newClients });
  };

  // Add new client
  const addClient = () => {
    const newClient: Client = {
      x: 13900 + Math.random() * 400 - 200, // Tokyo area coordinates
      y: 14200 + Math.random() * 400 - 200,
      delivery: Math.floor(Math.random() * 20) + 5,
      service_duration: 10
    };
    onDataChange({
      ...problemData,
      clients: [...problemData.clients, newClient]
    });
  };

  // Remove client
  const removeClient = (index: number) => {
    const newClients = problemData.clients.filter((_, i) => i !== index);
    onDataChange({ ...problemData, clients: newClients });
  };

  // Update depot data
  const updateDepot = (index: number, field: keyof Depot, value: any) => {
    const newDepots = [...problemData.depots];
    newDepots[index] = { ...newDepots[index], [field]: value };
    onDataChange({ ...problemData, depots: newDepots });
  };

  // Add new depot
  const addDepot = () => {
    const newDepot: Depot = {
      x: 13900 + Math.random() * 200 - 100,
      y: 14200 + Math.random() * 200 - 100
    };
    onDataChange({
      ...problemData,
      depots: [...problemData.depots, newDepot]
    });
  };

  // Remove depot
  const removeDepot = (index: number) => {
    const newDepots = problemData.depots.filter((_, i) => i !== index);
    onDataChange({ ...problemData, depots: newDepots });
  };

  // Update vehicle type data
  const updateVehicleType = (index: number, field: keyof VehicleType, value: any) => {
    const newVehicleTypes = [...problemData.vehicle_types];
    newVehicleTypes[index] = { ...newVehicleTypes[index], [field]: value };
    onDataChange({ ...problemData, vehicle_types: newVehicleTypes });
  };

  // Add new vehicle type
  const addVehicleType = () => {
    const newVehicleType: VehicleType = {
      num_available: 1,
      capacity: 100,
      start_depot: 0,
      end_depot: 0,
      tw_early: 480,  // 8:00 AM
      tw_late: 1080   // 6:00 PM
    };
    onDataChange({
      ...problemData,
      vehicle_types: [...problemData.vehicle_types, newVehicleType]
    });
  };

  // Remove vehicle type
  const removeVehicleType = (index: number) => {
    const newVehicleTypes = problemData.vehicle_types.filter((_, i) => i !== index);
    onDataChange({ ...problemData, vehicle_types: newVehicleTypes });
  };

  const handlePanelChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedPanel(isExpanded ? panel : false);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Problem Data Editor
        </Typography>
        {onSave && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<SaveIcon />}
            onClick={onSave}
          >
            Apply Changes
          </Button>
        )}
      </Box>

      {/* Clients Editor */}
      <Accordion expanded={expandedPanel === 'clients'} onChange={handlePanelChange('clients')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">
            Clients ({problemData.clients.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 2 }}>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={addClient}
              size="small"
            >
              Add Client
            </Button>
          </Box>
          
          {problemData.clients.map((client, index) => (
            <Paper key={index} sx={{ p: 2, mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2">Client {index}</Typography>
                <IconButton
                  color="error"
                  size="small"
                  onClick={() => removeClient(index)}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="X Coordinate"
                    type="number"
                    value={client.x}
                    onChange={(e) => updateClient(index, 'x', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Y Coordinate"
                    type="number"
                    value={client.y}
                    onChange={(e) => updateClient(index, 'y', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Delivery Demand"
                    type="number"
                    value={client.delivery}
                    onChange={(e) => updateClient(index, 'delivery', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Pickup Demand"
                    type="number"
                    value={client.pickup || ''}
                    onChange={(e) => updateClient(index, 'pickup', e.target.value ? Number(e.target.value) : undefined)}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Service Duration (min)"
                    type="number"
                    value={client.service_duration || ''}
                    onChange={(e) => updateClient(index, 'service_duration', e.target.value ? Number(e.target.value) : undefined)}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Time Window Start"
                    type="time"
                    value={formatTimeForInput(client.tw_early)}
                    onChange={(e) => updateClient(index, 'tw_early', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Time Window End"
                    type="time"
                    value={formatTimeForInput(client.tw_late)}
                    onChange={(e) => updateClient(index, 'tw_late', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Prize"
                    type="number"
                    value={client.prize || ''}
                    onChange={(e) => updateClient(index, 'prize', e.target.value ? Number(e.target.value) : undefined)}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={client.required !== false}
                        onChange={(e) => updateClient(index, 'required', e.target.checked)}
                      />
                    }
                    label="Required Client"
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
        </AccordionDetails>
      </Accordion>

      {/* Depots Editor */}
      <Accordion expanded={expandedPanel === 'depots'} onChange={handlePanelChange('depots')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">
            Depots ({problemData.depots.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 2 }}>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={addDepot}
              size="small"
            >
              Add Depot
            </Button>
          </Box>
          
          {problemData.depots.map((depot, index) => (
            <Paper key={index} sx={{ p: 2, mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2">Depot {index}</Typography>
                <IconButton
                  color="error"
                  size="small"
                  onClick={() => removeDepot(index)}
                  disabled={problemData.depots.length <= 1}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="X Coordinate"
                    type="number"
                    value={depot.x}
                    onChange={(e) => updateDepot(index, 'x', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Y Coordinate"
                    type="number"
                    value={depot.y}
                    onChange={(e) => updateDepot(index, 'y', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Operating Hours Start"
                    type="time"
                    value={formatTimeForInput(depot.tw_early)}
                    onChange={(e) => updateDepot(index, 'tw_early', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Operating Hours End"
                    type="time"
                    value={formatTimeForInput(depot.tw_late)}
                    onChange={(e) => updateDepot(index, 'tw_late', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
        </AccordionDetails>
      </Accordion>

      {/* Vehicle Types Editor */}
      <Accordion expanded={expandedPanel === 'vehicles'} onChange={handlePanelChange('vehicles')}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">
            Vehicle Types ({problemData.vehicle_types.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ mb: 2 }}>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={addVehicleType}
              size="small"
            >
              Add Vehicle Type
            </Button>
          </Box>
          
          {problemData.vehicle_types.map((vehicleType, index) => (
            <Paper key={index} sx={{ p: 2, mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="subtitle2">Vehicle Type {index}</Typography>
                <IconButton
                  color="error"
                  size="small"
                  onClick={() => removeVehicleType(index)}
                  disabled={problemData.vehicle_types.length <= 1}
                >
                  <DeleteIcon />
                </IconButton>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Number Available"
                    type="number"
                    value={vehicleType.num_available}
                    onChange={(e) => updateVehicleType(index, 'num_available', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Capacity"
                    type="number"
                    value={vehicleType.capacity}
                    onChange={(e) => updateVehicleType(index, 'capacity', Number(e.target.value))}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Start Depot</InputLabel>
                    <Select
                      value={vehicleType.start_depot}
                      label="Start Depot"
                      onChange={(e) => updateVehicleType(index, 'start_depot', Number(e.target.value))}
                    >
                      {problemData.depots.map((_, depotIndex) => (
                        <MenuItem key={depotIndex} value={depotIndex}>
                          Depot {depotIndex}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6} md={3}>
                  <FormControl fullWidth size="small">
                    <InputLabel>End Depot</InputLabel>
                    <Select
                      value={vehicleType.end_depot ?? vehicleType.start_depot}
                      label="End Depot"
                      onChange={(e) => updateVehicleType(index, 'end_depot', Number(e.target.value))}
                    >
                      {problemData.depots.map((_, depotIndex) => (
                        <MenuItem key={depotIndex} value={depotIndex}>
                          Depot {depotIndex}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Fixed Cost"
                    type="number"
                    value={vehicleType.fixed_cost || ''}
                    onChange={(e) => updateVehicleType(index, 'fixed_cost', e.target.value ? Number(e.target.value) : undefined)}
                    fullWidth
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Shift Start"
                    type="time"
                    value={formatTimeForInput(vehicleType.tw_early)}
                    onChange={(e) => updateVehicleType(index, 'tw_early', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Shift End"
                    type="time"
                    value={formatTimeForInput(vehicleType.tw_late)}
                    onChange={(e) => updateVehicleType(index, 'tw_late', parseTimeInput(e.target.value))}
                    fullWidth
                    size="small"
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
                <Grid item xs={6} md={3}>
                  <TextField
                    label="Max Duration (min)"
                    type="number"
                    value={vehicleType.max_duration || ''}
                    onChange={(e) => updateVehicleType(index, 'max_duration', e.target.value ? Number(e.target.value) : undefined)}
                    fullWidth
                    size="small"
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default PyVRPDataEditor;