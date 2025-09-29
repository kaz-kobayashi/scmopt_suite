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
  Tooltip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  LocalShipping as TruckIcon,
  Speed as SpeedIcon,
  Scale as WeightIcon,
  Inventory as VolumeIcon,
} from '@mui/icons-material';

interface CapacityConstraint {
  id: string;
  name: string;
  unit: string;
  value: number;
}

interface VehicleSkill {
  id: string;
  name: string;
  description: string;
}

interface AdvancedVehicleType {
  id: string;
  name: string;
  num_available: number;
  start_depot: number;
  end_depot?: number;
  
  // Multiple capacity constraints
  capacities: CapacityConstraint[];
  
  // Distance and time limits
  max_distance?: number;      // meters
  max_duration?: number;      // minutes
  max_working_time?: number;  // minutes (excluding breaks)
  
  // Cost structure
  fixed_cost: number;
  distance_cost_per_km: number;
  time_cost_per_hour: number;
  overtime_cost_multiplier: number;
  
  // Skills and requirements
  skills: string[];           // skill IDs this vehicle possesses
  required_skills: string[];  // skills required for this vehicle type
  
  // Time windows and breaks
  tw_early: number;
  tw_late: number;
  break_duration?: number;    // minutes
  break_earliest?: number;    // earliest break start time
  break_latest?: number;      // latest break start time
  
  // Vehicle characteristics
  speed_factor: number;       // 1.0 = normal speed, 0.8 = 20% slower
  priority: number;           // 1-10, higher = preferred
  fuel_type: 'gasoline' | 'diesel' | 'electric' | 'hybrid';
  emissions_factor: number;   // kg CO2 per km
}

interface AdvancedVehicleConfigPanelProps {
  vehicleTypes: AdvancedVehicleType[];
  onVehicleTypesChange: (vehicleTypes: AdvancedVehicleType[]) => void;
  availableSkills: VehicleSkill[];
  onSkillsChange: (skills: VehicleSkill[]) => void;
}

const AdvancedVehicleConfigPanel: React.FC<AdvancedVehicleConfigPanelProps> = ({
  vehicleTypes,
  onVehicleTypesChange,
  availableSkills,
  onSkillsChange
}) => {
  const [editingVehicle, setEditingVehicle] = useState<AdvancedVehicleType | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [skillDialogOpen, setSkillDialogOpen] = useState(false);
  const [newSkill, setNewSkill] = useState<Partial<VehicleSkill>>({});

  const createNewVehicleType = (): AdvancedVehicleType => ({
    id: `vehicle_${Date.now()}`,
    name: `車両タイプ ${vehicleTypes.length + 1}`,
    num_available: 1,
    start_depot: 0,
    capacities: [
      { id: 'weight', name: '重量', unit: 'kg', value: 1000 },
      { id: 'volume', name: '体積', unit: 'm³', value: 10 }
    ],
    max_distance: 200000,
    max_duration: 480,
    max_working_time: 420,
    fixed_cost: 5000,
    distance_cost_per_km: 50,
    time_cost_per_hour: 2000,
    overtime_cost_multiplier: 1.5,
    skills: [],
    required_skills: [],
    tw_early: 480,
    tw_late: 1080,
    speed_factor: 1.0,
    priority: 5,
    fuel_type: 'gasoline',
    emissions_factor: 0.2
  });

  const handleAddVehicleType = () => {
    setEditingVehicle(createNewVehicleType());
    setDialogOpen(true);
  };

  const handleEditVehicleType = (vehicle: AdvancedVehicleType) => {
    setEditingVehicle({ ...vehicle });
    setDialogOpen(true);
  };

  const handleSaveVehicleType = () => {
    if (!editingVehicle) return;

    const existingIndex = vehicleTypes.findIndex(v => v.id === editingVehicle.id);
    let newVehicleTypes;

    if (existingIndex >= 0) {
      newVehicleTypes = [...vehicleTypes];
      newVehicleTypes[existingIndex] = editingVehicle;
    } else {
      newVehicleTypes = [...vehicleTypes, editingVehicle];
    }

    onVehicleTypesChange(newVehicleTypes);
    setDialogOpen(false);
    setEditingVehicle(null);
  };

  const handleDeleteVehicleType = (id: string) => {
    onVehicleTypesChange(vehicleTypes.filter(v => v.id !== id));
  };

  const handleAddCapacity = () => {
    if (!editingVehicle) return;
    
    const newCapacity: CapacityConstraint = {
      id: `capacity_${Date.now()}`,
      name: '新制約',
      unit: '個',
      value: 100
    };

    setEditingVehicle({
      ...editingVehicle,
      capacities: [...editingVehicle.capacities, newCapacity]
    });
  };

  const handleUpdateCapacity = (index: number, field: keyof CapacityConstraint, value: any) => {
    if (!editingVehicle) return;

    const newCapacities = [...editingVehicle.capacities];
    newCapacities[index] = { ...newCapacities[index], [field]: value };

    setEditingVehicle({
      ...editingVehicle,
      capacities: newCapacities
    });
  };

  const handleDeleteCapacity = (index: number) => {
    if (!editingVehicle) return;

    setEditingVehicle({
      ...editingVehicle,
      capacities: editingVehicle.capacities.filter((_, i) => i !== index)
    });
  };

  const handleAddSkill = () => {
    if (!newSkill.name || !newSkill.description) return;

    const skill: VehicleSkill = {
      id: `skill_${Date.now()}`,
      name: newSkill.name,
      description: newSkill.description
    };

    onSkillsChange([...availableSkills, skill]);
    setNewSkill({});
    setSkillDialogOpen(false);
  };

  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TruckIcon color="primary" />
          <Typography variant="h6">高度な車両設定</Typography>
        </Box>
        <Box>
          <Button
            variant="outlined"
            onClick={() => setSkillDialogOpen(true)}
            startIcon={<AddIcon />}
            sx={{ mr: 1 }}
          >
            スキル管理
          </Button>
          <Button
            variant="contained"
            onClick={handleAddVehicleType}
            startIcon={<AddIcon />}
          >
            車両タイプ追加
          </Button>
        </Box>
      </Box>

      {/* Vehicle Types Table */}
      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>車両名</TableCell>
              <TableCell>台数</TableCell>
              <TableCell>容量制約</TableCell>
              <TableCell>制限</TableCell>
              <TableCell>コスト構造</TableCell>
              <TableCell>スキル</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {vehicleTypes.map((vehicle) => (
              <TableRow key={vehicle.id}>
                <TableCell>
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {vehicle.name}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {vehicle.fuel_type} | 優先度: {vehicle.priority}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>{vehicle.num_available}</TableCell>
                <TableCell>
                  {vehicle.capacities.map((cap, index) => (
                    <Chip
                      key={index}
                      label={`${cap.name}: ${cap.value}${cap.unit}`}
                      size="small"
                      sx={{ mr: 0.5, mb: 0.5 }}
                    />
                  ))}
                </TableCell>
                <TableCell>
                  <Typography variant="caption" display="block">
                    <SpeedIcon sx={{ fontSize: 12, mr: 0.5 }} />
                    最大: {Math.round((vehicle.max_distance || 0) / 1000)}km
                  </Typography>
                  <Typography variant="caption" display="block">
                    ⏱️ 最大: {formatTime(vehicle.max_duration || 0)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" display="block">
                    固定: ¥{vehicle.fixed_cost.toLocaleString()}
                  </Typography>
                  <Typography variant="caption" display="block">
                    距離: ¥{vehicle.distance_cost_per_km}/km
                  </Typography>
                </TableCell>
                <TableCell>
                  {vehicle.skills.length > 0 ? (
                    vehicle.skills.map(skillId => {
                      const skill = availableSkills.find(s => s.id === skillId);
                      return skill ? (
                        <Chip key={skillId} label={skill.name} size="small" />
                      ) : null;
                    })
                  ) : (
                    <Typography variant="caption" color="textSecondary">なし</Typography>
                  )}
                </TableCell>
                <TableCell>
                  <IconButton
                    onClick={() => handleEditVehicleType(vehicle)}
                    size="small"
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    onClick={() => handleDeleteVehicleType(vehicle.id)}
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

      {/* Vehicle Type Edit Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingVehicle?.id.startsWith('vehicle_') ? '車両タイプ追加' : '車両タイプ編集'}
        </DialogTitle>
        <DialogContent>
          {editingVehicle && (
            <Box sx={{ mt: 1 }}>
              {/* Basic Info */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>基本情報</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="車両名"
                        value={editingVehicle.name}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          name: e.target.value
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="使用可能台数"
                        type="number"
                        value={editingVehicle.num_available}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          num_available: parseInt(e.target.value)
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth>
                        <InputLabel>燃料タイプ</InputLabel>
                        <Select
                          value={editingVehicle.fuel_type}
                          onChange={(e) => setEditingVehicle({
                            ...editingVehicle,
                            fuel_type: e.target.value as any
                          })}
                        >
                          <MenuItem value="gasoline">ガソリン</MenuItem>
                          <MenuItem value="diesel">ディーゼル</MenuItem>
                          <MenuItem value="electric">電気</MenuItem>
                          <MenuItem value="hybrid">ハイブリッド</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="優先度 (1-10)"
                        type="number"
                        value={editingVehicle.priority}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          priority: parseInt(e.target.value)
                        })}
                        fullWidth
                        InputProps={{ inputProps: { min: 1, max: 10 } }}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Capacity Constraints */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>容量制約</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mb: 2 }}>
                    <Button
                      onClick={handleAddCapacity}
                      startIcon={<AddIcon />}
                      size="small"
                    >
                      制約追加
                    </Button>
                  </Box>
                  {editingVehicle.capacities.map((capacity, index) => (
                    <Grid container spacing={2} key={index} sx={{ mb: 2 }}>
                      <Grid item xs={3}>
                        <TextField
                          label="制約名"
                          value={capacity.name}
                          onChange={(e) => handleUpdateCapacity(index, 'name', e.target.value)}
                          size="small"
                          fullWidth
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <TextField
                          label="値"
                          type="number"
                          value={capacity.value}
                          onChange={(e) => handleUpdateCapacity(index, 'value', parseFloat(e.target.value))}
                          size="small"
                          fullWidth
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <TextField
                          label="単位"
                          value={capacity.unit}
                          onChange={(e) => handleUpdateCapacity(index, 'unit', e.target.value)}
                          size="small"
                          fullWidth
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <IconButton
                          onClick={() => handleDeleteCapacity(index)}
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Grid>
                    </Grid>
                  ))}
                </AccordionDetails>
              </Accordion>

              {/* Skills Section */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>スキル設定</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" gutterBottom>
                        車両が持つスキル
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                        {availableSkills.map((skill) => (
                          <Chip
                            key={skill.id}
                            label={skill.name}
                            onClick={() => {
                              const currentSkills = editingVehicle.skills || [];
                              if (currentSkills.includes(skill.id)) {
                                setEditingVehicle({
                                  ...editingVehicle,
                                  skills: currentSkills.filter(s => s !== skill.id)
                                });
                              } else {
                                setEditingVehicle({
                                  ...editingVehicle,
                                  skills: [...currentSkills, skill.id]
                                });
                              }
                            }}
                            color={editingVehicle.skills?.includes(skill.id) ? 'primary' : 'default'}
                            variant={editingVehicle.skills?.includes(skill.id) ? 'filled' : 'outlined'}
                          />
                        ))}
                      </Box>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2" gutterBottom>
                        必要スキル（この車両タイプを使用するのに必要なスキル）
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {availableSkills.map((skill) => (
                          <Chip
                            key={skill.id}
                            label={skill.name}
                            onClick={() => {
                              const currentRequiredSkills = editingVehicle.required_skills || [];
                              if (currentRequiredSkills.includes(skill.id)) {
                                setEditingVehicle({
                                  ...editingVehicle,
                                  required_skills: currentRequiredSkills.filter(s => s !== skill.id)
                                });
                              } else {
                                setEditingVehicle({
                                  ...editingVehicle,
                                  required_skills: [...currentRequiredSkills, skill.id]
                                });
                              }
                            }}
                            color={editingVehicle.required_skills?.includes(skill.id) ? 'secondary' : 'default'}
                            variant={editingVehicle.required_skills?.includes(skill.id) ? 'filled' : 'outlined'}
                          />
                        ))}
                      </Box>
                      <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                        クリックしてスキルを選択/解除
                      </Typography>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Limits and Costs */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>制限・コスト</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="最大走行距離 (km)"
                        type="number"
                        value={Math.round((editingVehicle.max_distance || 0) / 1000)}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          max_distance: parseInt(e.target.value) * 1000
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="最大稼働時間 (時間)"
                        type="number"
                        value={Math.round((editingVehicle.max_duration || 0) / 60)}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          max_duration: parseInt(e.target.value) * 60
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="固定コスト (円)"
                        type="number"
                        value={editingVehicle.fixed_cost}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          fixed_cost: parseInt(e.target.value)
                        })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="距離コスト (円/km)"
                        type="number"
                        value={editingVehicle.distance_cost_per_km}
                        onChange={(e) => setEditingVehicle({
                          ...editingVehicle,
                          distance_cost_per_km: parseInt(e.target.value)
                        })}
                        fullWidth
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
          <Button onClick={handleSaveVehicleType} variant="contained">
            保存
          </Button>
        </DialogActions>
      </Dialog>

      {/* Skill Management Dialog */}
      <Dialog
        open={skillDialogOpen}
        onClose={() => setSkillDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>スキル管理</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 1 }}>
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <TextField
                  label="スキル名"
                  value={newSkill.name || ''}
                  onChange={(e) => setNewSkill({ ...newSkill, name: e.target.value })}
                  fullWidth
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="説明"
                  value={newSkill.description || ''}
                  onChange={(e) => setNewSkill({ ...newSkill, description: e.target.value })}
                  fullWidth
                />
              </Grid>
            </Grid>
            <Button
              onClick={handleAddSkill}
              variant="contained"
              startIcon={<AddIcon />}
              disabled={!newSkill.name || !newSkill.description}
            >
              スキル追加
            </Button>

            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                既存スキル
              </Typography>
              {availableSkills.map((skill) => (
                <Chip
                  key={skill.id}
                  label={skill.name}
                  sx={{ mr: 1, mb: 1 }}
                  onDelete={() => onSkillsChange(availableSkills.filter(s => s.id !== skill.id))}
                />
              ))}
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSkillDialogOpen(false)}>閉じる</Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default AdvancedVehicleConfigPanel;