import React from 'react';
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
  Slider,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  HelpOutline as HelpIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

interface SolverConfig {
  // Basic settings
  time_limit: number;           // seconds
  population_size: number;      // genetic algorithm population
  random_seed?: number;         // for reproducible results
  
  // Genetic algorithm parameters
  crossover_probability: number;
  mutation_probability: number;
  repair_probability: number;
  
  // Convergence settings
  max_iterations_without_improvement: number;
  target_gap: number;           // percentage
  
  // Penalty management
  penalty_scaling: number;
  penalty_start: number;
  
  // Advanced settings
  enable_local_search: boolean;
  enable_diversity_management: boolean;
  fleet_minimization: boolean;
}

interface SolverConfigPanelProps {
  config: SolverConfig;
  onConfigChange: (config: SolverConfig) => void;
}

const SolverConfigPanel: React.FC<SolverConfigPanelProps> = ({
  config,
  onConfigChange
}) => {
  const handleChange = (field: keyof SolverConfig, value: any) => {
    onConfigChange({
      ...config,
      [field]: value
    });
  };

  const resetToDefaults = () => {
    onConfigChange({
      time_limit: 60,
      population_size: 25,
      random_seed: undefined,
      crossover_probability: 0.95,
      mutation_probability: 0.02,
      repair_probability: 0.5,
      max_iterations_without_improvement: 1000,
      target_gap: 1.0,
      penalty_scaling: 1.0,
      penalty_start: 100,
      enable_local_search: true,
      enable_diversity_management: true,
      fleet_minimization: false
    });
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <SettingsIcon color="primary" />
        <Typography variant="h6">ソルバー設定</Typography>
        <Tooltip title="デフォルト値に戻す">
          <IconButton onClick={resetToDefaults} size="small">
            <HelpIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Basic Settings */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">基本設定</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                label="計算時間制限 (秒)"
                type="number"
                value={config.time_limit}
                onChange={(e) => handleChange('time_limit', parseInt(e.target.value))}
                fullWidth
                InputProps={{
                  inputProps: { min: 1, max: 3600 }
                }}
                helperText="最大計算時間（1-3600秒）"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                label="集団サイズ"
                type="number"
                value={config.population_size}
                onChange={(e) => handleChange('population_size', parseInt(e.target.value))}
                fullWidth
                InputProps={{
                  inputProps: { min: 10, max: 100 }
                }}
                helperText="遺伝的アルゴリズムの集団サイズ（10-100）"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                label="ランダムシード（オプション）"
                type="number"
                value={config.random_seed || ''}
                onChange={(e) => handleChange('random_seed', e.target.value ? parseInt(e.target.value) : undefined)}
                fullWidth
                helperText="結果を再現可能にする（空白で毎回異なる結果）"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.fleet_minimization}
                    onChange={(e) => handleChange('fleet_minimization', e.target.checked)}
                  />
                }
                label="車両数最小化"
              />
              <Typography variant="caption" display="block" color="textSecondary">
                コストよりも車両数の最小化を優先
              </Typography>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Genetic Algorithm Parameters */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">遺伝的アルゴリズム設定</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography gutterBottom>交叉確率: {(config.crossover_probability * 100).toFixed(0)}%</Typography>
              <Slider
                value={config.crossover_probability}
                onChange={(_, value) => handleChange('crossover_probability', value)}
                min={0.5}
                max={1.0}
                step={0.05}
                marks={[
                  { value: 0.5, label: '50%' },
                  { value: 0.75, label: '75%' },
                  { value: 1.0, label: '100%' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>突然変異確率: {(config.mutation_probability * 100).toFixed(1)}%</Typography>
              <Slider
                value={config.mutation_probability}
                onChange={(_, value) => handleChange('mutation_probability', value)}
                min={0.01}
                max={0.1}
                step={0.005}
                marks={[
                  { value: 0.01, label: '1%' },
                  { value: 0.05, label: '5%' },
                  { value: 0.1, label: '10%' }
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>修復確率: {(config.repair_probability * 100).toFixed(0)}%</Typography>
              <Slider
                value={config.repair_probability}
                onChange={(_, value) => handleChange('repair_probability', value)}
                min={0.1}
                max={1.0}
                step={0.1}
                marks={[
                  { value: 0.1, label: '10%' },
                  { value: 0.5, label: '50%' },
                  { value: 1.0, label: '100%' }
                ]}
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Convergence Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">収束条件</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                label="改善なし最大反復回数"
                type="number"
                value={config.max_iterations_without_improvement}
                onChange={(e) => handleChange('max_iterations_without_improvement', parseInt(e.target.value))}
                fullWidth
                InputProps={{
                  inputProps: { min: 100, max: 5000 }
                }}
                helperText="この回数改善がないと終了"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                label="目標ギャップ (%)"
                type="number"
                value={config.target_gap}
                onChange={(e) => handleChange('target_gap', parseFloat(e.target.value))}
                fullWidth
                InputProps={{
                  inputProps: { min: 0.1, max: 10.0, step: 0.1 }
                }}
                helperText="この値以下で収束とみなす"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Advanced Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">高度な設定</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enable_local_search}
                    onChange={(e) => handleChange('enable_local_search', e.target.checked)}
                  />
                }
                label="局所探索有効化"
              />
              <Typography variant="caption" display="block" color="textSecondary">
                解の品質向上のための局所探索
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enable_diversity_management}
                    onChange={(e) => handleChange('enable_diversity_management', e.target.checked)}
                  />
                }
                label="多様性管理"
              />
              <Typography variant="caption" display="block" color="textSecondary">
                集団の多様性を維持して局所最適解を回避
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>ペナルティスケーリング: {config.penalty_scaling.toFixed(1)}</Typography>
              <Slider
                value={config.penalty_scaling}
                onChange={(_, value) => handleChange('penalty_scaling', value)}
                min={0.1}
                max={2.0}
                step={0.1}
                marks={[
                  { value: 0.5, label: '0.5' },
                  { value: 1.0, label: '1.0' },
                  { value: 1.5, label: '1.5' }
                ]}
              />
              <Typography variant="caption" color="textSecondary">
                制約違反に対するペナルティの強度
              </Typography>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
};

export default SolverConfigPanel;