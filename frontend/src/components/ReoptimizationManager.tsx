import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  FormControlLabel,
  Switch,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
  IconButton,
  Collapse
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  Memory as MemoryIcon,
  LocalFireDepartment as PriorityIcon,
  Build as BottleneckIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  History as HistoryIcon,
  Assessment as MetricsIcon
} from '@mui/icons-material';

interface ReoptimizationManagerProps {
  scheduleId: string;
  currentSolution: any;
  onReoptimizationComplete?: (result: any) => void;
}

interface ReoptimizationResult {
  schedule_id: string;
  success: boolean;
  new_solution: any;
  optimization_time: number;
  changes_summary: any;
  impact_metrics: any;
  error_message?: string;
}

interface ReoptimizationHistory {
  timestamp: string;
  strategy: string;
  trigger: string;
  success: boolean;
  optimization_time: number;
  improvement_ratio?: number;
}

const ReoptimizationManager: React.FC<ReoptimizationManagerProps> = ({
  scheduleId,
  currentSolution,
  onReoptimizationComplete
}) => {
  const [loading, setLoading] = useState(false);
  const [showAdvancedDialog, setShowAdvancedDialog] = useState(false);
  const [showHistoryDialog, setShowHistoryDialog] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [reoptimizationHistory, setReoptimizationHistory] = useState<ReoptimizationHistory[]>([]);
  const [expandedSettings, setExpandedSettings] = useState(false);

  // Configuration state
  const [reoptimizationType, setReoptimizationType] = useState('incremental');
  const [strategy, setStrategy] = useState('');
  const [timeLimit, setTimeLimit] = useState(60);
  const [preserveCompleted, setPreserveCompleted] = useState(true);
  const [autoReoptimize, setAutoReoptimize] = useState(true);
  const [eventThreshold, setEventThreshold] = useState(3);

  // Current optimization result
  const [lastResult, setLastResult] = useState<ReoptimizationResult | null>(null);

  const strategies = [
    { value: '', label: '自動選択' },
    { value: 'complete', label: '完全再最適化', icon: <RefreshIcon />, description: 'スケジュール全体を再構築' },
    { value: 'incremental', label: '増分最適化', icon: <SpeedIcon />, description: '影響範囲のみ最適化' },
    { value: 'local_search', label: '局所探索', icon: <TimelineIcon />, description: '局所的な改善を探索' },
    { value: 'time_window', label: '時間窓制約', icon: <MemoryIcon />, description: '未来の時間窓のみ最適化' },
    { value: 'priority_focused', label: '優先度重点', icon: <PriorityIcon />, description: '高優先度ジョブを優先' },
    { value: 'bottleneck_focused', label: 'ボトルネック重点', icon: <BottleneckIcon />, description: 'ボトルネック解消に集中' }
  ];

  const executeReoptimization = async (customStrategy?: string) => {
    if (!scheduleId) {
      setErrorMessage('スケジュールIDが設定されていません');
      return;
    }

    setLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const requestBody = {
        reoptimization_type: reoptimizationType,
        time_limit: timeLimit,
        preserve_completed_jobs: preserveCompleted,
        ...(customStrategy && { strategy: customStrategy })
      };

      const response = await fetch(
        `http://localhost:8000/api/realtime/schedules/${scheduleId}/reoptimize`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody)
        }
      );

      if (!response.ok) {
        throw new Error('再最適化リクエストが失敗しました');
      }

      const result: ReoptimizationResult = await response.json();
      setLastResult(result);

      if (result.success) {
        setSuccessMessage(
          `再最適化が完了しました (${result.optimization_time.toFixed(2)}秒)`
        );
        
        // Add to history
        const historyEntry: ReoptimizationHistory = {
          timestamp: new Date().toISOString(),
          strategy: customStrategy || strategy || reoptimizationType,
          trigger: 'manual',
          success: result.success,
          optimization_time: result.optimization_time,
          improvement_ratio: result.impact_metrics?.improvement_ratio
        };
        
        setReoptimizationHistory(prev => [historyEntry, ...prev.slice(0, 9)]);
        
        if (onReoptimizationComplete) {
          onReoptimizationComplete(result);
        }
      } else {
        setErrorMessage(result.error_message || '再最適化に失敗しました');
      }
    } catch (error) {
      setErrorMessage(`エラー: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const getStrategyIcon = (strategyValue: string) => {
    const strategy = strategies.find(s => s.value === strategyValue);
    return strategy?.icon || <RefreshIcon />;
  };

  const getStrategyColor = (strategyValue: string) => {
    const colorMap: { [key: string]: string } = {
      'complete': '#f44336',
      'incremental': '#4caf50',
      'local_search': '#ff9800',
      'time_window': '#2196f3',
      'priority_focused': '#9c27b0',
      'bottleneck_focused': '#795548'
    };
    return colorMap[strategyValue] || '#757575';
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    if (seconds < 60) return `${seconds.toFixed(2)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        再最適化管理
      </Typography>

      {errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setErrorMessage('')}>
          {errorMessage}
        </Alert>
      )}

      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage('')}>
          {successMessage}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                クイックアクション
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="contained"
                    color="primary"
                    startIcon={<SpeedIcon />}
                    onClick={() => executeReoptimization('incremental')}
                    disabled={loading}
                  >
                    増分最適化
                  </Button>
                </Grid>
                
                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="contained"
                    color="secondary"
                    startIcon={<RefreshIcon />}
                    onClick={() => executeReoptimization('complete')}
                    disabled={loading}
                  >
                    完全再最適化
                  </Button>
                </Grid>

                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<PriorityIcon />}
                    onClick={() => executeReoptimization('priority_focused')}
                    disabled={loading}
                  >
                    優先度重点
                  </Button>
                </Grid>

                <Grid item xs={6}>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<BottleneckIcon />}
                    onClick={() => executeReoptimization('bottleneck_focused')}
                    disabled={loading}
                  >
                    ボトルネック解消
                  </Button>
                </Grid>
              </Grid>

              {loading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    再最適化を実行中...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Current Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                現在の状況
              </Typography>

              {currentSolution?.metrics && (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      メイクスパン
                    </Typography>
                    <Typography variant="h6">
                      {currentSolution.metrics.makespan}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      平均稼働率
                    </Typography>
                    <Typography variant="h6">
                      {(currentSolution.metrics.average_machine_utilization * 100).toFixed(1)}%
                    </Typography>
                  </Grid>

                  {currentSolution.metrics.total_tardiness !== undefined && (
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        総遅延時間
                      </Typography>
                      <Typography variant="h6">
                        {currentSolution.metrics.total_tardiness}
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              )}

              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<SettingsIcon />}
                  onClick={() => setShowAdvancedDialog(true)}
                >
                  詳細設定
                </Button>
                
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<HistoryIcon />}
                  onClick={() => setShowHistoryDialog(true)}
                >
                  履歴
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Last Optimization Result */}
        {lastResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  <MetricsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  最新の最適化結果
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      ステータス
                    </Typography>
                    <Chip
                      label={lastResult.success ? '成功' : '失敗'}
                      color={lastResult.success ? 'success' : 'error'}
                      size="small"
                    />
                  </Grid>

                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      実行時間
                    </Typography>
                    <Typography variant="body1">
                      {formatDuration(lastResult.optimization_time)}
                    </Typography>
                  </Grid>

                  {lastResult.impact_metrics?.improvement_ratio !== undefined && (
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">
                        改善率
                      </Typography>
                      <Typography 
                        variant="body1"
                        color={lastResult.impact_metrics.improvement_ratio > 0 ? 'success.main' : 'text.primary'}
                      >
                        {lastResult.impact_metrics.improvement_ratio > 0 ? '+' : ''}
                        {lastResult.impact_metrics.improvement_ratio.toFixed(2)}%
                      </Typography>
                    </Grid>
                  )}

                  {lastResult.impact_metrics?.new_makespan && (
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">
                        新しいメイクスパン
                      </Typography>
                      <Typography variant="body1">
                        {lastResult.impact_metrics.new_makespan}
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Advanced Configuration Dialog */}
      <Dialog open={showAdvancedDialog} onClose={() => setShowAdvancedDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>高度な再最適化設定</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>戦略</InputLabel>
                <Select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                  {strategies.map((strat) => (
                    <MenuItem key={strat.value} value={strat.value}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {strat.icon}
                        <Box sx={{ ml: 1 }}>
                          <Typography variant="body2">{strat.label}</Typography>
                          {strat.description && (
                            <Typography variant="caption" color="text.secondary">
                              {strat.description}
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>最適化タイプ</InputLabel>
                <Select
                  value={reoptimizationType}
                  onChange={(e) => setReoptimizationType(e.target.value)}
                >
                  <MenuItem value="incremental">増分</MenuItem>
                  <MenuItem value="full">完全</MenuItem>
                  <MenuItem value="emergency">緊急</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="時間制限 (秒)"
                type="number"
                value={timeLimit}
                onChange={(e) => setTimeLimit(Number(e.target.value))}
                inputProps={{ min: 10, max: 600 }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="イベント閾値"
                type="number"
                value={eventThreshold}
                onChange={(e) => setEventThreshold(Number(e.target.value))}
                inputProps={{ min: 1, max: 10 }}
                helperText="この数のイベントで自動再最適化"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={preserveCompleted}
                    onChange={(e) => setPreserveCompleted(e.target.checked)}
                  />
                }
                label="完了済みジョブを保持"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoReoptimize}
                    onChange={(e) => setAutoReoptimize(e.target.checked)}
                  />
                }
                label="自動再最適化を有効化"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAdvancedDialog(false)}>
            キャンセル
          </Button>
          <Button 
            variant="contained"
            onClick={() => {
              executeReoptimization(strategy);
              setShowAdvancedDialog(false);
            }}
            disabled={loading}
          >
            実行
          </Button>
        </DialogActions>
      </Dialog>

      {/* History Dialog */}
      <Dialog open={showHistoryDialog} onClose={() => setShowHistoryDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle>再最適化履歴</DialogTitle>
        <DialogContent>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>実行時刻</TableCell>
                  <TableCell>戦略</TableCell>
                  <TableCell>トリガー</TableCell>
                  <TableCell>ステータス</TableCell>
                  <TableCell>実行時間</TableCell>
                  <TableCell>改善率</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {reoptimizationHistory.map((entry, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {new Date(entry.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStrategyIcon(entry.strategy)}
                        label={entry.strategy}
                        size="small"
                        sx={{ 
                          bgcolor: getStrategyColor(entry.strategy),
                          color: 'white',
                          '& .MuiChip-icon': { color: 'white' }
                        }}
                      />
                    </TableCell>
                    <TableCell>{entry.trigger}</TableCell>
                    <TableCell>
                      <Chip
                        label={entry.success ? '成功' : '失敗'}
                        color={entry.success ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{formatDuration(entry.optimization_time)}</TableCell>
                    <TableCell>
                      {entry.improvement_ratio !== undefined ? (
                        <Typography
                          color={entry.improvement_ratio > 0 ? 'success.main' : 'text.primary'}
                        >
                          {entry.improvement_ratio > 0 ? '+' : ''}
                          {entry.improvement_ratio.toFixed(2)}%
                        </Typography>
                      ) : (
                        '---'
                      )}
                    </TableCell>
                  </TableRow>
                ))}
                {reoptimizationHistory.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      履歴がありません
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowHistoryDialog(false)}>
            閉じる
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ReoptimizationManager;