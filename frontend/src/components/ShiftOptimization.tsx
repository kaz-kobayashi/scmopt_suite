import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardActions,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Collapse,
  FormControlLabel,
  Switch
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  CloudDownload as DownloadIcon,
  Assessment as AssessmentIcon,
  Schedule as ScheduleIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  PlayArrow as PlayIcon,
  TableChart as TableChartIcon,
  ViewColumn as ViewColumnIcon,
  Settings as SettingsIcon,
  People as PeopleIcon,
  Business as BusinessIcon,
  Timeline as TimelineIcon,
  Storage as DataIcon,
  BarChart as AnalyticsIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import apiClient from '../services/apiClient';
import ShiftVisualization from './ShiftVisualization';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

interface StaffData {
  name: string;
  wage_per_period: number;
  max_period: number;
  max_day: number;
  job_set: number[];
  day_off: number[];
  start: number;
  end: number;
  request?: { [key: number]: number[] };
}

interface RequirementData {
  day_type: string;
  job: number;
  period: number;
  requirement: number;
}

const ShiftOptimization: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // データ生成用の状態
  const [dataGenParams, setDataGenParams] = useState({
    start_date: '2024-01-01',
    end_date: '2024-01-07',
    start_time: '09:00',
    end_time: '21:00',
    freq: '1h',
    job_list: ['レジ打ち', '接客']
  });

  // 生成されたサンプルデータ
  const [sampleData, setSampleData] = useState<any>(null);
  
  // 最適化結果
  const [optimizationResult, setOptimizationResult] = useState<any>(null);

  // 最適化パラメータ
  const [optimizationParams, setOptimizationParams] = useState({
    theta: 1,
    lb_penalty: 10000,
    ub_penalty: 0,
    job_change_penalty: 10,
    break_penalty: 10000,
    max_day_penalty: 5000,
    time_limit: 30,
    random_seed: 1
  });

  // 日別リクエスト機能の使用フラグ
  const [useRequestMode, setUseRequestMode] = useState(false);

  // ダイアログの状態
  const [staffDialogOpen, setStaffDialogOpen] = useState(false);
  const [requirementDialogOpen, setRequirementDialogOpen] = useState(false);

  // 展開状態の管理
  const [expandedSections, setExpandedSections] = useState<{ [key: string]: boolean }>({
    staff: false,
    requirement: false,
    result: false
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError(null);
    setSuccess(null);
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // サンプルデータ生成
  const generateSampleData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post('/shift/generate-sample-data', dataGenParams);
      setSampleData(response.data);
      setSuccess('サンプルデータが正常に生成されました');
    } catch (err: any) {
      setError(`データ生成エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // シフト最適化実行
  const runOptimization = async () => {
    if (!sampleData) {
      setError('まずサンプルデータを生成してください');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const requestData = {
        period_df: sampleData.period_df,
        break_df: sampleData.break_df,
        day_df: sampleData.day_df,
        job_df: sampleData.job_df,
        staff_df: sampleData.staff_df.map((staff: any) => ({
          ...staff,
          job_set: Array.isArray(staff.job_set) ? staff.job_set : JSON.parse(staff.job_set || '[]'),
          day_off: Array.isArray(staff.day_off) ? staff.day_off : JSON.parse(staff.day_off || '[]')
        })),
        requirement_df: sampleData.requirement_df,
        ...optimizationParams
      };

      // 日別リクエスト対応版を使用するかどうか
      const endpoint = useRequestMode ? '/shift/optimize-with-requests' : '/shift/optimize';
      const response = await apiClient.post(endpoint, requestData);
      setOptimizationResult(response.data);
      
      if (response.data.status === 0) {
        setSuccess(`シフト最適化が正常に完了しました${useRequestMode ? '（日別リクエスト対応版）' : ''}`);
      } else {
        setError(`最適化エラー: ${response.data.message}`);
      }
    } catch (err: any) {
      setError(`最適化エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 実行可能性分析
  const analyzeFeasibility = async () => {
    if (!sampleData) {
      setError('まずサンプルデータを生成してください');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        period_data: JSON.stringify(sampleData.period_df),
        day_data: JSON.stringify(sampleData.day_df),
        job_data: JSON.stringify(sampleData.job_df),
        staff_data: JSON.stringify(sampleData.staff_df),
        requirement_data: JSON.stringify(sampleData.requirement_df)
      });

      const response = await apiClient.get(`/shift/estimate-feasibility?${params}`);
      // 実行可能性チャートの表示処理をここに追加
      setSuccess('実行可能性分析が完了しました');
    } catch (err: any) {
      setError(`分析エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Excel出力
  const exportToExcel = async () => {
    if (!optimizationResult) {
      setError('まず最適化を実行してください');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post('/shift/export-excel', optimizationResult, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `shift_schedule_${new Date().toISOString().slice(0, 10)}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      setSuccess('Excelファイルがダウンロードされました');
    } catch (err: any) {
      setError(`Excel出力エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ガントチャートExcel出力
  const exportGanttToExcel = async () => {
    if (!optimizationResult) {
      setError('まず最適化を実行してください');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post('/shift/export-gantt-excel', {
        job_assign: optimizationResult.job_assign,
        sampleData: sampleData
      }, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `shift_gantt_${new Date().toISOString().slice(0, 10)}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      setSuccess('ガントチャートExcelファイルがダウンロードされました');
    } catch (err: any) {
      setError(`ガントチャートExcel出力エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 全シフトExcel出力
  const exportAllShiftToExcel = async () => {
    if (!optimizationResult) {
      setError('まず最適化を実行してください');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.post('/shift/export-allshift-excel', {
        optimizationResult: optimizationResult,
        sampleData: sampleData
      }, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `all_shifts_${new Date().toISOString().slice(0, 10)}.xlsx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      setSuccess('全シフトExcelファイルがダウンロードされました');
    } catch (err: any) {
      setError(`全シフトExcel出力エラー: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        <ScheduleIcon sx={{ mr: 2, verticalAlign: 'middle' }} />
        シフト最適化システム
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="shift optimization tabs">
          <Tab label="システム設定" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="スタッフ管理" icon={<PeopleIcon />} iconPosition="start" />
          <Tab label="シフト要件" icon={<BusinessIcon />} iconPosition="start" />
          <Tab label="最適化実行" icon={<PlayIcon />} iconPosition="start" />
          <Tab label="結果分析" icon={<AssessmentIcon />} iconPosition="start" />
          <Tab label="リアルタイム監視" icon={<TimelineIcon />} iconPosition="start" />
          <Tab label="データ管理" icon={<DataIcon />} iconPosition="start" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    サンプルデータ生成
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="開始日"
                        type="date"
                        value={dataGenParams.start_date}
                        onChange={(e) => setDataGenParams(prev => ({...prev, start_date: e.target.value}))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="終了日"
                        type="date"
                        value={dataGenParams.end_date}
                        onChange={(e) => setDataGenParams(prev => ({...prev, end_date: e.target.value}))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="開始時刻"
                        type="time"
                        value={dataGenParams.start_time}
                        onChange={(e) => setDataGenParams(prev => ({...prev, start_time: e.target.value}))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <TextField
                        fullWidth
                        label="終了時刻"
                        type="time"
                        value={dataGenParams.end_time}
                        onChange={(e) => setDataGenParams(prev => ({...prev, end_time: e.target.value}))}
                        InputLabelProps={{ shrink: true }}
                      />
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <FormControl fullWidth>
                        <InputLabel>時間間隔</InputLabel>
                        <Select
                          value={dataGenParams.freq}
                          onChange={(e) => setDataGenParams(prev => ({...prev, freq: e.target.value}))}
                        >
                          <MenuItem value="30min">30分</MenuItem>
                          <MenuItem value="1h">1時間</MenuItem>
                          <MenuItem value="2h">2時間</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={8}>
                      <TextField
                        fullWidth
                        label="ジョブリスト（カンマ区切り）"
                        value={dataGenParams.job_list.join(', ')}
                        onChange={(e) => setDataGenParams(prev => ({
                          ...prev, 
                          job_list: e.target.value.split(',').map(s => s.trim())
                        }))}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
                <CardActions>
                  <Button
                    variant="contained"
                    onClick={generateSampleData}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
                  >
                    サンプルデータ生成
                  </Button>
                </CardActions>
              </Card>
            </Grid>

          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {/* スタッフ管理 */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    スタッフ管理
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    スタッフの情報、スキル、勤務可能時間などを管理します
                  </Typography>
                  
                  {sampleData?.staff_df && sampleData.staff_df.length > 0 ? (
                    <Box>
                      <Typography variant="subtitle1" gutterBottom>
                        登録スタッフ一覧（{sampleData.staff_df.length}名）
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>スタッフ名</TableCell>
                              <TableCell>最大連続勤務日数</TableCell>
                              <TableCell>担当可能ジョブ</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {sampleData.staff_df.map((staff: any, index: number) => (
                              <TableRow key={index}>
                                <TableCell>{staff.name}</TableCell>
                                <TableCell>{staff.max_consecutive_days}日</TableCell>
                                <TableCell>
                                  {staff.job_set && staff.job_set.map((jobId: number, jobIndex: number) => (
                                    <Chip 
                                      key={jobIndex}
                                      label={sampleData.job_df?.find((j: any) => j.id === jobId)?.description || `Job ${jobId}`} 
                                      size="small" 
                                      sx={{ mr: 0.5 }}
                                      color="primary"
                                    />
                                  ))}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  ) : (
                    <Alert severity="info">
                      スタッフ情報を表示するには、まず「システム設定」タブでサンプルデータを生成してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {/* シフト要件 */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    シフト要件
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    各日、各時間帯のシフト要件（必要人数）を設定します
                  </Typography>
                  
                  {sampleData?.requirement_df && sampleData.requirement_df.length > 0 ? (
                    <Box>
                      <Typography variant="subtitle1" gutterBottom>
                        シフト要件一覧
                      </Typography>
                      <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
                        <Table stickyHeader>
                          <TableHead>
                            <TableRow>
                              <TableCell>日種別</TableCell>
                              <TableCell>時間帯</TableCell>
                              <TableCell>ジョブ</TableCell>
                              <TableCell>必要人数</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {sampleData.requirement_df.slice(0, 50).map((req: any, index: number) => (
                              <TableRow key={index}>
                                <TableCell>{req.day_type}</TableCell>
                                <TableCell>{sampleData.period_df?.find((p: any) => p.id === req.period)?.description || `Period ${req.period}`}</TableCell>
                                <TableCell>
                                  <Chip label={sampleData.job_df?.find((j: any) => j.id === req.job)?.description || `Job ${req.job}`} size="small" color="primary" />
                                </TableCell>
                                <TableCell>{req.requirement}名</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      {sampleData.requirement_df.length > 50 && (
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          表示を50件に制限しています。全{sampleData.requirement_df.length}件のデータが存在します。
                        </Typography>
                      )}
                    </Box>
                  ) : (
                    <Alert severity="info">
                      シフト要件を表示するには、まず「システム設定」タブでサンプルデータを生成してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          {/* 最適化実行 */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    最適化実行
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    シフトスケジュールの最適化を実行します
                  </Typography>
                  
                  {sampleData ? (
                    <Box>
                      <Grid container spacing={2} sx={{ mb: 3 }}>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>データ概要</Typography>
                            <Typography variant="body2">スタッフ数: {sampleData.staff_df?.length || 0}名</Typography>
                            <Typography variant="body2">対象期間: {sampleData.day_df?.length || 0}日間</Typography>
                            <Typography variant="body2">シフト要件数: {sampleData.requirement_df?.length || 0}件</Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>最適化設定</Typography>
                            <Typography variant="body2">アルゴリズム: 混合整数計画法</Typography>
                            <Typography variant="body2">目的: コスト最小化</Typography>
                            <Typography variant="body2">制約: 勤務規則準拠</Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                      
                      <Button
                        variant="contained"
                        color="primary"
                        size="large"
                        onClick={runOptimization}
                        disabled={loading}
                        startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
                        fullWidth
                      >
                        {loading ? 'シフト最適化を実行中...' : 'シフト最適化を実行'}
                      </Button>
                      
                      {optimizationResult && (
                        <Alert severity="success" sx={{ mt: 2 }}>
                          最適化が完了しました。「結果分析」タブで詳細を確認してください。
                        </Alert>
                      )}
                    </Box>
                  ) : (
                    <Alert severity="warning">
                      最適化を実行するには、まず「システム設定」タブでサンプルデータを生成してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          {/* Results Analysis */}
          {optimizationResult ? (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      最適化結果概要
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      ステータス: {optimizationResult.status === 0 ? '成功' : 'エラー'}
                    </Typography>
                    {optimizationResult.message && (
                      <Typography variant="body2" color="textSecondary">
                        {optimizationResult.message}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <ShiftVisualization 
                  optimizationResult={optimizationResult}
                  sampleData={sampleData}
                />
              </Grid>

              {optimizationResult.cost_df && (
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        コスト内訳
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>項目</TableCell>
                              <TableCell align="right">値</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {optimizationResult.cost_df.map((row: any, index: number) => (
                              <TableRow key={index}>
                                <TableCell>{row.penalty}</TableCell>
                                <TableCell align="right">{row.value}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {optimizationResult.staff_df && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Box display="flex" alignItems="center" onClick={() => toggleSection('result')}>
                        <Typography variant="h6" sx={{ flexGrow: 1, cursor: 'pointer' }}>
                          スタッフシフト結果 ({optimizationResult.staff_df.length}名)
                        </Typography>
                        <IconButton>
                          {expandedSections.result ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Box>
                      <Collapse in={expandedSections.result}>
                        <TableContainer sx={{ mt: 2, maxHeight: 400 }}>
                          <Table stickyHeader size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>スタッフ名</TableCell>
                                <TableCell>最大勤務期間違反</TableCell>
                                {sampleData?.day_df?.map((day: any, index: number) => (
                                  <TableCell key={index}>{day.day}</TableCell>
                                ))}
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {optimizationResult.staff_df.map((staff: any, index: number) => (
                                <TableRow key={index}>
                                  <TableCell>{staff.name}</TableCell>
                                  <TableCell>{staff['max day violation']}</TableCell>
                                  {sampleData?.day_df?.map((day: any, dayIndex: number) => {
                                    const shiftKey = `Shift for Day ${dayIndex}`;
                                    const shiftValue = staff[shiftKey];
                                    return (
                                      <TableCell key={dayIndex}>
                                        {shiftValue ? (
                                          <Chip 
                                            label={shiftValue} 
                                            size="small" 
                                            color={shiftValue.includes('_') ? 'primary' : 'default'}
                                          />
                                        ) : (
                                          <Chip label="休み" size="small" color="secondary" />
                                        )}
                                      </TableCell>
                                    );
                                  })}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Collapse>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          ) : (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="textSecondary">
                最適化を実行すると結果がここに表示されます
              </Typography>
            </Box>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={5}>
          {/* Real-time Monitoring */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    リアルタイムシフト監視
                  </Typography>
                  
                  {optimizationResult ? (
                    <Box>
                      <Typography variant="body1" gutterBottom>
                        現在のシフト状況をリアルタイムで監視します。
                      </Typography>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                            <Typography variant="h4" color="white">
                              {optimizationResult.staff_df?.filter((s: any) => s['max day violation'] === 0).length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">違反なしスタッフ</Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                            <Typography variant="h4" color="white">
                              {optimizationResult.staff_df?.filter((s: any) => s['max day violation'] > 0).length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">違反ありスタッフ</Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light' }}>
                            <Typography variant="h4" color="white">
                              {optimizationResult.staff_df?.length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">総スタッフ数</Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                      
                      <Button
                        variant="outlined"
                        startIcon={<AnalyticsIcon />}
                        sx={{ mt: 2 }}
                        onClick={analyzeFeasibility}
                        disabled={loading}
                      >
                        実行可能性分析
                      </Button>
                    </Box>
                  ) : (
                    <Alert severity="info">
                      リアルタイム監視を開始するには、まず最適化を実行してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={6}>
          {/* Data Management */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    データ管理・エクスポート
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    最適化結果とデータのエクスポート機能
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item>
                      <Button
                        variant="outlined"
                        onClick={exportToExcel}
                        disabled={loading || !optimizationResult}
                        startIcon={<DownloadIcon />}
                      >
                        基本Excel出力
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="outlined"
                        onClick={exportGanttToExcel}
                        disabled={loading || !optimizationResult}
                        startIcon={<TableChartIcon />}
                      >
                        ガントチャートExcel
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="outlined"
                        onClick={exportAllShiftToExcel}
                        disabled={loading || !optimizationResult}
                        startIcon={<ViewColumnIcon />}
                      >
                        全シフトExcel
                      </Button>
                    </Grid>
                    <Grid item>
                      <Button
                        variant="outlined"
                        onClick={analyzeFeasibility}
                        disabled={loading || !sampleData}
                        startIcon={<AssessmentIcon />}
                      >
                        実行可能性分析
                      </Button>
                    </Grid>
                  </Grid>
                  
                  {!optimizationResult && (
                    <Alert severity="warning" sx={{ mt: 2 }}>
                      エクスポート機能を使用するには、まず最適化を実行してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>


      </Paper>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <CircularProgress />
        </Box>
      )}
    </Box>
  );
};

export default ShiftOptimization;