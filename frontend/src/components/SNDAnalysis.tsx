import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Container,
  Button,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Switch,
  FormControlLabel,
  Slider
} from '@mui/material';
import { 
  Upload as UploadIcon,
  Settings as SettingsIcon, 
  Map as MapIcon,
  Assessment as ResultsIcon,
  GetApp as ExportIcon,
  DataObject as DataIcon,
  AccountTree as NetworkIcon,
  PlayArrow as ExecuteIcon,
  Visibility as VisualizeIcon,
  Timeline as MonitorIcon,
  Policy as PolicyIcon
} from '@mui/icons-material';
import SNDMapVisualization from './SNDMapVisualization';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`snd-tabpanel-${index}`}
      aria-labelledby={`snd-tab-${index}`}
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

function a11yProps(index: number) {
  return {
    id: `snd-tab-${index}`,
    'aria-controls': `snd-tabpanel-${index}`,
  };
}

const SNDAnalysis: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sampleData, setSampleData] = useState<any>(null);
  const [optimizationResults, setOptimizationResults] = useState<any>(null);
  const [visualizationData, setVisualizationData] = useState<any>(null);
  
  // Optimization parameters
  const [parameters, setParameters] = useState({
    costPerDistance: 20,
    costPerTime: 8000,
    capacity: 1000,
    maxCpuTime: 10,
    useScaling: true,
    kPaths: 10,
    alpha: 0.5,
    maxIterations: 100,
    useOSRM: false
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleGenerateSampleData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/snd/sample-data', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSampleData(data);
      setSessionId(data.session_id);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'サンプルデータの生成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    if (!sampleData) {
      setError('最初にデータを読み込んでください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/snd/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dc_data: sampleData.dc_data,
          od_data: Object.entries(sampleData.od_data).map(([origin, destinations]: [string, any]) => ({
            [origin]: destinations
          })).reduce((acc, curr) => ({ ...acc, ...curr }), {}),
          cost_per_distance: parameters.costPerDistance,
          cost_per_time: parameters.costPerTime,
          capacity: parameters.capacity,
          max_cpu_time: parameters.maxCpuTime,
          use_scaling: parameters.useScaling,
          k_paths: parameters.kPaths,
          alpha: parameters.alpha,
          max_iterations: parameters.maxIterations,
          use_osrm: parameters.useOSRM
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setOptimizationResults(results);
      setSessionId(results.session_id);
      
      // Move to results tab
      setCurrentTab(3);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateVisualization = async (destinationFilter?: string) => {
    if (!sessionId) {
      setError('セッションIDがありません');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/snd/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          destination_filter: destinationFilter || null
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const vizData = await response.json();
      setVisualizationData(vizData);
      
      // Move to visualization tab if not already there
      if (currentTab !== 2) {
        setCurrentTab(2);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '可視化データの生成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDestinationFilter = async (destination: string | null) => {
    await handleGenerateVisualization(destination || undefined);
  };

  const handleExportResults = async (format: string) => {
    if (!sessionId) {
      setError('セッションIDがありません');
      return;
    }

    try {
      const response = await fetch('/api/snd/export-results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          format: format
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Download the file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `snd_results.${format === 'excel' ? 'xlsx' : 'zip'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'エクスポートに失敗しました');
    }
  };

  return (
    <Container maxWidth="xl">
      <Paper elevation={3} sx={{ mt: 2, mb: 2 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="SND Analysis tabs">
            <Tab 
              label="システム設定" 
              icon={<SettingsIcon />} 
              iconPosition="start"
              {...a11yProps(0)} 
            />
            <Tab 
              label="データ管理" 
              icon={<DataIcon />}
              iconPosition="start" 
              {...a11yProps(1)} 
            />
            <Tab 
              label="ネットワークモデル" 
              icon={<NetworkIcon />}
              iconPosition="start" 
              {...a11yProps(2)} 
            />
            <Tab 
              label="最適化実行" 
              icon={<ExecuteIcon />}
              iconPosition="start" 
              {...a11yProps(3)} 
            />
            <Tab 
              label="結果可視化" 
              icon={<VisualizeIcon />}
              iconPosition="start" 
              {...a11yProps(4)} 
            />
            <Tab 
              label="リアルタイム監視" 
              icon={<MonitorIcon />}
              iconPosition="start" 
              {...a11yProps(5)} 
            />
            <Tab 
              label="ポリシー管理" 
              icon={<PolicyIcon />}
              iconPosition="start" 
              {...a11yProps(6)} 
            />
          </Tabs>
        </Box>

        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ m: 2 }}>
            {error}
          </Alert>
        )}

        <TabPanel value={currentTab} index={0}>
          <Typography variant="h6" gutterBottom>
            システム設定 - Service Network Design (SENDO)
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    サンプルデータ
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    日本の主要都市間のサンプルデータを使用して分析を開始
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={handleGenerateSampleData}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <UploadIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? '読み込み中...' : 'サンプルデータを生成'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    データアップロード
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    DC.csvとod.csvファイルをアップロード（実装予定）
                  </Typography>
                  <Button
                    variant="outlined"
                    disabled
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    CSVファイルアップロード（実装予定）
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {sampleData && (
            <Box sx={{ mt: 3 }}>
              <Alert severity="success">
                データが正常に読み込まれました: {sampleData.dc_data?.length || 0}箇所のDC
              </Alert>
            </Box>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={1}>
          <Typography variant="h6" gutterBottom>
            データ管理
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            ネットワーク設計に必要なデータのインポート・エクスポート・管理
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    データファイル管理
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<UploadIcon />}
                    fullWidth
                    sx={{ mb: 2 }}
                  >
                    データをインポート
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ExportIcon />}
                    fullWidth
                  >
                    データをエクスポート
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    データ概要
                  </Typography>
                  <Typography variant="body2">
                    ノード数: {sampleData ? sampleData.dc_data?.length : 0}<br/>
                    エッジ数: 0<br/>
                    需要ポイント: {sampleData ? Object.keys(sampleData.od_data || {}).length : 0}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    データ品質チェック
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    アップロードされたデータの品質と整合性をチェック
                  </Typography>
                  <Button
                    variant="outlined"
                    disabled={!sampleData}
                    fullWidth
                  >
                    データ品質チェック実行
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={2}>
          <Typography variant="h6" gutterBottom>
            ネットワークモデル設定
          </Typography>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    コスト設定
                  </Typography>
                  
                  <TextField
                    label="距離あたりコスト (円/km)"
                    type="number"
                    value={parameters.costPerDistance}
                    onChange={(e) => setParameters({...parameters, costPerDistance: Number(e.target.value)})}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="時間あたりコスト (円/時間)"
                    type="number"
                    value={parameters.costPerTime}
                    onChange={(e) => setParameters({...parameters, costPerTime: Number(e.target.value)})}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="車両容量"
                    type="number"
                    value={parameters.capacity}
                    onChange={(e) => setParameters({...parameters, capacity: Number(e.target.value)})}
                    fullWidth
                    margin="normal"
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    アルゴリズム設定
                  </Typography>
                  
                  <TextField
                    label="最大計算時間 (秒)"
                    type="number"
                    value={parameters.maxCpuTime}
                    onChange={(e) => setParameters({...parameters, maxCpuTime: Number(e.target.value)})}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="k最短路数"
                    type="number"
                    value={parameters.kPaths}
                    onChange={(e) => setParameters({...parameters, kPaths: Number(e.target.value)})}
                    fullWidth
                    margin="normal"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={parameters.useScaling}
                        onChange={(e) => setParameters({...parameters, useScaling: e.target.checked})}
                      />
                    }
                    label="勾配スケーリング法を使用"
                  />
                  
                  {parameters.useScaling && (
                    <>
                      <Typography gutterBottom>
                        アルファ値: {parameters.alpha}
                      </Typography>
                      <Slider
                        value={parameters.alpha}
                        onChange={(e, newValue) => setParameters({...parameters, alpha: newValue as number})}
                        min={0.1}
                        max={1.0}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                      
                      <TextField
                        label="最大反復回数"
                        type="number"
                        value={parameters.maxIterations}
                        onChange={(e) => setParameters({...parameters, maxIterations: Number(e.target.value)})}
                        fullWidth
                        margin="normal"
                      />
                    </>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

        </TabPanel>

        <TabPanel value={currentTab} index={3}>
          <Typography variant="h6" gutterBottom>
            最適化実行
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            設定されたパラメータでネットワーク設計最適化を実行します
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    最適化設定確認
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    距離コスト: {parameters.costPerDistance} 円/km<br/>
                    時間コスト: {parameters.costPerTime} 円/時間<br/>
                    車両容量: {parameters.capacity}
                  </Typography>
                  
                  <Box sx={{ textAlign: 'center', mt: 3 }}>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={handleOptimize}
                      disabled={loading || !sampleData}
                      startIcon={loading ? <CircularProgress size={20} /> : <ExecuteIcon />}
                    >
                      {loading ? '最適化実行中...' : '最適化を実行'}
                    </Button>
                    
                    {!sampleData && (
                      <Alert severity="warning" sx={{ mt: 2 }}>
                        まずシステム設定でサンプルデータを生成してください。
                      </Alert>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={4}>
          <Typography variant="h6" gutterBottom>
            結果可視化
          </Typography>

          <Box sx={{ mb: 2 }}>
            <Button
              variant="contained"
              onClick={() => handleGenerateVisualization()}
              disabled={loading || !sessionId}
              startIcon={loading ? <CircularProgress size={20} /> : <MapIcon />}
            >
              {loading ? '可視化生成中...' : '可視化を生成'}
            </Button>
          </Box>

          {visualizationData ? (
            <SNDMapVisualization 
              visualizationData={visualizationData}
              onDestinationFilter={handleDestinationFilter}
            />
          ) : (
            <Card>
              <CardContent>
                <Box sx={{ height: 400, bgcolor: 'grey.100', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="h6" color="text.secondary">
                    可視化を生成してください
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={5}>
          <Typography variant="h6" gutterBottom>
            リアルタイム監視
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            ネットワーク設計の最適化状況をリアルタイムで監視
          </Typography>
          
          {optimizationResults ? (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent sx={{ textAlign: 'center', bgcolor: 'success.light' }}>
                    <Typography variant="h4" color="white">
                      {optimizationResults.paths_generated || 0}
                    </Typography>
                    <Typography variant="subtitle2" color="white">生成パス数</Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent sx={{ textAlign: 'center', bgcolor: 'info.light' }}>
                    <Typography variant="h4" color="white">
                      {optimizationResults.computation_time?.toFixed(2) || 0}s
                    </Typography>
                    <Typography variant="subtitle2" color="white">計算時間</Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent sx={{ textAlign: 'center', bgcolor: 'warning.light' }}>
                    <Typography variant="h4" color="white">
                      ¥{optimizationResults.total_cost?.toLocaleString() || 0}
                    </Typography>
                    <Typography variant="subtitle2" color="white">総コスト</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          ) : (
            <Alert severity="info">
              リアルタイム監視を開始するには、まず最適化を実行してください。
            </Alert>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={6}>
          <Typography variant="h6" gutterBottom>
            ポリシー管理
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            ネットワーク設計ポリシーの管理とエクスポート機能
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    レポート・エクスポート
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<ExportIcon />}
                      disabled={!optimizationResults}
                    >
                      最適化結果をエクスポート
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<ExportIcon />}
                      disabled={!optimizationResults}
                    >
                      ネットワーク図をエクスポート
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<ExportIcon />}
                      disabled={!optimizationResults}
                    >
                      詳細レポート生成
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    設定ポリシー
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    最適化設定を保存・読み込みします
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button
                      variant="outlined"
                      disabled
                    >
                      設定を保存
                    </Button>
                    <Button
                      variant="outlined"
                      disabled
                    >
                      設定を読み込み
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default SNDAnalysis;