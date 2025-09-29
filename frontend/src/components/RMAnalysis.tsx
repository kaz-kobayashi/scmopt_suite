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
  Slider,
  Link
} from '@mui/material';
import { 
  DataObject as DataIcon,
  TrendingUp as PricingIcon,
  Functions as ValueIcon,
  Insights as OptimizeIcon,
  Policy as PolicyIcon,
  Psychology as ProspectIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  Assessment as AnalyticsIcon,
  Timeline as MonitorIcon,
  Storage as DataManagementIcon
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface SampleData {
  demand: Record<string, number>;
  revenue: Record<string, number>;
  usage_matrix: Record<string, number>;
  capacity: Record<string, number>;
  description?: string;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`rm-tabpanel-${index}`}
      aria-labelledby={`rm-tab-${index}`}
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
    id: `rm-tab-${index}`,
    'aria-controls': `rm-tabpanel-${index}`,
  };
}

const RMAnalysis: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleData, setSampleData] = useState<SampleData | null>(null);
  
  // Dynamic Pricing State
  const [dynamicPricingParams, setDynamicPricingParams] = useState({
    actions: [15.0, 20.0, 25.0, 30.0, 35.0],
    epochs: 10,
    beta0: 50.0,
    beta1: -1.0,
    sigma: 1.0,
    tPeriods: 512,
    delta: 0.1,
    scaling: 1.0
  });
  const [dynamicPricingResults, setDynamicPricingResults] = useState<any>(null);
  
  // Value Function State
  const [valueFunctionParams, setValueFunctionParams] = useState({
    capacity: 1000,
    periods: 50,
    actions: [15.0, 20.0, 25.0, 30.0, 35.0],
    beta0: 50.0,
    beta1: -1.0,
    sigma: 0.0,
    nSamples: 1
  });
  const [valueFunctionResults, setValueFunctionResults] = useState<any>(null);
  
  // Revenue Management State
  const [revenueOptimizationResults, setRevenueOptimizationResults] = useState<any>(null);
  
  // Control Policy State
  const [controlPolicyResults, setControlPolicyResults] = useState<any>(null);
  
  // Prospect Theory State
  const [prospectParams, setProspectParams] = useState({
    periods: 50,
    alpha: 0.5,
    beta: 0.8,
    gamma: 0.8,
    zeta: 8.0,
    eta: 12.0,
    initialReferencePrice: 25.0,
    baseDemandParams: [100.0, 2.0, 25.0] // [d0, a, p0]
  });
  const [prospectResults, setProspectResults] = useState<any>(null);

  // CSV Upload State
  const [uploadDataType, setUploadDataType] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadResults, setUploadResults] = useState<any>(null);
  const [uploadedData, setUploadedData] = useState<{[key: string]: any}>({});

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleGenerateSampleData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/sample-data', {
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
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'サンプルデータの生成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDynamicPricing = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/dynamic-pricing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          actions: dynamicPricingParams.actions,
          epochs: dynamicPricingParams.epochs,
          beta_params: [dynamicPricingParams.beta0, dynamicPricingParams.beta1],
          sigma: dynamicPricingParams.sigma,
          t_periods: dynamicPricingParams.tPeriods,
          delta: dynamicPricingParams.delta,
          scaling: dynamicPricingParams.scaling
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setDynamicPricingResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '動的価格最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleValueFunctionCalculation = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/value-function', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          capacity: valueFunctionParams.capacity,
          periods: valueFunctionParams.periods,
          actions: valueFunctionParams.actions,
          beta_params: [valueFunctionParams.beta0, valueFunctionParams.beta1],
          sigma: valueFunctionParams.sigma,
          n_samples: valueFunctionParams.nSamples
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setValueFunctionResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '価値関数計算に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRevenueOptimization = async (method: number) => {
    if (!sampleData) {
      setError('まずサンプルデータを生成してください');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/revenue-optimization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          demand: sampleData.demand,
          capacity: sampleData.capacity,
          revenue: sampleData.revenue,
          usage_matrix: sampleData.usage_matrix,
          method: method,
          n_samples: 100
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setRevenueOptimizationResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '収益管理最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleBidPriceControl = async () => {
    if (!sampleData) {
      setError('まずサンプルデータを生成してください');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/bid-price-control', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          demand: sampleData.demand,
          capacity: sampleData.capacity,
          revenue: sampleData.revenue,
          usage_matrix: sampleData.usage_matrix,
          method: 0,
          n_samples: 100,
          random_seed: 123
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setControlPolicyResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '入札価格コントロールに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleNestedBookingLimit = async () => {
    if (!sampleData) {
      setError('まずサンプルデータを生成してください');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/nested-booking-limit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          demand: sampleData.demand,
          capacity: sampleData.capacity,
          revenue: sampleData.revenue,
          usage_matrix: sampleData.usage_matrix,
          method: 0,
          n_samples: 100,
          random_seed: 123
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setControlPolicyResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : '入れ子上限コントロールに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleProspectPricing = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/rm/prospect-pricing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          periods: prospectParams.periods,
          alpha: prospectParams.alpha,
          beta: prospectParams.beta,
          gamma: prospectParams.gamma,
          zeta: prospectParams.zeta,
          eta: prospectParams.eta,
          initial_reference_price: prospectParams.initialReferencePrice,
          base_demand_params: prospectParams.baseDemandParams
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setProspectResults(results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'プロスペクト理論分析に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleCSVUpload = async () => {
    if (!selectedFile || !uploadDataType) {
      setError('ファイルとデータタイプを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('data_type', uploadDataType);

      const response = await fetch('/api/rm/upload-csv', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      setUploadResults(results);
      
      // Store uploaded data for later use
      setUploadedData({
        ...uploadedData,
        [uploadDataType]: results.parsed_data
      });

      // If we have all required data types, update sampleData
      const newUploadedData = { ...uploadedData, [uploadDataType]: results.parsed_data };
      if (newUploadedData.demand && newUploadedData.revenue && 
          newUploadedData.capacity && newUploadedData.usage_matrix) {
        setSampleData({
          demand: newUploadedData.demand,
          revenue: newUploadedData.revenue,
          capacity: newUploadedData.capacity,
          usage_matrix: newUploadedData.usage_matrix,
          description: 'Uploaded CSV data'
        });
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'CSVアップロードに失敗しました');
    } finally {
      setLoading(false);
      setSelectedFile(null);
      const fileInput = document.getElementById('csv-upload') as HTMLInputElement;
      if (fileInput) {
        fileInput.value = '';
      }
    }
  };

  const handleDownloadSample = async (dataType: string) => {
    try {
      const response = await fetch(`/api/rm/download-sample/${dataType}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${dataType}_sample.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      setError(`サンプルファイルのダウンロードに失敗しました: ${error}`);
    }
  };

  return (
    <Container maxWidth="xl">
      <Paper elevation={3} sx={{ mt: 2, mb: 2 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="RM Analysis tabs">
            <Tab 
              label="システム設定" 
              icon={<SettingsIcon />} 
              iconPosition="start"
              {...a11yProps(0)} 
            />
            <Tab 
              label="需要モデル" 
              icon={<PricingIcon />}
              iconPosition="start" 
              {...a11yProps(1)} 
            />
            <Tab 
              label="価値関数" 
              icon={<ValueIcon />}
              iconPosition="start" 
              {...a11yProps(2)} 
            />
            <Tab 
              label="収益最適化" 
              icon={<OptimizeIcon />}
              iconPosition="start" 
              {...a11yProps(3)} 
            />
            <Tab 
              label="制御方策" 
              icon={<PolicyIcon />}
              iconPosition="start" 
              {...a11yProps(4)} 
            />
            <Tab 
              label="結果分析" 
              icon={<AnalyticsIcon />}
              iconPosition="start" 
              {...a11yProps(5)} 
            />
            <Tab 
              label="データ管理" 
              icon={<DataManagementIcon />}
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
            システム設定 - Revenue Management (MERMO)
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    システム初期化
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    収益管理システムの基本設定とサンプルデータ生成
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={handleGenerateSampleData}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <DataIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? '初期化中...' : 'サンプルデータ生成'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            
            {/* System Configuration */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    システムパラメータ
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="最大サンプル数"
                        type="number"
                        defaultValue="100"
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="乱数シード"
                        type="number"
                        defaultValue="123"
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth margin="normal">
                        <InputLabel>除外手法</InputLabel>
                        <Select defaultValue="deterministic" label="除外手法">
                          <MenuItem value="deterministic">確定的モデル</MenuItem>
                          <MenuItem value="sampling">サンプリングモデル</MenuItem>
                          <MenuItem value="recourse">リコースモデル</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    CSVデータアップロード
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    需要、収益、容量、使用関係マトリックスのCSVファイルアップロード
                  </Typography>
                  
                  <Box sx={{ mb: 2, p: 1, bgcolor: 'info.light', borderRadius: 1 }}>
                    <Typography variant="body2" color="info.contrastText">
                      サンプルファイル: 
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('demand')}
                        sx={{ mx: 1 }}
                      >
                        需要データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('revenue')}
                        sx={{ mx: 1 }}
                      >
                        収益データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('capacity')}
                        sx={{ mx: 1 }}
                      >
                        容量データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('usage_matrix')}
                        sx={{ mx: 1 }}
                      >
                        使用関係
                      </Link>
                    </Typography>
                  </Box>
                  
                  <FormControl fullWidth margin="normal">
                    <InputLabel>データタイプ</InputLabel>
                    <Select
                      value={uploadDataType}
                      onChange={(e) => setUploadDataType(e.target.value)}
                      label="データタイプ"
                    >
                      <MenuItem value="demand">需要データ</MenuItem>
                      <MenuItem value="revenue">収益データ</MenuItem>
                      <MenuItem value="capacity">容量データ</MenuItem>
                      <MenuItem value="usage_matrix">使用関係マトリックス</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    id="csv-upload"
                  />
                  <label htmlFor="csv-upload">
                    <Button
                      variant="outlined"
                      component="span"
                      disabled={loading}
                      fullWidth
                      sx={{ mt: 1, mb: 1 }}
                    >
                      CSVファイル選択
                    </Button>
                  </label>
                  
                  {selectedFile && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      選択ファイル: {selectedFile.name}
                    </Typography>
                  )}
                  
                  <Button
                    variant="contained"
                    onClick={handleCSVUpload}
                    disabled={loading || !selectedFile || !uploadDataType}
                    fullWidth
                    startIcon={loading ? <CircularProgress size={20} /> : null}
                  >
                    {loading ? 'アップロード中...' : 'CSVアップロード'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {sampleData && (
            <Box sx={{ mt: 3 }}>
              <Alert severity="success">
                データが正常に読み込まれました: {Object.keys(sampleData.demand).length}件のサービス、{Object.keys(sampleData.capacity).length}件のリソース
              </Alert>
              
              <Card sx={{ mt: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>データ概要</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>項目</TableCell>
                          <TableCell>値</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>サービス数</TableCell>
                          <TableCell>{Object.keys(sampleData.demand).length}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>リソース数</TableCell>
                          <TableCell>{Object.keys(sampleData.capacity).length}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>総需要量</TableCell>
                          <TableCell>{Object.values(sampleData.demand).reduce((a, b) => Number(a) + Number(b), 0)}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>総収益ポテンシャル</TableCell>
                          <TableCell>¥{Object.values(sampleData.revenue).reduce((a, b) => Number(a) + Number(b), 0).toLocaleString()}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Box>
          )}

          {uploadResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>アップロード結果</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {uploadResults.row_count}
                      </Typography>
                      <Typography variant="body2">データ行数</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {uploadResults.column_count}
                      </Typography>
                      <Typography variant="body2">列数</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="info.main">
                        {uploadResults.data_type}
                      </Typography>
                      <Typography variant="body2">データタイプ</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color={uploadResults.validation_status === 'success' ? 'success.main' : 'error.main'}>
                        {uploadResults.validation_status === 'success' ? '成功' : 'エラー'}
                      </Typography>
                      <Typography variant="body2">検証状態</Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                {uploadResults.error_messages && uploadResults.error_messages.length > 0 && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      {uploadResults.error_messages.join(', ')}
                    </Typography>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}

          {Object.keys(uploadedData).length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>アップロード済みデータ</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>データタイプ</TableCell>
                        <TableCell>要素数</TableCell>
                        <TableCell>状態</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(uploadedData).map(([dataType, data]) => (
                        <TableRow key={dataType}>
                          <TableCell>{dataType}</TableCell>
                          <TableCell>{Object.keys(data as object).length}</TableCell>
                          <TableCell>
                            <Typography color="success.main">✓ 完了</Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={1}>
          <Typography variant="h6" gutterBottom>
            需要モデル - 動的価格最適化
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    パラメータ設定
                  </Typography>
                  
                  <TextField
                    label="価格候補 (カンマ区切り)"
                    value={dynamicPricingParams.actions.join(',')}
                    onChange={(e) => setDynamicPricingParams({
                      ...dynamicPricingParams, 
                      actions: e.target.value.split(',').map(Number)
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="エポック数"
                    type="number"
                    value={dynamicPricingParams.epochs}
                    onChange={(e) => setDynamicPricingParams({
                      ...dynamicPricingParams, 
                      epochs: Number(e.target.value)
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="需要関数切片 (β₀)"
                    type="number"
                    value={dynamicPricingParams.beta0}
                    onChange={(e) => setDynamicPricingParams({
                      ...dynamicPricingParams, 
                      beta0: Number(e.target.value)
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="需要関数傾き (β₁)"
                    type="number"
                    value={dynamicPricingParams.beta1}
                    onChange={(e) => setDynamicPricingParams({
                      ...dynamicPricingParams, 
                      beta1: Number(e.target.value)
                    })}
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
                    実行
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleDynamicPricing}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <PricingIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? '最適化実行中...' : '動的価格最適化を実行'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {dynamicPricingResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>結果</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        ¥{dynamicPricingResults.total_reward?.toLocaleString() || 0}
                      </Typography>
                      <Typography variant="body2">総収益</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {dynamicPricingResults.estimated_beta?.[0]?.toFixed(2) || 0}
                      </Typography>
                      <Typography variant="body2">推定β₀</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {dynamicPricingResults.estimated_beta?.[1]?.toFixed(2) || 0}
                      </Typography>
                      <Typography variant="body2">推定β₁</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={2}>
          <Typography variant="h6" gutterBottom>
            価値関数 - 動的計画法
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    パラメータ設定
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="在庫容量"
                        type="number"
                        value={valueFunctionParams.capacity}
                        onChange={(e) => setValueFunctionParams({
                          ...valueFunctionParams, 
                          capacity: Number(e.target.value)
                        })}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="期間数"
                        type="number"
                        value={valueFunctionParams.periods}
                        onChange={(e) => setValueFunctionParams({
                          ...valueFunctionParams, 
                          periods: Number(e.target.value)
                        })}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                  </Grid>
                  
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleValueFunctionCalculation}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <ValueIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? '計算中...' : '価値関数を計算'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              {valueFunctionResults && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>シミュレーション結果</Typography>
                    <Typography variant="h4" color="primary" gutterBottom>
                      ¥{valueFunctionResults.total_reward?.toLocaleString() || 0}
                    </Typography>
                    <Typography variant="body2">総収益</Typography>
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={3}>
          <Typography variant="h6" gutterBottom>
            収益最適化 - 最適化モデル
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    最適化手法
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    3つの異なるアプローチで収益管理問題を解きます
                  </Typography>
                  
                  <Grid container spacing={2} sx={{ mt: 2 }}>
                    <Grid item xs={12} md={4}>
                      <Button
                        variant="contained"
                        onClick={() => handleRevenueOptimization(0)}
                        disabled={loading || !sampleData}
                        fullWidth
                      >
                        確定的モデル
                      </Button>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Button
                        variant="contained"
                        onClick={() => handleRevenueOptimization(1)}
                        disabled={loading || !sampleData}
                        fullWidth
                      >
                        サンプリングモデル
                      </Button>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Button
                        variant="contained"
                        onClick={() => handleRevenueOptimization(2)}
                        disabled={loading || !sampleData}
                        fullWidth
                      >
                        リコースモデル
                      </Button>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {revenueOptimizationResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  最適化結果 ({revenueOptimizationResults.method_used})
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        ¥{revenueOptimizationResults.objective_value?.toLocaleString() || 0}
                      </Typography>
                      <Typography variant="body2">目的関数値</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {revenueOptimizationResults.computation_time?.toFixed(3) || 0}s
                      </Typography>
                      <Typography variant="body2">計算時間</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {Object.keys(revenueOptimizationResults.dual_variables || {}).length}
                      </Typography>
                      <Typography variant="body2">双対変数数</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={4}>
          <Typography variant="h6" gutterBottom>
            制御方策 - シミュレーション
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    入札価格コントロール方策
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    双対変数を用いた入札価格による需要受入判定
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={handleBidPriceControl}
                    disabled={loading || !sampleData}
                    startIcon={loading ? <CircularProgress size={20} /> : <PolicyIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? 'シミュレーション実行中...' : '入札価格シミュレーション'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    入れ子上限コントロール方策
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    優先順位に基づく座席数上限設定
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={handleNestedBookingLimit}
                    disabled={loading || !sampleData}
                    startIcon={loading ? <CircularProgress size={20} /> : <PolicyIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? 'シミュレーション実行中...' : '入れ子上限シミュレーション'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {controlPolicyResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  制御方策結果
                </Typography>
                <Typography variant="h4" color="primary" gutterBottom>
                  ¥{controlPolicyResults.total_revenue?.toLocaleString() || 0}
                </Typography>
                <Typography variant="body2">
                  総収益 ({controlPolicyResults.method_used})
                </Typography>
              </CardContent>
            </Card>
          )}
        </TabPanel>

        <TabPanel value={currentTab} index={5}>
          <Typography variant="h6" gutterBottom>
            結果分析 - プロスペクト理論分析
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    パラメータ設定
                  </Typography>
                  
                  <TextField
                    label="基本需要量 (d0)"
                    type="number"
                    value={prospectParams.baseDemandParams[0]}
                    onChange={(e) => setProspectParams({
                      ...prospectParams, 
                      baseDemandParams: [Number(e.target.value), prospectParams.baseDemandParams[1], prospectParams.baseDemandParams[2]]
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="価格感度 (a)"
                    type="number"
                    value={prospectParams.baseDemandParams[1]}
                    onChange={(e) => setProspectParams({
                      ...prospectParams, 
                      baseDemandParams: [prospectParams.baseDemandParams[0], Number(e.target.value), prospectParams.baseDemandParams[2]]
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="基準価格 (p0)"
                    type="number"
                    value={prospectParams.baseDemandParams[2]}
                    onChange={(e) => setProspectParams({
                      ...prospectParams, 
                      baseDemandParams: [prospectParams.baseDemandParams[0], prospectParams.baseDemandParams[1], Number(e.target.value)]
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <TextField
                    label="初期参照価格"
                    type="number"
                    value={prospectParams.initialReferencePrice}
                    onChange={(e) => setProspectParams({
                      ...prospectParams, 
                      initialReferencePrice: Number(e.target.value)
                    })}
                    fullWidth
                    margin="normal"
                  />
                  
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="α (参照価格更新率)"
                        type="number"
                        inputProps={{ step: 0.1, min: 0, max: 1 }}
                        value={prospectParams.alpha}
                        onChange={(e) => setProspectParams({
                          ...prospectParams, 
                          alpha: Number(e.target.value)
                        })}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="期間数"
                        type="number"
                        value={prospectParams.periods}
                        onChange={(e) => setProspectParams({
                          ...prospectParams, 
                          periods: Number(e.target.value)
                        })}
                        fullWidth
                        margin="normal"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    実行
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleProspectPricing}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <ProspectIcon />}
                    fullWidth
                    sx={{ mt: 2 }}
                  >
                    {loading ? '分析実行中...' : 'プロスペクト理論分析'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {prospectResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>結果</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        ¥{prospectResults.total_revenue?.toLocaleString() || 0}
                      </Typography>
                      <Typography variant="body2">総収益</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {prospectResults.optimal_prices?.length || 0}
                      </Typography>
                      <Typography variant="body2">期間数</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        ¥{Math.round((prospectResults.total_revenue || 0) / (prospectResults.optimal_prices?.length || 1)).toLocaleString()}
                      </Typography>
                      <Typography variant="body2">平均期間収益</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </TabPanel>
        
        <TabPanel value={currentTab} index={6}>
          <Typography variant="h6" gutterBottom>
            データ管理 - インポート/エクスポート
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    CSVデータアップロード
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    需要、収益、容量、使用関係マトリックスのCSVファイルアップロード
                  </Typography>
                  
                  <Box sx={{ mb: 2, p: 1, bgcolor: 'info.light', borderRadius: 1 }}>
                    <Typography variant="body2" color="info.contrastText">
                      サンプルファイル: 
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('demand')}
                        sx={{ mx: 1 }}
                      >
                        需要データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('revenue')}
                        sx={{ mx: 1 }}
                      >
                        収益データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('capacity')}
                        sx={{ mx: 1 }}
                      >
                        容量データ
                      </Link>
                      <Link 
                        component="button" 
                        variant="body2" 
                        onClick={() => handleDownloadSample('usage_matrix')}
                        sx={{ mx: 1 }}
                      >
                        使用関係
                      </Link>
                    </Typography>
                  </Box>
                  
                  <FormControl fullWidth margin="normal">
                    <InputLabel>データタイプ</InputLabel>
                    <Select
                      value={uploadDataType}
                      onChange={(e) => setUploadDataType(e.target.value)}
                      label="データタイプ"
                    >
                      <MenuItem value="demand">需要データ</MenuItem>
                      <MenuItem value="revenue">収益データ</MenuItem>
                      <MenuItem value="capacity">容量データ</MenuItem>
                      <MenuItem value="usage_matrix">使用関係マトリックス</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                    id="csv-upload"
                  />
                  <label htmlFor="csv-upload">
                    <Button
                      variant="outlined"
                      component="span"
                      disabled={loading}
                      fullWidth
                      sx={{ mt: 1, mb: 1 }}
                    >
                      CSVファイル選択
                    </Button>
                  </label>
                  
                  {selectedFile && (
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      選択ファイル: {selectedFile.name}
                    </Typography>
                  )}
                  
                  <Button
                    variant="contained"
                    onClick={handleCSVUpload}
                    disabled={loading || !selectedFile || !uploadDataType}
                    fullWidth
                    startIcon={loading ? <CircularProgress size={20} /> : null}
                  >
                    {loading ? 'アップロード中...' : 'CSVアップロード'}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            
            {/* Export Functions */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    結果エクスポート
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    分析結果とデータのエクスポート
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={!dynamicPricingResults}
                        fullWidth
                        sx={{ mb: 1 }}
                      >
                        動的価格結果エクスポート
                      </Button>
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={!valueFunctionResults}
                        fullWidth
                        sx={{ mb: 1 }}
                      >
                        価値関数結果エクスポート
                      </Button>
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={!revenueOptimizationResults}
                        fullWidth
                        sx={{ mb: 1 }}
                      >
                        収益最適化結果エクスポート
                      </Button>
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={!controlPolicyResults}
                        fullWidth
                        sx={{ mb: 1 }}
                      >
                        制御方策結果エクスポート
                      </Button>
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                        disabled={!prospectResults}
                        fullWidth
                      >
                        プロスペクト理論結果エクスポート
                      </Button>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
          
          {/* Upload Results Display */}
          {uploadResults && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>アップロード結果</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {uploadResults.row_count}
                      </Typography>
                      <Typography variant="body2">データ行数</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="secondary">
                        {uploadResults.column_count}
                      </Typography>
                      <Typography variant="body2">列数</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="info.main">
                        {uploadResults.data_type}
                      </Typography>
                      <Typography variant="body2">データタイプ</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color={uploadResults.validation_status === 'success' ? 'success.main' : 'error.main'}>
                        {uploadResults.validation_status === 'success' ? '成功' : 'エラー'}
                      </Typography>
                      <Typography variant="body2">検証状態</Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                {uploadResults.error_messages && uploadResults.error_messages.length > 0 && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      {uploadResults.error_messages.join(', ')}
                    </Typography>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* Uploaded Data Status */}
          {Object.keys(uploadedData).length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>アップロード済みデータ</Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>データタイプ</TableCell>
                        <TableCell>要素数</TableCell>
                        <TableCell>状態</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(uploadedData).map(([dataType, data]) => (
                        <TableRow key={dataType}>
                          <TableCell>{dataType}</TableCell>
                          <TableCell>{Object.keys(data as object).length}</TableCell>
                          <TableCell>
                            <Typography color="success.main">✓ 完了</Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default RMAnalysis;