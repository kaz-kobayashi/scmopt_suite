import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  FormControlLabel,
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { Upload as CloudUploadIcon, Download as DownloadIcon } from '@mui/icons-material';
import apiClient, { ApiService, CO2Result, DistanceMatrixResult } from '../services/apiClient';
import { CO2EmissionsChart } from './DataVisualization';
import RouteMap from './RouteMap';
import AdvancedRoutingVisualization from './AdvancedRoutingVisualization';

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
      id={`routing-tabpanel-${index}`}
      aria-labelledby={`routing-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// Helper function to parse CSV content
const parseCSVContent = (csvText: string): any[] => {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  const locations: any[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const location: any = {};
    headers.forEach((header, index) => {
      const value = values[index]?.trim();
      if (header === 'lat' || header === 'lon' || header === 'demand') {
        location[header] = parseFloat(value) || 0;
      } else {
        location[header] = value;
      }
    });
    locations.push(location);
  }
  return locations;
};

const Routing: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // CO2 Calculation State
  const [co2Params, setCo2Params] = useState({
    capacity: 10,
    rate: 0.8,
    diesel: true,
  });
  const [co2Result, setCo2Result] = useState<CO2Result | null>(null);

  // Distance Matrix State
  const [locationFile, setLocationFile] = useState<File | null>(null);
  const [distanceResult, setDistanceResult] = useState<DistanceMatrixResult | null>(null);

  // Route Optimization State
  const [routeFile, setRouteFile] = useState<File | null>(null);
  const [vehicleCapacity, setVehicleCapacity] = useState(1000);
  const [maxRoutes, setMaxRoutes] = useState(4);
  const [maxRuntime, setMaxRuntime] = useState(30);
  const [depotName, setDepotName] = useState('Tokyo_DC');
  const [routeResult, setRouteResult] = useState<any>(null);
  const [routeLocations, setRouteLocations] = useState<any[]>([]);

  // Advanced VRP State
  const [vrpFile, setVrpFile] = useState<File | null>(null);
  const [vrpCapacity, setVrpCapacity] = useState(1000);
  const [vrpMaxRoutes, setVrpMaxRoutes] = useState(4);
  const [vrpDepotName, setVrpDepotName] = useState('Tokyo_DC');
  const [vrpMaxRuntime, setVrpMaxRuntime] = useState(30);
  const [vrpResult, setVrpResult] = useState<any>(null);
  const [vrpLocations, setVrpLocations] = useState<any[]>([]);

  // Delivery Schedule State
  const [scheduleFile, setScheduleFile] = useState<File | null>(null);
  const [workingStart, setWorkingStart] = useState(8);
  const [workingEnd, setWorkingEnd] = useState(18);
  const [serviceTime, setServiceTime] = useState(30);
  const [scheduleResult, setScheduleResult] = useState<any>(null);

  // Emissions Analysis State
  const [emissionsParams, setEmissionsParams] = useState({
    distanceKm: 100,
    capacityKg: 1000,
    loadingRate: 0.7,
    fuelType: 'gasoline'
  });
  const [emissionsResult, setEmissionsResult] = useState<any>(null);

  // VRPLIB Benchmark State
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);
  const [benchmarkMaxRuntime, setBenchmarkMaxRuntime] = useState(60);
  const [benchmarkMaxIterations, setBenchmarkMaxIterations] = useState(10000);
  const [benchmarkSeed, setBenchmarkSeed] = useState(42);
  const [benchmarkResult, setBenchmarkResult] = useState<any>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError(null);
  };

  const handleCO2Calculation = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.calculateCO2(
        co2Params.capacity,
        co2Params.rate,
        co2Params.diesel
      );
      setCo2Result(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to calculate CO2 emissions');
    } finally {
      setLoading(false);
    }
  };

  const handleDistanceMatrix = async () => {
    if (!locationFile) {
      setError('Please select a location file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.generateDistanceMatrix(locationFile);
      setDistanceResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate distance matrix');
    } finally {
      setLoading(false);
    }
  };

  const handleLocationFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setLocationFile(selectedFile);
      setError(null);
    }
  };

  const handleRouteOptimization = async () => {
    if (!routeFile) {
      setError('まずルートファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.optimizeRoutes(routeFile, vehicleCapacity, maxRoutes, depotName, maxRuntime);
      setRouteResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ルート最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleAdvancedVRP = async () => {
    if (!vrpFile) {
      setError('Please select a VRP file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.optimizeAdvancedVRP(vrpFile, vrpCapacity, vrpMaxRoutes, vrpDepotName, vrpMaxRuntime);
      setVrpResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to perform advanced VRP');
    } finally {
      setLoading(false);
    }
  };

  const handleDeliverySchedule = async () => {
    if (!scheduleFile) {
      setError('Please select a schedule file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.createDeliverySchedule(scheduleFile, workingStart, workingEnd, serviceTime);
      setScheduleResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create delivery schedule');
    } finally {
      setLoading(false);
    }
  };

  const handleEmissionsAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.analyzeRouteEmissions(
        emissionsParams.distanceKm,
        emissionsParams.capacityKg,
        emissionsParams.loadingRate,
        emissionsParams.fuelType
      );
      setEmissionsResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze emissions');
    } finally {
      setLoading(false);
    }
  };

  const handleBenchmarkSolve = async () => {
    if (!benchmarkFile) {
      setError('VRPLIBファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Use fetch directly as fallback
      const formData = new FormData();
      formData.append('file', benchmarkFile);
      formData.append('max_runtime', benchmarkMaxRuntime.toString());
      formData.append('max_iterations', benchmarkMaxIterations.toString());
      formData.append('seed', benchmarkSeed.toString());

      const response = await fetch('http://127.0.0.1:8000/api/routing/vrplib-solve', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout((benchmarkMaxRuntime + 20) * 1000)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'VRPLIBベンチマーク解決に失敗しました');
      }

      const result = await response.json();
      setBenchmarkResult(result);
    } catch (err: any) {
      setError(err.message || 'VRPLIBベンチマーク解決に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleBenchmarkSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Load X-n439-k37.vrp benchmark
      const response = await fetch('/sample_data/X-n439-k37.vrp');
      if (!response.ok) {
        throw new Error('Failed to fetch benchmark data');
      }
      const vrpText = await response.text();
      const blob = new Blob([vrpText], { type: 'text/plain' });
      const sampleFile = new File([blob], 'X-n439-k37.vrp', { type: 'text/plain' });
      
      setBenchmarkFile(sampleFile);
      
      // Automatically run the benchmark using fetch directly
      const formData = new FormData();
      formData.append('file', sampleFile);
      formData.append('max_runtime', benchmarkMaxRuntime.toString());
      formData.append('max_iterations', benchmarkMaxIterations.toString());
      formData.append('seed', benchmarkSeed.toString());

      const apiResponse = await fetch('http://127.0.0.1:8000/api/routing/vrplib-solve', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout((benchmarkMaxRuntime + 20) * 1000)
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.detail || 'VRPLIBベンチマーク解決に失敗しました');
      }

      const result = await apiResponse.json();
      setBenchmarkResult(result);
    } catch (err: any) {
      setError(err.message || 'ベンチマークデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRouteSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/locations_sample.csv');
      if (!response.ok) {
        throw new Error('Failed to fetch sample data');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'locations_sample.csv', { type: 'text/csv' });
      
      setRouteFile(sampleFile);
      
      // Parse locations
      const locations = parseCSVContent(csvText);
      setRouteLocations(locations);
      
      // Optionally, run optimization automatically
      const result = await ApiService.optimizeRoutes(sampleFile, vehicleCapacity, maxRoutes, depotName, maxRuntime);
      setRouteResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDistanceMatrixSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/locations_sample.csv');
      if (!response.ok) {
        throw new Error('Failed to fetch sample data');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'locations_sample.csv', { type: 'text/csv' });
      
      setLocationFile(sampleFile);
      
      // Automatically generate distance matrix
      const result = await ApiService.generateDistanceMatrix(sampleFile);
      setDistanceResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleAdvancedVRPSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/locations_sample.csv');
      if (!response.ok) {
        throw new Error('Failed to fetch sample data');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'locations_sample.csv', { type: 'text/csv' });
      
      setVrpFile(sampleFile);
      
      // Parse locations
      const locations = parseCSVContent(csvText);
      setVrpLocations(locations);
      
      // Run advanced VRP automatically
      const result = await ApiService.optimizeAdvancedVRP(sampleFile, vrpCapacity, vrpMaxRoutes, vrpDepotName, vrpMaxRuntime);
      setVrpResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDeliveryScheduleSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/orders_sample.csv');
      if (!response.ok) {
        throw new Error('Failed to fetch sample data');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'orders_sample.csv', { type: 'text/csv' });
      
      setScheduleFile(sampleFile);
      
      // Create delivery schedule automatically
      const result = await ApiService.createDeliverySchedule(sampleFile, workingStart, workingEnd, serviceTime);
      setScheduleResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  // Sample data download functions
  const downloadSampleData = async (filename: string, displayName: string) => {
    try {
      const response = await fetch(`/sample_data/${filename}`);
      if (!response.ok) {
        throw new Error('Failed to fetch sample data');
      }
      const text = await response.text();
      const blob = new Blob([text], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = displayName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Failed to download ${displayName}`);
    }
  };

  const renderCO2Calculator = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CO2排出量計算パラメータ
              </Typography>
              
              <TextField
                fullWidth
                label="車両積載量（トン）"
                type="number"
                value={co2Params.capacity}
                onChange={(e) => setCo2Params({ ...co2Params, capacity: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, step: 0.1 }}
              />

              <TextField
                fullWidth
                label="積載率（0-1）"
                type="number"
                value={co2Params.rate}
                onChange={(e) => setCo2Params({ ...co2Params, rate: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, max: 1, step: 0.1 }}
                helperText="車両積載量に対する利用割合"
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={co2Params.diesel}
                    onChange={(e) => setCo2Params({ ...co2Params, diesel: e.target.checked })}
                  />
                }
                label="ディーゼル車両"
                sx={{ mb: 2 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleCO2Calculation}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'CO2排出量計算'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {co2Result && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  排出量計算結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  燃料種別: {co2Result.emissions_calculation.fuel_type}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  燃料消費量: {co2Result.emissions_calculation.fuel_consumption_L_per_ton_km.toFixed(3)} L/トン・km
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  CO2排出量: {co2Result.emissions_calculation.co2_emissions_g_per_ton_km.toFixed(2)} g/トン・km
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  効率スコア: {co2Result.optimization_suggestions.efficiency_score}/100
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  年間燃料消費量（推定）: {co2Result.annual_estimates.estimated_annual_fuel_consumption_L.toLocaleString()} L
                </Typography>
                <Typography variant="body2">
                  年間CO2排出量（推定）: {co2Result.annual_estimates.estimated_annual_co2_emissions_kg.toLocaleString()} kg
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {co2Result && (
          <Grid item xs={12}>
            <CO2EmissionsChart data={co2Result} />
          </Grid>
        )}
        
        {/* Add Advanced CO2 Visualization */}
        {co2Result && (
          <Grid item xs={12}>
            <AdvancedRoutingVisualization
              co2Result={co2Result}
              type="co2"
            />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderDistanceMatrix = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                距離行列生成
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="location-file-input"
                  type="file"
                  onChange={handleLocationFileChange}
                />
                <label htmlFor="location-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    位置データをアップロード (CSV)
                  </Button>
                </label>
                {locationFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {locationFile.name}
                  </Typography>
                )}
              </Box>

              <Alert severity="info" sx={{ mb: 2 }}>
                CSVファイルには列が必要: 'name', 'lat', 'lon'
              </Alert>

              <Button
                fullWidth
                variant="contained"
                onClick={handleDistanceMatrix}
                disabled={loading || !locationFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '距離行列を生成'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleDistanceMatrixSampleData}
                disabled={loading}
                sx={{ mb: 1 }}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => downloadSampleData('locations_sample.csv', 'locations_sample.csv')}
                disabled={loading}
              >
                サンプルデータをダウンロード
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {distanceResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  距離行列結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  地点数: {distanceResult.matrix_size}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  距離単位: {distanceResult.units.distance}
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  時間単位: {distanceResult.units.duration}
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  地点一覧:
                </Typography>
                <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                  {distanceResult.locations.map((location, index) => (
                    <Typography key={index} variant="body2" sx={{ mb: 0.5 }}>
                      • {location}
                    </Typography>
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {distanceResult && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  距離行列プレビュー
                </Typography>
                <Box sx={{ overflow: 'auto', maxHeight: 400 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f5f5f5' }}>
                          地点
                        </th>
                        {distanceResult.locations.slice(0, 5).map((loc, index) => (
                          <th key={index} style={{ border: '1px solid #ccc', padding: '8px', backgroundColor: '#f5f5f5' }}>
                            {loc}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {distanceResult.locations.slice(0, 5).map((fromLoc, rowIndex) => (
                        <tr key={rowIndex}>
                          <td style={{ border: '1px solid #ccc', padding: '8px', fontWeight: 'bold' }}>
                            {fromLoc}
                          </td>
                          {distanceResult.locations.slice(0, 5).map((toLoc, colIndex) => (
                            <td key={colIndex} style={{ border: '1px solid #ccc', padding: '8px' }}>
                              {distanceResult.distance_matrix[fromLoc][toLoc].toFixed(1)} km
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
                {distanceResult.locations.length > 5 && (
                  <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                    最初の5x5行列を表示。完全な行列は{distanceResult.matrix_size}地点。
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
        
        {/* Add Advanced Distance Matrix Visualization */}
        {distanceResult && (
          <Grid item xs={12}>
            <AdvancedRoutingVisualization
              distanceMatrixResult={distanceResult}
              type="distance-matrix"
            />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderRouteOptimization = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                基本配送ルート最適化
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="route-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setRouteFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="route-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    位置データをアップロード (CSV)
                  </Button>
                </label>
                {routeFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {routeFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="車両積載量 (kg)"
                type="number"
                value={vehicleCapacity}
                onChange={(e) => setVehicleCapacity(parseFloat(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, step: 0.1 }}
                helperText="各車両の最大積載量（キログラム）"
              />

              <TextField
                fullWidth
                label="最大ルート数"
                type="number"
                value={maxRoutes}
                onChange={(e) => setMaxRoutes(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="デポ名"
                value={depotName}
                onChange={(e) => setDepotName(e.target.value)}
                sx={{ mb: 2 }}
                helperText="データ内のデポ（配送センター）の名前"
              />

              <TextField
                fullWidth
                label="最適化時間制限（秒）"
                type="number"
                value={maxRuntime}
                onChange={(e) => setMaxRuntime(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 5, max: 300, step: 5 }}
                helperText="PyVRP最適化アルゴリズムの実行時間制限"
              />

              <Alert severity="info" sx={{ mb: 2 }}>
                CSVファイル必要列: name, lat, lon, demand（オプション）
              </Alert>

              <Button
                fullWidth
                variant="contained"
                onClick={handleRouteOptimization}
                disabled={loading || !routeFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ルートを最適化'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleRouteSampleData}
                disabled={loading}
                sx={{ mb: 1 }}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => downloadSampleData('locations_sample.csv', 'locations_sample.csv')}
                disabled={loading}
              >
                サンプルデータをダウンロード
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                サンプル: 東京DC + 関東地方19店舗（東京・神奈川・埼玉・千葉）
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {routeResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ルート最適化結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総ルート数: {routeResult.summary?.total_routes || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総距離: {routeResult.summary?.total_distance_km?.toFixed(2) || 0} km
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総積載量: {routeResult.summary?.total_load?.toFixed(2) || 0} kg
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  デポ: {routeResult.summary?.depot_location || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  訪問店舗数: {routeResult.summary?.locations_served || 0}
                </Typography>
                {routeResult.summary?.avg_capacity_utilization && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    平均積載率: {(routeResult.summary.avg_capacity_utilization * 100).toFixed(1)}%
                  </Typography>
                )}
                {routeResult.optimization_info?.runtime_seconds && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    最適化時間: {routeResult.optimization_info.runtime_seconds.toFixed(2)}秒
                  </Typography>
                )}
                {routeResult.optimization_info?.algorithm && (
                  <Typography variant="body2">
                    アルゴリズム: {routeResult.optimization_info.algorithm}
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {routeResult && routeResult.optimized_routes && (
          <>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ルート詳細
                  </Typography>
                  {routeResult.optimized_routes.map((route: any, index: number) => (
                    <Box key={index} sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        ルート {route.route_id}: {route.total_distance?.toFixed(2)} km
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        積載量: {route.total_load?.toFixed(2)} kg | 訪問地点: {route.locations?.join(' → ') || 'N/A'}
                      </Typography>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ルート地図
                  </Typography>
                  {routeLocations.length > 0 && (
                    <RouteMap
                      locations={routeLocations}
                      routes={routeResult.optimized_routes}
                      depot={depotName}
                    />
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            {/* Add Advanced Route Optimization Visualization */}
            <Grid item xs={12}>
              <AdvancedRoutingVisualization
                routeOptimizationResult={{
                  routes: routeResult.optimized_routes || [],
                  summary: routeResult.summary || {},
                  optimization_stats: {
                    best_cost: routeResult.optimization_info?.best_cost || 0,
                    iterations: routeResult.optimization_info?.iterations || 0,
                    runtime_seconds: routeResult.optimization_info?.runtime_seconds || 0,
                    algorithm: routeResult.optimization_info?.algorithm || 'Unknown',
                    convergence: routeResult.optimization_info?.convergence || false
                  }
                }}
                type="route-optimization"
              />
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );

  const renderAdvancedVRP = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                高度な車両配送問題（VRP）
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="vrp-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setVrpFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="vrp-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    位置データをアップロード (CSV)
                  </Button>
                </label>
                {vrpFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {vrpFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="車両積載量 (kg)"
                type="number"
                value={vrpCapacity}
                onChange={(e) => setVrpCapacity(parseFloat(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, step: 0.1 }}
                helperText="各車両の最大積載量（キログラム）"
              />

              <TextField
                fullWidth
                label="最大ルート数"
                type="number"
                value={vrpMaxRoutes}
                onChange={(e) => setVrpMaxRoutes(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="デポ名"
                value={vrpDepotName}
                onChange={(e) => setVrpDepotName(e.target.value)}
                sx={{ mb: 2 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleAdvancedVRP}
                disabled={loading || !vrpFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '高度VRP実行'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleAdvancedVRPSampleData}
                disabled={loading}
                sx={{ mb: 1 }}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => downloadSampleData('locations_sample.csv', 'locations_sample.csv')}
                disabled={loading}
              >
                サンプルデータをダウンロード
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {vrpResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  高度VRP結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  最適化手法: {vrpResult.optimization_method || '高度VRP'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  ルート効率: {vrpResult.performance_metrics?.route_efficiency || 'N/A'}%
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  距離最適化: {vrpResult.performance_metrics?.distance_optimization || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  サービスカバー率: {vrpResult.performance_metrics?.service_coverage || 'N/A'}
                </Typography>
                {vrpResult.vrp_result?.summary && (
                  <>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      総距離: {vrpResult.vrp_result.summary.total_distance?.toFixed(1) || 'N/A'} km
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      総ルート数: {vrpResult.vrp_result.summary.total_routes || 'N/A'}
                    </Typography>
                    <Typography variant="body2">
                      平均積載率: {(vrpResult.vrp_result.summary.avg_capacity_utilization * 100)?.toFixed(1) || 'N/A'}%
                    </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {vrpResult && vrpResult.vrp_result?.routes && vrpLocations.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Route Map
                </Typography>
                <RouteMap
                  locations={vrpLocations}
                  routes={vrpResult.vrp_result.routes}
                  depot={vrpDepotName}
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderDeliverySchedule = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                配送スケジュール最適化
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="schedule-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setScheduleFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="schedule-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    注文データをアップロード (CSV)
                  </Button>
                </label>
                {scheduleFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {scheduleFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="作業開始時刻（24時間形式）"
                type="number"
                value={workingStart}
                onChange={(e) => setWorkingStart(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 0, max: 23, step: 1 }}
              />

              <TextField
                fullWidth
                label="作業終了時刻（24時間形式）"
                type="number"
                value={workingEnd}
                onChange={(e) => setWorkingEnd(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, max: 24, step: 1 }}
              />

              <TextField
                fullWidth
                label="配送あたりのサービス時間（分）"
                type="number"
                value={serviceTime}
                onChange={(e) => setServiceTime(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleDeliverySchedule}
                disabled={loading || !scheduleFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'スケジュール最適化'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleDeliveryScheduleSampleData}
                disabled={loading}
                sx={{ mb: 1 }}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => downloadSampleData('orders_sample.csv', 'orders_sample.csv')}
                disabled={loading}
              >
                サンプルデータをダウンロード
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                サンプル: 関東地方の優先度付き19件の配送注文
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {scheduleResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  スケジュール最適化結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  スケジュール効率: {scheduleResult.optimization_summary?.schedule_efficiency || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  作業時間利用率: {scheduleResult.optimization_summary?.working_hours_utilization || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  推奨事項: {scheduleResult.optimization_summary?.recommendations || 'N/A'}
                </Typography>
                {scheduleResult.delivery_schedule?.summary && (
                  <>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      総配送数: {scheduleResult.delivery_schedule.summary.total_deliveries || 'N/A'}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      サービス時間: {scheduleResult.delivery_schedule.summary.total_service_time_hours?.toFixed(1) || 'N/A'} 時間
                    </Typography>
                    <Typography variant="body2">
                      移動時間: {scheduleResult.delivery_schedule.summary.total_travel_time_hours?.toFixed(1) || 'N/A'} 時間
                    </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );

  const renderVRPLIBBenchmark = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                VRPLIBベンチマーク問題
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                標準的なVRP（車両配送問題）ベンチマークインスタンスを解決します。
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".vrp"
                  style={{ display: 'none' }}
                  id="benchmark-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setBenchmarkFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="benchmark-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    VRPLIBファイルをアップロード (.vrp)
                  </Button>
                </label>
                {benchmarkFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {benchmarkFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="最適化時間制限（秒）"
                type="number"
                value={benchmarkMaxRuntime}
                onChange={(e) => setBenchmarkMaxRuntime(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 10, max: 600, step: 10 }}
                helperText="PyVRP最適化の時間制限（推奨：60-300秒）"
              />

              <TextField
                fullWidth
                label="最大イテレーション数"
                type="number"
                value={benchmarkMaxIterations}
                onChange={(e) => setBenchmarkMaxIterations(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1000, max: 100000, step: 1000 }}
                helperText="最適化の最大反復回数"
              />

              <TextField
                fullWidth
                label="乱数シード"
                type="number"
                value={benchmarkSeed}
                onChange={(e) => setBenchmarkSeed(parseInt(e.target.value))}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
                helperText="再現可能な結果のための乱数シード"
              />

              <Alert severity="info" sx={{ mb: 2 }}>
                VRPLIBファイル形式: NAME, TYPE, DIMENSION, EDGE_WEIGHT_TYPE, CAPACITY, NODE_COORD_SECTION, DEMAND_SECTION
              </Alert>

              <Button
                fullWidth
                variant="contained"
                onClick={handleBenchmarkSolve}
                disabled={loading || !benchmarkFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ベンチマーク実行'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleBenchmarkSampleData}
                disabled={loading}
                sx={{ mb: 1 }}
              >
                {loading ? <CircularProgress size={24} /> : 'X-n439-k37サンプル実行'}
              </Button>

              <Button
                fullWidth
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={() => downloadSampleData('X-n439-k37.vrp', 'X-n439-k37.vrp')}
                disabled={loading}
              >
                ベンチマークファイルをダウンロード
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                サンプル: X-n439-k37（438顧客、37車両、目標値: 36737）
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {benchmarkResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ベンチマーク結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  最適化手法: {benchmarkResult.optimization_method || 'PyVRP Standard'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  目的関数値: <strong>{benchmarkResult.performance_metrics?.objective_value?.toFixed(0) || 'N/A'}</strong>
                </Typography>
                {benchmarkResult.vrplib_result?.optimization_stats?.instance_info?.instance_name === 'X-n439-k37' && (
                  <Typography variant="body2" sx={{ mb: 1, color: benchmarkResult.performance_metrics?.objective_value < 37000 ? 'success.main' : 'warning.main' }}>
                    目標値36737からの差: {benchmarkResult.performance_metrics?.objective_value ? 
                      (benchmarkResult.performance_metrics.objective_value - 36737 > 0 ? '+' : '') + 
                      (benchmarkResult.performance_metrics.objective_value - 36737).toFixed(0) : 'N/A'}
                  </Typography>
                )}
                <Typography variant="body2" sx={{ mb: 1 }}>
                  生成ルート数: {benchmarkResult.performance_metrics?.routes_generated || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  サービス顧客数: {benchmarkResult.performance_metrics?.customers_served || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  最適化時間: {benchmarkResult.performance_metrics?.optimization_runtime || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  イテレーション数: {benchmarkResult.performance_metrics?.iterations || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  収束状況: {benchmarkResult.performance_metrics?.convergence ? '✓ 収束' : '× 未収束'}
                </Typography>

                {benchmarkResult.vrplib_result?.optimization_stats?.instance_info && (
                  <Box sx={{ mt: 2, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      インスタンス情報:
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 0.5 }}>
                      名前: {benchmarkResult.vrplib_result.optimization_stats.instance_info.instance_name || 'Unknown'}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 0.5 }}>
                      顧客数: {benchmarkResult.vrplib_result.optimization_stats.instance_info.num_clients || 'N/A'}
                    </Typography>
                    <Typography variant="body2">
                      デポ数: {benchmarkResult.vrplib_result.optimization_stats.instance_info.num_depots || 'N/A'}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {benchmarkResult && benchmarkResult.vrplib_result?.routes && (
          <>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ルート詳細
                  </Typography>
                  <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                    {benchmarkResult.vrplib_result.routes.slice(0, 5).map((route: any, index: number) => (
                      <Box key={index} sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          ルート {route.route_id}: 距離 {route.total_distance?.toFixed(2) || 'N/A'} km
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          需要: {route.total_demand?.toFixed(2) || 'N/A'} | 
                          積載率: {route.capacity_utilization ? (route.capacity_utilization * 100).toFixed(1) + '%' : 'N/A'}
                        </Typography>
                        <Typography variant="body2" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
                          順序: {route.locations?.slice(0, 10).join(' → ') || 'N/A'}
                          {route.locations?.length > 10 && '...'}
                        </Typography>
                      </Box>
                    ))}
                    {benchmarkResult.vrplib_result.routes.length > 5 && (
                      <Typography variant="caption" color="text.secondary">
                        最初の5ルートを表示。総ルート数: {benchmarkResult.vrplib_result.routes.length}
                      </Typography>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            {benchmarkResult.vrplib_result.map_locations && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      関東圏マップ表示
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                      VRPLIBベンチマーク座標を関東圏（東京・神奈川・埼玉・千葉）にスケーリングして表示
                    </Typography>
                    <RouteMap
                      locations={benchmarkResult.vrplib_result.map_locations}
                      routes={benchmarkResult.vrplib_result.routes}
                      depot="Tokyo_DC"
                    />
                  </CardContent>
                </Card>
              </Grid>
            )}
          </>
        )}
      </Grid>
    </Box>
  );

  const renderEmissionsAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ルート排出量分析
              </Typography>
              
              <TextField
                fullWidth
                label="ルート距離 (km)"
                type="number"
                value={emissionsParams.distanceKm}
                onChange={(e) => setEmissionsParams({ ...emissionsParams, distanceKm: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, step: 0.1 }}
              />

              <TextField
                fullWidth
                label="車両積載量 (kg)"
                type="number"
                value={emissionsParams.capacityKg}
                onChange={(e) => setEmissionsParams({ ...emissionsParams, capacityKg: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="積載率 (0-1)"
                type="number"
                value={emissionsParams.loadingRate}
                onChange={(e) => setEmissionsParams({ ...emissionsParams, loadingRate: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.1, max: 1, step: 0.1 }}
              />

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>燃料タイプ</InputLabel>
                <Select
                  value={emissionsParams.fuelType}
                  label="燃料タイプ"
                  onChange={(e) => setEmissionsParams({ ...emissionsParams, fuelType: e.target.value })}
                >
                  <MenuItem value="gasoline">ガソリン</MenuItem>
                  <MenuItem value="diesel">ディーゼル</MenuItem>
                </Select>
              </FormControl>

              <Button
                fullWidth
                variant="contained"
                onClick={handleEmissionsAnalysis}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : '排出量分析'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {emissionsResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  排出量分析結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総CO2排出量: {emissionsResult.emissions_analysis?.co2_emissions_kg?.toFixed(2) || 'N/A'} kg
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  燃料消費量: {emissionsResult.emissions_analysis?.fuel_consumption_L?.toFixed(2) || 'N/A'} L
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  持続可能性スコア: {emissionsResult.sustainability?.sustainability_score || 'N/A'}/100
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  kmあたり排出量: {emissionsResult.sustainability?.emissions_per_km || 'N/A'} kg/km
                </Typography>
                <Typography variant="body2">
                  効率評価: {emissionsResult.sustainability?.efficiency_rating || 'N/A'}
                </Typography>
                
                {emissionsResult.recommendations && emissionsResult.recommendations.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      推奨事項:
                    </Typography>
                    {emissionsResult.recommendations.map((rec: string, index: number) => (
                      <Typography key={index} variant="body2" sx={{ mb: 0.5 }}>
                        • {rec}
                      </Typography>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        配送・輸送最適化
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="routing tabs">
          <Tab label="CO2計算" />
          <Tab label="距離行列" />
          <Tab label="配送ルート最適化" />
          <Tab label="高度なVRP" />
          <Tab label="配送スケジュール" />
          <Tab label="排出量分析" />
          <Tab label="VRPLIBベンチマーク" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        {renderCO2Calculator()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        {renderDistanceMatrix()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        {renderRouteOptimization()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={3}>
        {renderAdvancedVRP()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={4}>
        {renderDeliverySchedule()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={5}>
        {renderEmissionsAnalysis()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={6}>
        {renderVRPLIBBenchmark()}
      </TabPanel>
    </Box>
  );
};

export default Routing;