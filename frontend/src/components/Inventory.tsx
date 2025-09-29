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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import apiClient, { ApiService, EOQResult, SimulationResult } from '../services/apiClient';
import { InventorySimulationChart } from './DataVisualization';
import MultiEchelonVisualization from './MultiEchelonVisualization';
import EOQVisualization from './EOQVisualization';
import InventorySimulationVisualization from './InventorySimulationVisualization';

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
      id={`inventory-tabpanel-${index}`}
      aria-labelledby={`inventory-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Inventory: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // EOQ State
  const [eoqParams, setEoqParams] = useState({
    K: 100,
    d: 1000,
    h: 0.2,
    b: 0.1,
    r: 0.15,
    c: 50,
    theta: 0.95,
  });
  const [eoqResult, setEoqResult] = useState<EOQResult | null>(null);

  // Simulation State
  const [simParams, setSimParams] = useState({
    n_samples: 1000,
    n_periods: 365,
    mu: 50,
    sigma: 10,
    LT: 1,
    Q: 100,
    R: 120,
    b: 10,
    h: 1,
    fc: 100,
  });
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);

  // Multi-Echelon State
  const [multiEchelonParams, setMultiEchelonParams] = useState({
    nPlants: 2,
    nDCs: 3,
    nRetailers: 5,
    holdingCost: 1.0,
    orderingCost: 50,
    transportCost: 2.0,
    demandMean: 100,
    demandStd: 20
  });
  const [multiEchelonResult, setMultiEchelonResult] = useState<any>(null);

  // Optimization State
  const [optimizationType, setOptimizationType] = useState('qr');
  const [qrParams, setQrParams] = useState({
    mu: 100,
    sigma: 20,
    LT: 2,
    b: 50,
    h: 5
  });
  const [ssParams, setSsParams] = useState({
    mu: 100,
    sigma: 20,
    LT: 2,
    b: 50,
    h: 5,
    service_level: 0.95
  });
  const [messaParams, setMessaParams] = useState({
    file: null,
    mu: 100,
    sigma: 20,
    LT: 2,
    k: 100,
    h: 5,
    b: 50
  });
  const [newsvendorParams, setNewsvendorParams] = useState({
    purchase_cost: 10,
    selling_price: 20,
    salvage_value: 5,
    demand_params: {
      distribution: 'normal',
      mean: 100,
      std: 20
    }
  });
  const [optimizationResult, setOptimizationResult] = useState<any>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError(null);
  };

  const handleEOQCalculation = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.calculateEOQ(eoqParams);
      setEoqResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'EOQ計算に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleSimulation = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.simulateInventory(simParams);
      setSimResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'シミュレーションの実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleMultiEchelonOptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const networkData = {
        plants: Array.from({length: multiEchelonParams.nPlants}, (_, i) => ({id: i, capacity: 10000})),
        distribution_centers: Array.from({length: multiEchelonParams.nDCs}, (_, i) => ({id: i, capacity: 5000})),
        retailers: Array.from({length: multiEchelonParams.nRetailers}, (_, i) => ({id: i, demand_mean: multiEchelonParams.demandMean, demand_std: multiEchelonParams.demandStd}))
      };

      const demandData = Array.from({length: multiEchelonParams.nRetailers}, () => 
        Array.from({length: 52}, () => multiEchelonParams.demandMean + (Math.random() - 0.5) * multiEchelonParams.demandStd * 2)
      );

      const costParams = {
        holding_cost: multiEchelonParams.holdingCost,
        ordering_cost: multiEchelonParams.orderingCost,
        transport_cost: multiEchelonParams.transportCost
      };

      const result = await ApiService.optimizeMultiEchelon(networkData, demandData, costParams);
      setMultiEchelonResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'マルチエシェロン在庫最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleQROptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('mu', qrParams.mu.toString());
      formData.append('sigma', qrParams.sigma.toString());
      formData.append('LT', qrParams.LT.toString());
      formData.append('b', qrParams.b.toString());
      formData.append('h', qrParams.h.toString());

      const response = await fetch('http://localhost:8000/api/inventory/optimize-qr', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('QR最適化に失敗しました');
      }

      const result = await response.json();
      setOptimizationResult(result);
    } catch (err: any) {
      setError(err.message || 'QR最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleSSOptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('mu', ssParams.mu.toString());
      formData.append('sigma', ssParams.sigma.toString());
      formData.append('LT', ssParams.LT.toString());
      formData.append('b', ssParams.b.toString());
      formData.append('h', ssParams.h.toString());
      formData.append('service_level', ssParams.service_level.toString());

      const response = await fetch('http://localhost:8000/api/inventory/optimize-ss', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('安全在庫最適化に失敗しました');
      }

      const result = await response.json();
      setOptimizationResult(result);
    } catch (err: any) {
      setError(err.message || '安全在庫最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleNewsvendorOptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/inventory/newsvendor-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newsvendorParams)
      });

      if (!response.ok) {
        throw new Error('ニューズベンダーモデル最適化に失敗しました');
      }

      const result = await response.json();
      setOptimizationResult(result);
    } catch (err: any) {
      setError(err.message || 'ニューズベンダーモデル最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleMESSAOptimization = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/inventory/messa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messaParams)
      });

      if (!response.ok) {
        throw new Error('MESSA最適化に失敗しました');
      }

      const result = await response.json();
      setOptimizationResult(result);
    } catch (err: any) {
      setError(err.message || 'MESSA最適化に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimization = () => {
    switch (optimizationType) {
      case 'qr':
        return handleQROptimization();
      case 'ss':
        return handleSSOptimization();
      case 'newsvendor':
        return handleNewsvendorOptimization();
      case 'messa':
        return handleMESSAOptimization();
      default:
        setError('不明な最適化タイプです');
    }
  };

  const renderEOQCalculator = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                EOQパラメータ
              </Typography>
              
              <TextField
                fullWidth
                label="固定発注コスト (K)"
                type="number"
                value={eoqParams.K}
                onChange={(e) => setEoqParams({ ...eoqParams, K: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="需要率 (d)"
                type="number"
                value={eoqParams.d}
                onChange={(e) => setEoqParams({ ...eoqParams, d: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="保管コスト (h)"
                type="number"
                value={eoqParams.h}
                onChange={(e) => setEoqParams({ ...eoqParams, h: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="欠品コスト (b)"
                type="number"
                value={eoqParams.b}
                onChange={(e) => setEoqParams({ ...eoqParams, b: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="利子率 (r)"
                type="number"
                value={eoqParams.r}
                onChange={(e) => setEoqParams({ ...eoqParams, r: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="単価 (c)"
                type="number"
                value={eoqParams.c}
                onChange={(e) => setEoqParams({ ...eoqParams, c: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="サービスレベル (theta)"
                type="number"
                value={eoqParams.theta}
                onChange={(e) => setEoqParams({ ...eoqParams, theta: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleEOQCalculation}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'EOQ計算'}
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

          {eoqResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  EOQ結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  最適発注量: {eoqResult.optimal_order_quantity.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  年間総コスト: ¥{eoqResult.total_annual_cost.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  年間発注コスト: ¥{eoqResult.annual_ordering_cost.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  年間保管コスト: ¥{eoqResult.annual_holding_cost.toFixed(2)}
                </Typography>
                <Typography variant="body2">
                  サイクル時間: {eoqResult.cycle_time_periods.toFixed(2)} 期間
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
        
        {/* Add EOQ Visualization */}
        {eoqResult && (
          <Grid item xs={12}>
            <EOQVisualization result={eoqResult} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderSimulation = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                シミュレーションパラメータ
              </Typography>
              
              <TextField
                fullWidth
                label="サンプル数"
                type="number"
                value={simParams.n_samples}
                onChange={(e) => setSimParams({ ...simParams, n_samples: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="期間数"
                type="number"
                value={simParams.n_periods}
                onChange={(e) => setSimParams({ ...simParams, n_periods: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="平均需要 (μ)"
                type="number"
                value={simParams.mu}
                onChange={(e) => setSimParams({ ...simParams, mu: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="需要標準偏差 (σ)"
                type="number"
                value={simParams.sigma}
                onChange={(e) => setSimParams({ ...simParams, sigma: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="リードタイム (LT)"
                type="number"
                value={simParams.LT}
                onChange={(e) => setSimParams({ ...simParams, LT: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="発注量 (Q)"
                type="number"
                value={simParams.Q}
                onChange={(e) => setSimParams({ ...simParams, Q: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="発注点 (R)"
                type="number"
                value={simParams.R}
                onChange={(e) => setSimParams({ ...simParams, R: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleSimulation}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'シミュレーション実行'}
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

          {simResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  シミュレーション結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  ポリシータイプ: {simResult.simulation_results.policy_type}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  平均コスト: ¥{simResult.simulation_results.average_cost_per_period.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  コスト標準偏差: ¥{simResult.simulation_results.cost_standard_deviation.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  平均在庫レベル: {simResult.simulation_results.average_inventory_level.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {simResult && (
          <Grid item xs={12}>
            <InventorySimulationChart data={simResult} />
          </Grid>
        )}
        
        {/* Add Inventory Simulation Visualization */}
        {simResult && (
          <Grid item xs={12}>
            <InventorySimulationVisualization result={simResult} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderMultiEchelonOptimization = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                マルチエシェロン在庫構成
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                ネットワーク構造:
              </Typography>
              
              <TextField
                fullWidth
                label="工場数"
                type="number"
                value={multiEchelonParams.nPlants}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, nPlants: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="配送センター数"
                type="number"
                value={multiEchelonParams.nDCs}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, nDCs: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="小売店数"
                type="number"
                value={multiEchelonParams.nRetailers}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, nRetailers: parseInt(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                コストパラメータ:
              </Typography>

              <TextField
                fullWidth
                label="単位当たり保管コスト"
                type="number"
                value={multiEchelonParams.holdingCost}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, holdingCost: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.01, step: 0.01 }}
              />

              <TextField
                fullWidth
                label="注文当たり発注コスト"
                type="number"
                value={multiEchelonParams.orderingCost}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, orderingCost: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="単位当たり輸送コスト"
                type="number"
                value={multiEchelonParams.transportCost}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, transportCost: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 0.01, step: 0.01 }}
              />

              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                需要パラメータ:
              </Typography>

              <TextField
                fullWidth
                label="期間当たり平均需要"
                type="number"
                value={multiEchelonParams.demandMean}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, demandMean: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <TextField
                fullWidth
                label="需要標準偏差"
                type="number"
                value={multiEchelonParams.demandStd}
                onChange={(e) => setMultiEchelonParams({ ...multiEchelonParams, demandStd: parseFloat(e.target.value) })}
                sx={{ mb: 2 }}
                inputProps={{ min: 1, step: 1 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleMultiEchelonOptimization}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'マルチエシェロン在庫最適化'}
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

          {multiEchelonResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  マルチエシェロン最適化結果
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  最適化ステータス: 完了
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総コスト見積もり: ¥{multiEchelonResult.summary?.total_cost_estimate?.toFixed(2) || 'N/A'}
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  サービスレベル: {multiEchelonResult.performance_metrics?.average_service_level?.toFixed(1) || 'N/A'}%
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  在庫回転率: {multiEchelonResult.performance_metrics?.inventory_turnover?.toFixed(1) || 'N/A'}
                </Typography>
                
                <Typography variant="body2" sx={{ mb: 2 }}>
                  ネットワーク構成: {multiEchelonResult.summary?.network_configuration || 'N/A'}
                </Typography>

                {multiEchelonResult.recommendations && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      推奨事項:
                    </Typography>
                    {multiEchelonResult.recommendations.map((rec: string, index: number) => (
                      <Typography key={index} variant="body2" sx={{ mb: 0.5, fontSize: '0.875rem' }}>
                        • {rec}
                      </Typography>
                    ))}
                  </Box>
                )}

                {multiEchelonResult.optimization_result?.echelon_policies && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      エシェロンポリシー:
                    </Typography>
                    {Object.entries(multiEchelonResult.optimization_result.echelon_policies).map(([echelon, policy]: [string, any]) => (
                      <Box key={echelon} sx={{ mb: 1, p: 1, backgroundColor: '#fafafa', borderRadius: 1 }}>
                        <Typography variant="caption" sx={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                          {echelon}:
                        </Typography>
                        <Typography variant="caption" sx={{ ml: 1 }}>
                          EOQ: {policy.eoq?.toFixed(0)}, Safety Stock: {policy.safety_stock?.toFixed(0)}, Reorder Point: {policy.reorder_point?.toFixed(0)}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
      
      {/* Add visualization component */}
      {multiEchelonResult && (
        <MultiEchelonVisualization 
          result={{
            total_cost: multiEchelonResult.summary?.total_cost_estimate || 0,
            service_level: (multiEchelonResult.performance_metrics?.average_service_level || 0) / 100,
            total_inventory: multiEchelonResult.performance_metrics?.inventory_turnover || 0,
            network_structure: {
              Plants: multiEchelonParams.nPlants,
              DCs: multiEchelonParams.nDCs,
              Retailers: multiEchelonParams.nRetailers
            },
            recommendations: multiEchelonResult.recommendations || [],
            echelon_policies: {
              Plant: {
                EOQ: multiEchelonResult.optimization_result?.echelon_policies?.plant?.eoq || 448,
                Safety_Stock: multiEchelonResult.optimization_result?.echelon_policies?.plant?.safety_stock || 19,
                Reorder_Point: multiEchelonResult.optimization_result?.echelon_policies?.plant?.reorder_point || 720
              },
              DC: {
                EOQ: multiEchelonResult.optimization_result?.echelon_policies?.dc?.eoq || 200,
                Safety_Stock: multiEchelonResult.optimization_result?.echelon_policies?.dc?.safety_stock || 13,
                Reorder_Point: multiEchelonResult.optimization_result?.echelon_policies?.dc?.reorder_point || 313
              },
              Retail: {
                EOQ: multiEchelonResult.optimization_result?.echelon_policies?.retail?.eoq || 71,
                Safety_Stock: multiEchelonResult.optimization_result?.echelon_policies?.retail?.safety_stock || 6,
                Reorder_Point: multiEchelonResult.optimization_result?.echelon_policies?.retail?.reorder_point || 106
              }
            }
          }}
        />
      )}
    </Box>
  );

  const renderOptimization = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                在庫最適化パラメータ
              </Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>最適化タイプ</InputLabel>
                <Select
                  value={optimizationType}
                  onChange={(e) => setOptimizationType(e.target.value)}
                  label="最適化タイプ"
                >
                  <MenuItem value="qr">Q-R最適化</MenuItem>
                  <MenuItem value="ss">安全在庫最適化</MenuItem>
                  <MenuItem value="newsvendor">ニューズベンダーモデル</MenuItem>
                  <MenuItem value="messa">MESSA最適化</MenuItem>
                </Select>
              </FormControl>

              {optimizationType === 'qr' && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    Q-R最適化パラメータ:
                  </Typography>
                  <TextField
                    fullWidth
                    label="平均需要 (μ)"
                    type="number"
                    value={qrParams.mu}
                    onChange={(e) => setQrParams({ ...qrParams, mu: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="需要標準偏差 (σ)"
                    type="number"
                    value={qrParams.sigma}
                    onChange={(e) => setQrParams({ ...qrParams, sigma: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="リードタイム (LT)"
                    type="number"
                    value={qrParams.LT}
                    onChange={(e) => setQrParams({ ...qrParams, LT: parseInt(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="欠品コスト (b)"
                    type="number"
                    value={qrParams.b}
                    onChange={(e) => setQrParams({ ...qrParams, b: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="保管コスト (h)"
                    type="number"
                    value={qrParams.h}
                    onChange={(e) => setQrParams({ ...qrParams, h: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              {optimizationType === 'ss' && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    安全在庫最適化パラメータ:
                  </Typography>
                  <TextField
                    fullWidth
                    label="平均需要 (μ)"
                    type="number"
                    value={ssParams.mu}
                    onChange={(e) => setSsParams({ ...ssParams, mu: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="需要標準偏差 (σ)"
                    type="number"
                    value={ssParams.sigma}
                    onChange={(e) => setSsParams({ ...ssParams, sigma: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="リードタイム (LT)"
                    type="number"
                    value={ssParams.LT}
                    onChange={(e) => setSsParams({ ...ssParams, LT: parseInt(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="欠品コスト (b)"
                    type="number"
                    value={ssParams.b}
                    onChange={(e) => setSsParams({ ...ssParams, b: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="保管コスト (h)"
                    type="number"
                    value={ssParams.h}
                    onChange={(e) => setSsParams({ ...ssParams, h: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="サービスレベル"
                    type="number"
                    value={ssParams.service_level}
                    onChange={(e) => setSsParams({ ...ssParams, service_level: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                    inputProps={{ min: 0, max: 1, step: 0.01 }}
                  />
                </Box>
              )}

              {optimizationType === 'newsvendor' && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    ニューズベンダーモデルパラメータ:
                  </Typography>
                  <TextField
                    fullWidth
                    label="仕入価格"
                    type="number"
                    value={newsvendorParams.purchase_cost}
                    onChange={(e) => setNewsvendorParams({ ...newsvendorParams, purchase_cost: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="販売価格"
                    type="number"
                    value={newsvendorParams.selling_price}
                    onChange={(e) => setNewsvendorParams({ ...newsvendorParams, selling_price: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="残存価値"
                    type="number"
                    value={newsvendorParams.salvage_value}
                    onChange={(e) => setNewsvendorParams({ ...newsvendorParams, salvage_value: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="需要平均"
                    type="number"
                    value={newsvendorParams.demand_params.mean}
                    onChange={(e) => setNewsvendorParams({ 
                      ...newsvendorParams, 
                      demand_params: { ...newsvendorParams.demand_params, mean: parseFloat(e.target.value) }
                    })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="需要標準偏差"
                    type="number"
                    value={newsvendorParams.demand_params.std}
                    onChange={(e) => setNewsvendorParams({ 
                      ...newsvendorParams, 
                      demand_params: { ...newsvendorParams.demand_params, std: parseFloat(e.target.value) }
                    })}
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              {optimizationType === 'messa' && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    MESSA最適化パラメータ:
                  </Typography>
                  <TextField
                    fullWidth
                    label="平均需要 (μ)"
                    type="number"
                    value={messaParams.mu}
                    onChange={(e) => setMessaParams({ ...messaParams, mu: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="需要標準偏差 (σ)"
                    type="number"
                    value={messaParams.sigma}
                    onChange={(e) => setMessaParams({ ...messaParams, sigma: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="リードタイム (LT)"
                    type="number"
                    value={messaParams.LT}
                    onChange={(e) => setMessaParams({ ...messaParams, LT: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="固定発注コスト (k)"
                    type="number"
                    value={messaParams.k}
                    onChange={(e) => setMessaParams({ ...messaParams, k: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="保管コスト (h)"
                    type="number"
                    value={messaParams.h}
                    onChange={(e) => setMessaParams({ ...messaParams, h: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="欠品コスト (b)"
                    type="number"
                    value={messaParams.b}
                    onChange={(e) => setMessaParams({ ...messaParams, b: parseFloat(e.target.value) })}
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              <Button
                fullWidth
                variant="contained"
                onClick={handleOptimization}
                disabled={loading}
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '最適化実行'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {optimizationResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  最適化結果
                </Typography>
                
                {optimizationType === 'qr' && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Q-R最適化結果:
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      最適発注量 (Q): {optimizationResult.optimized_parameters?.optimal_order_quantity?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      最適発注点 (R): {optimizationResult.optimized_parameters?.optimal_reorder_point?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      期待コスト（期間当たり）: ¥{optimizationResult.optimized_parameters?.expected_cost_per_period?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      初期発注量: {optimizationResult.initial_parameters?.initial_order_quantity?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      初期発注点: {optimizationResult.initial_parameters?.initial_reorder_point?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      年間欠品コスト: ¥{optimizationResult.shortage_cost?.toFixed(2)}
                    </Typography>
                  </Box>
                )}

                {optimizationType === 'ss' && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      安全在庫最適化結果:
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      最適安全在庫: {optimizationResult.safety_stock?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      発注点: {optimizationResult.reorder_point?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      年間総コスト: ¥{optimizationResult.total_cost?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      サービスレベル: {(optimizationResult.service_level * 100)?.toFixed(1)}%
                    </Typography>
                  </Box>
                )}

                {optimizationType === 'newsvendor' && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      ニューズベンダーモデル結果:
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      最適発注量: {optimizationResult.optimal_order_quantity?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      期待利益: ¥{optimizationResult.expected_profit?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      期待収益: ¥{optimizationResult.expected_revenue?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      期待コスト: ¥{optimizationResult.expected_cost?.toFixed(2)}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      充足確率: {(optimizationResult.service_level * 100)?.toFixed(1)}%
                    </Typography>
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
        在庫管理
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="inventory tabs">
          <Tab label="EOQ計算機" />
          <Tab label="在庫シミュレーション" />
          <Tab label="最適化" />
          <Tab label="マルチエシェロン" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        {renderEOQCalculator()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        {renderSimulation()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        {renderOptimization()}
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        {renderMultiEchelonOptimization()}
      </TabPanel>
    </Box>
  );
};

export default Inventory;