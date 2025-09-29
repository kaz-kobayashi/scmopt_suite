import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Paper, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import Plot from 'react-plotly.js';

interface CO2Result {
  emissions_calculation: {
    fuel_consumption_L_per_ton_km: number;
    co2_emissions_g_per_ton_km: number;
    fuel_type: string;
    vehicle_capacity_tons: number;
    loading_rate: number;
  };
  annual_estimates: {
    estimated_annual_distance_km: number;
    estimated_annual_fuel_consumption_L: number;
    estimated_annual_co2_emissions_kg: number;
  };
  optimization_suggestions: {
    improve_loading_rate: boolean;
    consider_larger_vehicle: boolean;
    efficiency_score: number;
  };
}

interface DistanceMatrixResult {
  distance_matrix: {
    [location: string]: {
      [destination: string]: number;
    };
  };
  duration_matrix: {
    [location: string]: {
      [destination: string]: number;
    };
  };
  locations: string[];
  matrix_size: number;
  units: {
    distance: string;
    duration: string;
  };
}

interface RouteOptimizationResult {
  routes: Array<{
    route_id: number;
    sequence: number[];
    locations: string[];
    total_distance: number;
    total_demand: number;
    capacity_utilization: number;
  }>;
  summary: {
    total_routes: number;
    total_distance: number;
    total_demand_served: number;
    avg_capacity_utilization: number;
    customers_served: number;
    customers_unserved: number;
  };
  optimization_stats: {
    best_cost: number;
    iterations: number;
    runtime_seconds: number;
    algorithm: string;
    convergence: boolean;
  };
}

interface AdvancedRoutingVisualizationProps {
  co2Result?: CO2Result;
  distanceMatrixResult?: DistanceMatrixResult;
  routeOptimizationResult?: RouteOptimizationResult;
  type: 'co2' | 'distance-matrix' | 'route-optimization';
}

const AdvancedRoutingVisualization: React.FC<AdvancedRoutingVisualizationProps> = ({
  co2Result,
  distanceMatrixResult,
  routeOptimizationResult,
  type
}) => {

  const renderCO2Visualization = () => {
    if (!co2Result) return null;

    // Emissions breakdown
    const emissionsData = [
      {
        labels: ['年間CO2排出量', '削減可能量'],
        values: [
          co2Result.annual_estimates.estimated_annual_co2_emissions_kg,
          co2Result.annual_estimates.estimated_annual_co2_emissions_kg * (1 - co2Result.optimization_suggestions.efficiency_score)
        ],
        type: 'pie' as const,
        marker: {
          colors: ['#E74C3C', '#2ECC71']
        }
      }
    ];

    // Fuel consumption vs capacity utilization
    const efficiencyData = [
      {
        x: ['現在の積載率', '推奨積載率'],
        y: [
          co2Result.emissions_calculation.loading_rate * 100,
          Math.min(95, co2Result.emissions_calculation.loading_rate * 100 + 20)
        ],
        type: 'bar' as const,
        marker: {
          color: ['#F39C12', '#2ECC71']
        },
        name: '積載率 (%)'
      }
    ];

    // Annual estimates comparison
    const annualData = [
      {
        x: ['距離', '燃料消費', 'CO2排出'],
        y: [
          co2Result.annual_estimates.estimated_annual_distance_km / 1000, // Convert to thousand km
          co2Result.annual_estimates.estimated_annual_fuel_consumption_L / 1000, // Convert to thousand L
          co2Result.annual_estimates.estimated_annual_co2_emissions_kg / 1000 // Convert to tons
        ],
        type: 'bar' as const,
        marker: {
          color: ['#3498DB', '#E67E22', '#E74C3C']
        },
        text: [
          `${co2Result.annual_estimates.estimated_annual_distance_km.toFixed(0)} km`,
          `${co2Result.annual_estimates.estimated_annual_fuel_consumption_L.toFixed(0)} L`,
          `${co2Result.annual_estimates.estimated_annual_co2_emissions_kg.toFixed(0)} kg`
        ],
        textposition: 'outside' as const
      }
    ];

    return (
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="error">
                    年間CO2排出量
                  </Typography>
                  <Typography variant="h3">
                    {(co2Result.annual_estimates.estimated_annual_co2_emissions_kg / 1000).toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    トン CO2
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="warning.main">
                    燃費効率性
                  </Typography>
                  <Typography variant="h3">
                    {(co2Result.optimization_suggestions.efficiency_score * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    効率スコア
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    積載率
                  </Typography>
                  <Typography variant="h3">
                    {(co2Result.emissions_calculation.loading_rate * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    現在の積載効率
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    年間走行距離
                  </Typography>
                  <Typography variant="h3">
                    {(co2Result.annual_estimates.estimated_annual_distance_km / 1000).toFixed(0)}K
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    キロメートル
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* CO2 Emissions Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CO2排出量の内訳
              </Typography>
              <Plot
                data={emissionsData as any}
                layout={{
                  height: 400,
                  margin: { t: 20, l: 20, r: 20, b: 20 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Loading Rate Efficiency */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                積載率効率性
              </Typography>
              <Plot
                data={efficiencyData as any}
                layout={{
                  title: { text: '現在 vs 推奨積載率' },
                  xaxis: { title: { text: '状態' } },
                  yaxis: { title: { text: '積載率 (%)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Annual Estimates */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                年間推定値
              </Typography>
              <Plot
                data={annualData as any}
                layout={{
                  title: { text: '年間の距離・燃料・排出量' },
                  xaxis: { title: { text: '指標' } },
                  yaxis: { title: { text: '値 (千単位)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Optimization Suggestions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              最適化提案
            </Typography>
            <Grid container spacing={2}>
              {co2Result.optimization_suggestions.improve_loading_rate && (
                <Grid item xs={12} md={6}>
                  <Chip label="積載率改善推奨" color="warning" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    積載率を向上させることで、CO2排出量を削減できます
                  </Typography>
                </Grid>
              )}
              {co2Result.optimization_suggestions.consider_larger_vehicle && (
                <Grid item xs={12} md={6}>
                  <Chip label="車両サイズ見直し" color="info" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    より大きな車両の使用を検討してください
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  const renderDistanceMatrixVisualization = () => {
    if (!distanceMatrixResult) return null;

    // Create heatmap data for distance matrix
    const locations = distanceMatrixResult.locations;
    const matrix = distanceMatrixResult.distance_matrix;
    
    const heatmapData: number[][] = [];
    const xLabels: string[] = [];
    const yLabels: string[] = [];
    
    locations.forEach(origin => {
      const row: number[] = [];
      yLabels.push(origin);
      locations.forEach(dest => {
        xLabels.push(dest);
        row.push(matrix[origin]?.[dest] || 0);
      });
      heatmapData.push(row);
    });

    const distanceHeatmapData = [
      {
        z: heatmapData,
        x: locations,
        y: locations,
        type: 'heatmap' as const,
        colorscale: 'Viridis' as const,
        showscale: true
      }
    ];

    // Statistics
    const allDistances = Object.values(matrix).flatMap(row => 
      Object.values(row).filter(dist => dist > 0)
    );
    const avgDistance = allDistances.reduce((a, b) => a + b, 0) / allDistances.length;
    const maxDistance = Math.max(...allDistances);
    const minDistance = Math.min(...allDistances.filter(d => d > 0));

    return (
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    総拠点数
                  </Typography>
                  <Typography variant="h3">
                    {distanceMatrixResult.matrix_size}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    拠点
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    平均距離
                  </Typography>
                  <Typography variant="h3">
                    {avgDistance.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {distanceMatrixResult.units.distance}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="error">
                    最長距離
                  </Typography>
                  <Typography variant="h3">
                    {maxDistance.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {distanceMatrixResult.units.distance}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    最短距離
                  </Typography>
                  <Typography variant="h3">
                    {minDistance.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {distanceMatrixResult.units.distance}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Distance Matrix Heatmap */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                距離マトリクス ヒートマップ
              </Typography>
              <Plot
                data={distanceHeatmapData as any}
                layout={{
                  title: { text: '拠点間距離の可視化' },
                  xaxis: { title: { text: '到着地' } },
                  yaxis: { title: { text: '出発地' } },
                  height: 600,
                  margin: { t: 50, l: 100, r: 60, b: 100 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const renderRouteOptimizationVisualization = () => {
    if (!routeOptimizationResult) return null;

    const result = routeOptimizationResult;

    // Route efficiency analysis
    const routeEfficiencyData = [
      {
        x: result.routes.map(route => `ルート ${route.route_id}`),
        y: result.routes.map(route => route.capacity_utilization * 100),
        type: 'bar' as const,
        marker: {
          color: result.routes.map(route => 
            route.capacity_utilization > 0.8 ? '#2ECC71' : 
            route.capacity_utilization > 0.6 ? '#F39C12' : '#E74C3C'
          )
        },
        name: '積載率 (%)'
      }
    ];

    // Distance distribution
    const distanceData = [
      {
        x: result.routes.map(route => `ルート ${route.route_id}`),
        y: result.routes.map(route => route.total_distance),
        type: 'bar' as const,
        marker: {
          color: '#3498DB'
        },
        name: '距離 (km)'
      }
    ];

    return (
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    総ルート数
                  </Typography>
                  <Typography variant="h3">
                    {result.summary.total_routes}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    最適化されたルート
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    総走行距離
                  </Typography>
                  <Typography variant="h3">
                    {result.summary.total_distance.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    km
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    平均積載率
                  </Typography>
                  <Typography variant="h3">
                    {(result.summary.avg_capacity_utilization * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    車両容量利用率
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="warning.main">
                    顧客充足率
                  </Typography>
                  <Typography variant="h3">
                    {((result.summary.customers_served / (result.summary.customers_served + result.summary.customers_unserved)) * 100).toFixed(0)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {result.summary.customers_served}/{result.summary.customers_served + result.summary.customers_unserved} 顧客
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Route Efficiency Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ルート別積載率
              </Typography>
              <Plot
                data={routeEfficiencyData as any}
                layout={{
                  title: { text: '各ルートの容量利用率' },
                  xaxis: { title: { text: 'ルート' } },
                  yaxis: { title: { text: '積載率 (%)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Distance Distribution */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ルート別走行距離
              </Typography>
              <Plot
                data={distanceData as any}
                layout={{
                  title: { text: '各ルートの走行距離' },
                  xaxis: { title: { text: 'ルート' } },
                  yaxis: { title: { text: '距離 (km)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Route Details Table */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ルート詳細
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>ルートID</TableCell>
                      <TableCell align="right">顧客数</TableCell>
                      <TableCell align="right">距離 (km)</TableCell>
                      <TableCell align="right">需要量</TableCell>
                      <TableCell align="right">積載率 (%)</TableCell>
                      <TableCell>効率評価</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.routes.map((route) => (
                      <TableRow key={route.route_id}>
                        <TableCell component="th" scope="row">
                          ルート {route.route_id}
                        </TableCell>
                        <TableCell align="right">{route.locations.length - 2}</TableCell>
                        <TableCell align="right">{route.total_distance.toFixed(1)}</TableCell>
                        <TableCell align="right">{route.total_demand.toFixed(1)}</TableCell>
                        <TableCell align="right">{(route.capacity_utilization * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          {route.capacity_utilization > 0.8 ? (
                            <Chip label="高効率" color="success" size="small" />
                          ) : route.capacity_utilization > 0.6 ? (
                            <Chip label="標準" color="warning" size="small" />
                          ) : (
                            <Chip label="改善要" color="error" size="small" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Optimization Statistics */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              最適化統計
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>アルゴリズム:</strong> {result.optimization_stats.algorithm}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>実行時間:</strong> {result.optimization_stats.runtime_seconds.toFixed(2)} 秒
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>反復回数:</strong> {result.optimization_stats.iterations.toLocaleString()}
                </Typography>
                <Typography variant="body2">
                  <strong>収束状態:</strong> {result.optimization_stats.convergence ? '✅ 収束' : '❌ 未収束'}
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>最適コスト:</strong> {result.optimization_stats.best_cost.toFixed(2)}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>効率性評価:</strong> 
                  {result.summary.avg_capacity_utilization > 0.8 ? ' 非常に良好' :
                   result.summary.avg_capacity_utilization > 0.6 ? ' 良好' : ' 要改善'}
                </Typography>
                <Typography variant="body2">
                  <strong>推定節約:</strong> 従来比 {((1 - result.summary.avg_capacity_utilization) * 100).toFixed(0)}% のコスト削減
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  switch (type) {
    case 'co2':
      return renderCO2Visualization();
    case 'distance-matrix':
      return renderDistanceMatrixVisualization();
    case 'route-optimization':
      return renderRouteOptimizationVisualization();
    default:
      return null;
  }
};

export default AdvancedRoutingVisualization;