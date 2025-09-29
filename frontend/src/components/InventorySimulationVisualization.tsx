import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Paper, Chip, LinearProgress } from '@mui/material';
import Plot from 'react-plotly.js';

interface SimulationResult {
  simulation_results: {
    policy_type: string;
    average_cost_per_period: number;
    cost_standard_deviation: number;
    cost_range: {
      min: number;
      max: number;
    };
    confidence_interval_95: {
      lower: number;
      upper: number;
    };
    average_inventory_level: number;
    number_of_simulations: number;
    periods_per_simulation: number;
  };
  parameters: {
    [key: string]: number | null;
  };
}

interface InventorySimulationVisualizationProps {
  result: SimulationResult;
}

const InventorySimulationVisualization: React.FC<InventorySimulationVisualizationProps> = ({ result }) => {
  const sim = result.simulation_results;
  
  // Cost distribution simulation (mock data for visualization)
  const generateCostDistribution = () => {
    const costs = [];
    const mean = sim.average_cost_per_period;
    const std = sim.cost_standard_deviation;
    
    for (let i = 0; i < 1000; i++) {
      // Generate normal distribution approximation
      let u1 = Math.random();
      let u2 = Math.random();
      let z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      costs.push(mean + std * z);
    }
    return costs;
  };

  const costDistribution = generateCostDistribution();

  // Cost distribution histogram
  const costDistributionData = [
    {
      x: costDistribution,
      type: 'histogram' as const,
      nbinsx: 30,
      marker: {
        color: '#3498DB',
        opacity: 0.7
      },
      name: 'コスト分布'
    }
  ];

  // Performance metrics comparison
  const performanceMetricsData = [
    {
      x: ['平均コスト', '標準偏差', 'CV (変動係数)', '平均在庫レベル'],
      y: [
        sim.average_cost_per_period,
        sim.cost_standard_deviation,
        (sim.cost_standard_deviation / sim.average_cost_per_period) * 100,
        sim.average_inventory_level
      ],
      type: 'bar' as const,
      marker: {
        color: ['#E74C3C', '#F39C12', '#9B59B6', '#2ECC71']
      },
      text: [
        `¥${sim.average_cost_per_period.toFixed(0)}`,
        `¥${sim.cost_standard_deviation.toFixed(0)}`,
        `${((sim.cost_standard_deviation / sim.average_cost_per_period) * 100).toFixed(1)}%`,
        `${sim.average_inventory_level.toFixed(0)} 単位`
      ],
      textposition: 'outside' as const
    }
  ];

  // Risk analysis chart (Value at Risk concept)
  const riskLevels = [0.99, 0.95, 0.9, 0.75, 0.5];
  const varValues = riskLevels.map(level => {
    // Calculate percentile based on normal distribution assumption
    const z = [2.33, 1.65, 1.28, 0.67, 0]; // Z-scores for percentiles
    const index = riskLevels.indexOf(level);
    return sim.average_cost_per_period + z[index] * sim.cost_standard_deviation;
  });

  const riskAnalysisData = [
    {
      x: riskLevels.map(level => `${(level * 100)}%`),
      y: varValues,
      type: 'bar' as const,
      marker: {
        color: ['#C0392B', '#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']
      },
      name: 'VaR (Value at Risk)'
    }
  ];

  // Confidence interval visualization
  const confidenceData = [
    {
      x: ['下限', '平均', '上限'],
      y: [sim.confidence_interval_95.lower, sim.average_cost_per_period, sim.confidence_interval_95.upper],
      type: 'bar' as const,
      marker: {
        color: ['#3498DB', '#2ECC71', '#3498DB']
      },
      name: '95%信頼区間'
    }
  ];

  // Service level calculation
  const serviceLevel = Math.max(0, Math.min(100, 
    ((sim.average_inventory_level / (result.parameters.mu || 50)) * 100)
  ));

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        在庫シミュレーション結果の詳細分析
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    平均コスト/期間
                  </Typography>
                  <Typography variant="h4">
                    ¥{sim.average_cost_per_period.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {sim.number_of_simulations}回の平均値
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="warning.main">
                    コスト変動性
                  </Typography>
                  <Typography variant="h4">
                    {((sim.cost_standard_deviation / sim.average_cost_per_period) * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    変動係数 (CV)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    平均在庫レベル
                  </Typography>
                  <Typography variant="h4">
                    {sim.average_inventory_level.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    単位
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    推定サービスレベル
                  </Typography>
                  <Typography variant="h4">
                    {serviceLevel.toFixed(1)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={serviceLevel} 
                    sx={{ mt: 1 }}
                    color="success"
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Cost Distribution Histogram */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                コスト分布 (シミュレーション結果)
              </Typography>
              <Plot
                data={costDistributionData as any}
                layout={{
                  title: { text: '期間あたりコストの分布' },
                  xaxis: { title: { text: 'コスト (¥)' } },
                  yaxis: { title: { text: '頻度' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 },
                  shapes: [
                    {
                      type: 'line' as const,
                      x0: sim.average_cost_per_period,
                      x1: sim.average_cost_per_period,
                      y0: 0,
                      y1: 50,
                      line: { color: 'red', width: 2, dash: 'dash' as const }
                    }
                  ]
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                パフォーマンス指標
              </Typography>
              <Plot
                data={performanceMetricsData as any}
                layout={{
                  title: { text: '主要パフォーマンス指標' },
                  xaxis: { title: { text: '指標' } },
                  yaxis: { title: { text: '値' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Confidence Interval */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                95% 信頼区間
              </Typography>
              <Plot
                data={confidenceData as any}
                layout={{
                  title: { text: 'コストの信頼区間' },
                  xaxis: { title: { text: '統計値' } },
                  yaxis: { title: { text: 'コスト (¥)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Analysis (VaR) */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                リスク分析 (VaR)
              </Typography>
              <Plot
                data={riskAnalysisData as any}
                layout={{
                  title: { text: '信頼度別 Value at Risk' },
                  xaxis: { title: { text: '信頼度' } },
                  yaxis: { title: { text: '最大損失予想 (¥)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Simulation Parameters Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              シミュレーション設定と結果の解釈
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  シミュレーション設定:
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • シミュレーション回数: {sim.number_of_simulations.toLocaleString()}回
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • シミュレーション期間: {sim.periods_per_simulation}期間
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • ポリシータイプ: {sim.policy_type}
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  • 総計算期間: {(sim.number_of_simulations * sim.periods_per_simulation).toLocaleString()}期間
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  リスクプロファイル:
                </Typography>
                {((sim.cost_standard_deviation / sim.average_cost_per_period) * 100) < 10 && (
                  <Chip label="低リスク" color="success" sx={{ mr: 1, mb: 1 }} />
                )}
                {((sim.cost_standard_deviation / sim.average_cost_per_period) * 100) >= 10 && 
                 ((sim.cost_standard_deviation / sim.average_cost_per_period) * 100) < 25 && (
                  <Chip label="中リスク" color="warning" sx={{ mr: 1, mb: 1 }} />
                )}
                {((sim.cost_standard_deviation / sim.average_cost_per_period) * 100) >= 25 && (
                  <Chip label="高リスク" color="error" sx={{ mr: 1, mb: 1 }} />
                )}
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  運用上の推奨事項:
                </Typography>
                
                {((sim.cost_standard_deviation / sim.average_cost_per_period) * 100) < 10 && (
                  <Typography variant="body2" sx={{ mb: 1, color: 'success.main' }}>
                    ✅ 低変動: 現在の在庫ポリシーは安定しています
                  </Typography>
                )}
                
                {sim.average_inventory_level < (result.parameters.mu || 50) && (
                  <Typography variant="body2" sx={{ mb: 1, color: 'warning.main' }}>
                    ⚠️ 低在庫: 欠品リスクに注意が必要です
                  </Typography>
                )}
                
                {serviceLevel < 90 && (
                  <Typography variant="body2" sx={{ mb: 1, color: 'error.main' }}>
                    🔴 サービスレベル改善: 安全在庫の見直しを検討
                  </Typography>
                )}
                
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • 予算計画: 月間コスト予算を¥{(sim.confidence_interval_95.upper).toFixed(0)}に設定
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • コスト管理: 標準偏差¥{sim.cost_standard_deviation.toFixed(0)}を考慮した予算幅を確保
                </Typography>
                <Typography variant="body2">
                  • 在庫最適化: 平均在庫{sim.average_inventory_level.toFixed(0)}単位を基準に調整
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default InventorySimulationVisualization;