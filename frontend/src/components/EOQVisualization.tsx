import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Paper, Chip } from '@mui/material';
import Plot from 'react-plotly.js';

interface EOQResult {
  optimal_order_quantity: number;
  total_annual_cost: number;
  annual_ordering_cost: number;
  annual_holding_cost: number;
  cycle_time_periods: number;
  parameters: {
    fixed_cost: number;
    demand_rate: number;
    holding_cost: number;
    backorder_cost: number;
    interest_rate: number;
    unit_cost: number;
    service_level: number;
  };
}

interface EOQVisualizationProps {
  result: EOQResult;
}

const EOQVisualization: React.FC<EOQVisualizationProps> = ({ result }) => {
  // Cost breakdown chart data
  const costBreakdownData = [
    {
      labels: ['発注コスト', '保管コスト', 'その他コスト'],
      values: [
        result.annual_ordering_cost,
        result.annual_holding_cost,
        Math.max(0, result.total_annual_cost - result.annual_ordering_cost - result.annual_holding_cost)
      ],
      type: 'pie' as const,
      marker: {
        colors: ['#FF6B6B', '#4ECDC4', '#95E1D3']
      },
      textinfo: 'label+percent+value',
    }
  ];

  // EOQ sensitivity analysis - show cost curves
  const quantities = [];
  const totalCosts = [];
  const orderingCosts = [];
  const holdingCosts = [];
  
  for (let q = result.optimal_order_quantity * 0.5; q <= result.optimal_order_quantity * 2; q += result.optimal_order_quantity * 0.1) {
    const orderingCost = (result.parameters.demand_rate * result.parameters.fixed_cost) / q;
    const holdingCost = (q * result.parameters.holding_cost) / 2;
    const totalCost = orderingCost + holdingCost;
    
    quantities.push(q);
    totalCosts.push(totalCost);
    orderingCosts.push(orderingCost);
    holdingCosts.push(holdingCost);
  }

  const sensitivityAnalysisData = [
    {
      x: quantities,
      y: totalCosts,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: '総コスト',
      line: { color: '#3498DB', width: 3 }
    },
    {
      x: quantities,
      y: orderingCosts,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: '発注コスト',
      line: { color: '#FF6B6B', width: 2, dash: 'dash' }
    },
    {
      x: quantities,
      y: holdingCosts,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: '保管コスト',
      line: { color: '#4ECDC4', width: 2, dash: 'dash' }
    }
  ];

  // Parameters comparison (normalized for visualization)
  const normalizeValue = (value: number, max: number) => (value / max) * 100;
  const maxParam = Math.max(
    result.parameters.fixed_cost,
    result.parameters.demand_rate,
    result.parameters.holding_cost * 100,
    result.parameters.unit_cost
  );

  const parametersData = [
    {
      x: ['固定発注コスト', '需要率', '保管コスト率', '単価'],
      y: [
        normalizeValue(result.parameters.fixed_cost, maxParam),
        normalizeValue(result.parameters.demand_rate, maxParam),
        normalizeValue(result.parameters.holding_cost * 100, maxParam),
        normalizeValue(result.parameters.unit_cost, maxParam)
      ],
      type: 'bar' as const,
      marker: {
        color: ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
      },
      text: [
        `¥${result.parameters.fixed_cost}`,
        `${result.parameters.demand_rate} 単位/年`,
        `${(result.parameters.holding_cost * 100).toFixed(1)}%`,
        `¥${result.parameters.unit_cost}`
      ],
      textposition: 'outside' as const
    }
  ];

  // Key metrics for comparison
  const metrics = [
    {
      label: '最適発注量',
      value: result.optimal_order_quantity.toFixed(0),
      unit: '単位',
      color: 'primary',
      description: '在庫コストを最小化する発注量'
    },
    {
      label: '年間総コスト',
      value: result.total_annual_cost.toFixed(0),
      unit: '円',
      color: 'error',
      description: '発注・保管コストの合計'
    },
    {
      label: 'サイクル時間',
      value: result.cycle_time_periods.toFixed(1),
      unit: '期間',
      color: 'info',
      description: '在庫切れまでの期間'
    },
    {
      label: 'サービスレベル',
      value: (result.parameters.service_level * 100).toFixed(1),
      unit: '%',
      color: 'success',
      description: '需要充足率の目標値'
    }
  ];

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        EOQ分析結果の詳細可視化
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            {metrics.map((metric, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" color={metric.color as any}>
                      {metric.label}
                    </Typography>
                    <Typography variant="h3">
                      {metric.value}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {metric.unit}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {metric.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>

        {/* Cost Breakdown Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                年間コスト内訳
              </Typography>
              <Plot
                data={costBreakdownData as any}
                layout={{
                  height: 400,
                  margin: { t: 20, l: 20, r: 20, b: 20 },
                  showlegend: true,
                  legend: { orientation: 'h' as const, y: -0.2 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Parameters Visualization */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                入力パラメータ (正規化表示)
              </Typography>
              <Plot
                data={parametersData as any}
                layout={{
                  title: { text: 'EOQ計算に使用されたパラメータ' },
                  xaxis: { title: { text: 'パラメータ' } },
                  yaxis: { title: { text: '正規化値 (%)' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* EOQ Sensitivity Analysis */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                EOQ感度分析 - 発注量とコストの関係
              </Typography>
              <Plot
                data={sensitivityAnalysisData as any}
                layout={{
                  title: { text: '発注量による年間コストの変化' },
                  xaxis: { title: { text: '発注量 (単位)' } },
                  yaxis: { title: { text: '年間コスト (円)' } },
                  height: 500,
                  margin: { t: 50, l: 80, r: 20, b: 60 },
                  shapes: [
                    {
                      type: 'line' as const,
                      x0: result.optimal_order_quantity,
                      x1: result.optimal_order_quantity,
                      y0: 0,
                      y1: Math.max(...totalCosts),
                      line: { color: 'red', width: 2, dash: 'dot' as const }
                    }
                  ],
                  annotations: [
                    {
                      x: result.optimal_order_quantity,
                      y: Math.max(...totalCosts) * 0.9,
                      text: `最適EOQ: ${result.optimal_order_quantity.toFixed(0)}`,
                      showarrow: true,
                      arrowcolor: 'red',
                      font: { color: 'red' }
                    }
                  ]
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              EOQ分析の解釈と推奨アクション
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  コスト効率性:
                </Typography>
                {result.annual_ordering_cost > result.annual_holding_cost ? (
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Chip label="発注頻度を増やす" color="warning" sx={{ mr: 1 }} />
                    <Typography variant="body2">
                      発注コストが高いため、発注量を増やして頻度を減らすことを検討
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Chip label="在庫圧縮" color="info" sx={{ mr: 1 }} />
                    <Typography variant="body2">
                      保管コストが高いため、発注量を減らして在庫を圧縮することを検討
                    </Typography>
                  </Box>
                )}
                
                <Typography variant="subtitle2" gutterBottom>
                  サイクル管理:
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  {result.cycle_time_periods < 1 
                    ? '短期サイクル: 頻繁な発注が必要です'
                    : result.cycle_time_periods > 12
                    ? '長期サイクル: 在庫管理の負荷に注意が必要です'
                    : '適切なサイクル: バランスの取れた発注間隔です'
                  }
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  コスト削減の機会:
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • 発注コスト削減: EDIやシステム自動化で{((result.annual_ordering_cost * 0.2)).toFixed(0)}円の削減可能
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  • 保管効率化: 倉庫レイアウト改善で{((result.annual_holding_cost * 0.15)).toFixed(0)}円の削減可能
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  • 需要予測精度向上により安全在庫を{((result.parameters.demand_rate * 0.05)).toFixed(0)}単位削減可能
                </Typography>
                
                <Typography variant="subtitle2" gutterBottom>
                  リスク要因:
                </Typography>
                <Typography variant="body2">
                  • 需要変動: ±{((result.parameters.demand_rate * 0.1)).toFixed(0)}単位の変動でコストが±{((result.total_annual_cost * 0.05)).toFixed(0)}円変動
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default EOQVisualization;