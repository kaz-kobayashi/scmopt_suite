import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Chip, Paper } from '@mui/material';
import Plot from 'react-plotly.js';

interface ABCAnalysisResult {
  aggregated_data: Array<{
    [key: string]: string | number;
    abc: string;
    rank: number;
  }>;
  classified_data: Array<{
    [key: string]: string | number;
    abc: string;
    rank: number;
  }>;
  categories: {
    [key: number]: string[];
  };
  summary: {
    total_items: number;
    total_value: number;
    thresholds: number[];
  };
}

interface RiskPoolingResult {
  original_variance: number;
  pooled_variance: number;
  variance_reduction_percentage: number;
  pooling_benefit: number;
  risk_pooling_analysis: {
    individual_cv: number[];
    pooled_cv: number;
    correlation_matrix: number[][];
  };
}

interface ParetoResult {
  pareto_data: Array<{
    [key: string]: any;
    cumulative_percentage: number;
  }>;
  pareto_80_20: {
    items_80_percent: number;
    value_80_percent: number;
  };
}

interface AdvancedAnalyticsVisualizationProps {
  abcResult?: ABCAnalysisResult;
  riskPoolingResult?: RiskPoolingResult;
  paretoResult?: ParetoResult;
  treemapResult?: any;
  type: 'abc' | 'risk-pooling' | 'pareto' | 'treemap';
}

const AdvancedAnalyticsVisualization: React.FC<AdvancedAnalyticsVisualizationProps> = ({
  abcResult,
  riskPoolingResult,
  paretoResult,
  treemapResult,
  type
}) => {

  const renderABCVisualization = () => {
    if (!abcResult) return null;

    // ABC Category Distribution
    const abcCounts = { A: 0, B: 0, C: 0 };
    const abcValues = { A: 0, B: 0, C: 0 };
    
    abcResult.aggregated_data.forEach(item => {
      const category = item.abc as 'A' | 'B' | 'C';
      abcCounts[category]++;
      abcValues[category] += Number(item.value || 0);
    });

    const categoryDistributionData = [
      {
        labels: ['A: 重要', 'B: 中程度', 'C: 低重要'],
        values: [abcCounts.A, abcCounts.B, abcCounts.C],
        type: 'pie' as const,
        marker: {
          colors: ['#FF6B6B', '#4ECDC4', '#95E1D3']
        },
        textinfo: 'label+percent+value',
      }
    ];

    const valueDistributionData = [
      {
        x: ['A: 重要', 'B: 中程度', 'C: 低重要'],
        y: [abcValues.A, abcValues.B, abcValues.C],
        type: 'bar' as const,
        marker: {
          color: ['#FF6B6B', '#4ECDC4', '#95E1D3']
        },
        text: [
          `¥${abcValues.A.toLocaleString()}`,
          `¥${abcValues.B.toLocaleString()}`,
          `¥${abcValues.C.toLocaleString()}`
        ],
        textposition: 'outside' as const
      }
    ];

    // Cumulative analysis
    const sortedData = [...abcResult.aggregated_data].sort((a, b) => Number(b.value || 0) - Number(a.value || 0));
    const cumulativeData = sortedData.map((_, index) => {
      const cumValue = sortedData.slice(0, index + 1).reduce((sum, item) => sum + Number(item.value || 0), 0);
      return (cumValue / abcResult.summary.total_value) * 100;
    });

    const cumulativeAnalysisData = [
      {
        x: sortedData.map((_, index) => index + 1),
        y: cumulativeData,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: '累積割合',
        line: { color: '#3498DB', width: 3 },
        marker: { color: '#3498DB', size: 6 }
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
                    カテゴリA
                  </Typography>
                  <Typography variant="h4">
                    {abcCounts.A}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    アイテム数 ({((abcCounts.A / abcResult.summary.total_items) * 100).toFixed(1)}%)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    価値: ¥{abcValues.A.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="info.main">
                    カテゴリB
                  </Typography>
                  <Typography variant="h4">
                    {abcCounts.B}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    アイテム数 ({((abcCounts.B / abcResult.summary.total_items) * 100).toFixed(1)}%)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    価値: ¥{abcValues.B.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="success.main">
                    カテゴリC
                  </Typography>
                  <Typography variant="h4">
                    {abcCounts.C}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    アイテム数 ({((abcCounts.C / abcResult.summary.total_items) * 100).toFixed(1)}%)
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    価値: ¥{abcValues.C.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    総計
                  </Typography>
                  <Typography variant="h4">
                    {abcResult.summary.total_items}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    総アイテム数
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    総価値: ¥{abcResult.summary.total_value.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Category Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                カテゴリ別アイテム分布
              </Typography>
              <Plot
                data={categoryDistributionData as any}
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

        {/* Value Distribution Bar Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                カテゴリ別価値分布
              </Typography>
              <Plot
                data={valueDistributionData as any}
                layout={{
                  title: { text: '各カテゴリの総価値' },
                  xaxis: { title: { text: 'ABCカテゴリ' } },
                  yaxis: { title: { text: '価値 (¥)' } },
                  height: 400,
                  margin: { t: 50, l: 80, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Cumulative Analysis */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                累積価値分析 (パレート曲線)
              </Typography>
              <Plot
                data={cumulativeAnalysisData as any}
                layout={{
                  title: { text: 'アイテムランク vs 累積価値割合' },
                  xaxis: { title: { text: 'アイテムランク' } },
                  yaxis: { title: { text: '累積価値割合 (%)' }, range: [0, 100] },
                  height: 400,
                  margin: { t: 50, l: 80, r: 20, b: 60 },
                  shapes: [
                    {
                      type: 'line' as const,
                      x0: 0,
                      x1: abcResult.summary.total_items,
                      y0: 80,
                      y1: 80,
                      line: { color: 'red', width: 2, dash: 'dash' as const },
                    }
                  ],
                  annotations: [
                    {
                      x: abcResult.summary.total_items * 0.7,
                      y: 85,
                      text: '80%ライン',
                      showarrow: false,
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
      </Grid>
    );
  };

  const renderRiskPoolingVisualization = () => {
    if (!riskPoolingResult) return null;

    // Variance Comparison
    const varianceComparisonData = [
      {
        x: ['個別管理', 'プール管理'],
        y: [riskPoolingResult.original_variance, riskPoolingResult.pooled_variance],
        type: 'bar' as const,
        marker: {
          color: ['#FF6B6B', '#4ECDC4']
        },
        text: [
          riskPoolingResult.original_variance.toFixed(2),
          riskPoolingResult.pooled_variance.toFixed(2)
        ],
        textposition: 'outside' as const
      }
    ];

    // CV Comparison
    const cvComparisonData = [
      {
        x: riskPoolingResult.risk_pooling_analysis.individual_cv.map((_, index) => `拠点${index + 1}`),
        y: riskPoolingResult.risk_pooling_analysis.individual_cv,
        type: 'bar' as const,
        name: '個別CV',
        marker: { color: '#FF6B6B' }
      },
      {
        x: ['プール'],
        y: [riskPoolingResult.risk_pooling_analysis.pooled_cv],
        type: 'bar' as const,
        name: 'プールCV',
        marker: { color: '#4ECDC4' }
      }
    ];

    return (
      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    分散削減率
                  </Typography>
                  <Typography variant="h3" color="success.main">
                    {riskPoolingResult.variance_reduction_percentage.toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    リスクプールによる改善効果
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    プール効果
                  </Typography>
                  <Typography variant="h3">
                    {riskPoolingResult.pooling_benefit.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    統合による利益指標
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    プールCV
                  </Typography>
                  <Typography variant="h3">
                    {riskPoolingResult.risk_pooling_analysis.pooled_cv.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    変動係数 (統合後)
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {/* Variance Comparison */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                分散の比較
              </Typography>
              <Plot
                data={varianceComparisonData as any}
                layout={{
                  title: { text: '個別管理 vs プール管理' },
                  xaxis: { title: { text: '管理方式' } },
                  yaxis: { title: { text: '分散' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* CV Comparison */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                変動係数 (CV) の比較
              </Typography>
              <Plot
                data={cvComparisonData as any}
                layout={{
                  title: { text: '個別拠点 vs プール運用' },
                  xaxis: { title: { text: '拠点' } },
                  yaxis: { title: { text: '変動係数' } },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Benefits Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              リスクプーリング効果の解釈
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip label="効果大" color="success" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    分散削減率が20%以上: 高いプール効果
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip label="効果中" color="warning" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    分散削減率が10-20%: 中程度のプール効果
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Chip label="効果小" color="error" sx={{ mr: 1 }} />
                  <Typography variant="body2">
                    分散削減率が10%未満: 限定的なプール効果
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                  推奨アクション:
                </Typography>
                {riskPoolingResult.variance_reduction_percentage > 20 && (
                  <Typography variant="body2" color="success.main">
                    • 在庫統合を積極的に実施<br/>
                    • 安全在庫の大幅削減が可能<br/>
                    • コスト削減効果が期待大
                  </Typography>
                )}
                {riskPoolingResult.variance_reduction_percentage <= 20 && riskPoolingResult.variance_reduction_percentage > 10 && (
                  <Typography variant="body2" color="warning.main">
                    • 段階的な統合を検討<br/>
                    • 運用コストとのバランスを評価<br/>
                    • 特定カテゴリーでの実施を推奨
                  </Typography>
                )}
                {riskPoolingResult.variance_reduction_percentage <= 10 && (
                  <Typography variant="body2" color="error.main">
                    • 統合効果は限定的<br/>
                    • 他の在庫最適化手法を検討<br/>
                    • 需要予測精度の向上を優先
                  </Typography>
                )}
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  const renderParetoVisualization = () => {
    if (!paretoResult) return null;

    // Pareto Chart Data
    const paretoData = paretoResult.pareto_data.sort((a, b) => Number(b.value || 0) - Number(a.value || 0));
    
    const paretoChartData = [
      {
        x: paretoData.map((item, index) => index + 1),
        y: paretoData.map(item => Number(item.value || 0)),
        type: 'bar' as const,
        name: '価値',
        marker: { color: '#3498DB' },
        yaxis: 'y'
      },
      {
        x: paretoData.map((item, index) => index + 1),
        y: paretoData.map(item => item.cumulative_percentage),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: '累積%',
        line: { color: '#E74C3C', width: 3 },
        marker: { color: '#E74C3C', size: 6 },
        yaxis: 'y2'
      }
    ];

    return (
      <Grid container spacing={3}>
        {/* 80-20 Rule Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#e3f2fd' }}>
            <Typography variant="h6" gutterBottom>
              パレートの法則 (80-20ルール) 分析結果
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="h4" color="primary">
                  {paretoResult.pareto_80_20.items_80_percent}
                </Typography>
                <Typography variant="body1">
                  アイテム数で全体の80%の価値を占める
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h4" color="secondary">
                  {paretoResult.pareto_80_20.value_80_percent.toFixed(1)}%
                </Typography>
                <Typography variant="body1">
                  上位アイテムの価値集中度
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Pareto Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                パレート図 - アイテム別価値分析
              </Typography>
              <Plot
                data={paretoChartData as any}
                layout={{
                  title: { text: 'パレート分析: 価値の集中度' },
                  xaxis: { title: { text: 'アイテムランキング' } },
                  yaxis: { 
                    title: { text: '価値' },
                    side: 'left' as const
                  },
                  yaxis2: {
                    title: { text: '累積割合 (%)' },
                    side: 'right' as const,
                    overlaying: 'y' as const,
                    range: [0, 100]
                  },
                  height: 500,
                  margin: { t: 50, l: 80, r: 80, b: 60 },
                  shapes: [
                    {
                      type: 'line' as const,
                      x0: 0,
                      x1: paretoData.length,
                      y0: 80,
                      y1: 80,
                      yref: 'y2',
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
      </Grid>
    );
  };

  switch (type) {
    case 'abc':
      return renderABCVisualization();
    case 'risk-pooling':
      return renderRiskPoolingVisualization();
    case 'pareto':
      return renderParetoVisualization();
    default:
      return null;
  }
};

export default AdvancedAnalyticsVisualization;