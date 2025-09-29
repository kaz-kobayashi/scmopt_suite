import React from 'react';
import { Box, Typography, Card, CardContent, Grid, Paper, Chip } from '@mui/material';
import Plot from 'react-plotly.js';
import {
  Storage as InventoryIcon,
  DirectionsCar as RouteIcon,
  BarChart as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  LocalShipping as EcoIcon
} from '@mui/icons-material';

const DashboardSummary: React.FC = () => {
  // Mock data for demonstration - in real app, this would come from API
  const summaryData = {
    analytics: {
      abcAnalysis: { totalItems: 1245, categoryA: 249, categoryB: 373, categoryC: 623 },
      riskPooling: { varianceReduction: 34.2, poolingBenefit: 15.8 },
      paretoCompliance: 78.5
    },
    inventory: {
      totalValue: 2450000,
      eoqOptimization: { annualSavings: 125000, optimalQuantity: 1250 },
      serviceLevel: 94.5,
      turnoverRate: 8.2
    },
    routing: {
      totalRoutes: 37,
      totalDistance: 1847.5,
      co2Emissions: 2.3,
      fuelEfficiency: 87.2,
      costSavings: 48000
    }
  };

  // Performance metrics overview
  const performanceData = [
    {
      x: ['分析効率', '在庫最適化', 'ルート効率', '全体パフォーマンス'],
      y: [summaryData.analytics.paretoCompliance, summaryData.inventory.serviceLevel, summaryData.routing.fuelEfficiency, 88.1],
      type: 'bar' as const,
      marker: {
        color: ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
      },
      text: ['78.5%', '94.5%', '87.2%', '88.1%'],
      textposition: 'outside' as const
    }
  ];

  // Cost savings breakdown
  const costSavingsData = [
    {
      labels: ['在庫最適化', 'ルート最適化', 'リスクプーリング', 'その他効率化'],
      values: [
        summaryData.inventory.eoqOptimization.annualSavings,
        summaryData.routing.costSavings,
        30000,
        22000
      ],
      type: 'pie' as const,
      marker: {
        colors: ['#2ECC71', '#E74C3C', '#3498DB', '#F39C12']
      }
    }
  ];

  // Monthly trend simulation (mock data)
  const months = ['1月', '2月', '3月', '4月', '5月', '6月'];
  const monthlyTrendData = [
    {
      x: months,
      y: [82, 84, 87, 89, 91, 88],
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: '分析精度',
      line: { color: '#3498DB' }
    },
    {
      x: months,
      y: [91, 93, 94, 95, 94, 95],
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: '在庫サービスレベル',
      line: { color: '#2ECC71' }
    },
    {
      x: months,
      y: [83, 85, 86, 87, 88, 87],
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: 'ルート効率',
      line: { color: '#E74C3C' }
    }
  ];

  const keyMetrics = [
    {
      title: '総分析アイテム数',
      value: summaryData.analytics.abcAnalysis.totalItems.toLocaleString(),
      unit: 'items',
      color: '#3498DB',
      icon: <AnalyticsIcon />,
      trend: '+12%'
    },
    {
      title: '在庫総価値',
      value: `¥${(summaryData.inventory.totalValue / 1000000).toFixed(1)}M`,
      unit: '',
      color: '#2ECC71',
      icon: <InventoryIcon />,
      trend: '+5.2%'
    },
    {
      title: '総最適化ルート',
      value: summaryData.routing.totalRoutes.toString(),
      unit: 'routes',
      color: '#E74C3C',
      icon: <RouteIcon />,
      trend: '-8%'
    },
    {
      title: '年間コスト削減',
      value: `¥${((summaryData.inventory.eoqOptimization.annualSavings + summaryData.routing.costSavings + 52000) / 1000).toFixed(0)}K`,
      unit: '',
      color: '#F39C12',
      icon: <TrendingUpIcon />,
      trend: '+28%'
    }
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        SCMOPT Dashboard Overview
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 3 }}>
        サプライチェーン最適化の総合パフォーマンス
      </Typography>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {keyMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                  <Box sx={{ color: metric.color }}>
                    {metric.icon}
                  </Box>
                  <Chip 
                    label={metric.trend} 
                    size="small" 
                    color={metric.trend.startsWith('+') ? 'success' : 'error'}
                  />
                </Box>
                <Typography variant="h4" sx={{ color: metric.color, mb: 0.5 }}>
                  {metric.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metric.title} {metric.unit}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Performance Overview */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                パフォーマンス概要
              </Typography>
              <Plot
                data={performanceData as any}
                layout={{
                  title: { text: '各モジュールの効率スコア' },
                  xaxis: { title: { text: 'モジュール' } },
                  yaxis: { title: { text: '効率 (%)' }, range: [0, 100] },
                  height: 350,
                  margin: { t: 50, l: 60, r: 20, b: 80 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Cost Savings Breakdown */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                年間コスト削減内訳
              </Typography>
              <Plot
                data={costSavingsData as any}
                layout={{
                  height: 350,
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

        {/* Monthly Trend */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                月次パフォーマンストレンド
              </Typography>
              <Plot
                data={monthlyTrendData as any}
                layout={{
                  title: { text: '主要指標の推移' },
                  xaxis: { title: { text: '月' } },
                  yaxis: { title: { text: 'パフォーマンス (%)' }, range: [75, 100] },
                  height: 400,
                  margin: { t: 50, l: 60, r: 20, b: 60 }
                }}
                style={{ width: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Module Summaries */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom color="primary">
              📊 分析モジュール
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>ABC分析:</strong> {summaryData.analytics.abcAnalysis.totalItems}アイテム分類済み
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>リスクプーリング:</strong> {summaryData.analytics.riskPooling.varianceReduction}%分散削減
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              <strong>パレート準拠率:</strong> {summaryData.analytics.paretoCompliance}%
            </Typography>
            <Chip label="高精度分析" color="success" size="small" />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom color="success.main">
              📦 在庫管理モジュール
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>総在庫価値:</strong> ¥{(summaryData.inventory.totalValue / 1000000).toFixed(1)}M
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>年間削減額:</strong> ¥{(summaryData.inventory.eoqOptimization.annualSavings / 1000).toFixed(0)}K
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              <strong>サービスレベル:</strong> {summaryData.inventory.serviceLevel}%
            </Typography>
            <Chip label="最適化済み" color="success" size="small" />
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom color="error.main">
              🚚 配送最適化モジュール
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>最適化ルート:</strong> {summaryData.routing.totalRoutes}ルート
            </Typography>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>CO2削減:</strong> {summaryData.routing.co2Emissions}t/年
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              <strong>燃費効率:</strong> {summaryData.routing.fuelEfficiency}%
            </Typography>
            <Chip label="eco-friendly" color="success" size="small" />
          </Paper>
        </Grid>

        {/* System Health Status */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Typography variant="h6" gutterBottom>
              システム稼働状況
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <SpeedIcon sx={{ fontSize: 40, color: '#2ECC71', mb: 1 }} />
                  <Typography variant="h6" color="success.main">99.8%</Typography>
                  <Typography variant="body2">稼働時間</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <EcoIcon sx={{ fontSize: 40, color: '#3498DB', mb: 1 }} />
                  <Typography variant="h6" color="primary">2.3t</Typography>
                  <Typography variant="body2">CO2削減/年</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <TrendingUpIcon sx={{ fontSize: 40, color: '#F39C12', mb: 1 }} />
                  <Typography variant="h6" color="warning.main">¥225K</Typography>
                  <Typography variant="body2">年間削減効果</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={3}>
                <Box sx={{ textAlign: 'center' }}>
                  <AnalyticsIcon sx={{ fontSize: 40, color: '#9B59B6', mb: 1 }} />
                  <Typography variant="h6" color="secondary">1,245</Typography>
                  <Typography variant="body2">処理済みアイテム</Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardSummary;