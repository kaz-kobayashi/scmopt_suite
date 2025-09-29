import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  LinearProgress,
  Divider,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  BarChart as AnalyticsIcon,
  Storage as InventoryIcon,
  DirectionsCar as RouteIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  LocalShipping as EcoIcon,
  Factory as FactoryIcon,
  Schedule as ScheduleIcon,
  Description as TemplateIcon,
  Assignment as AdvancedVRPIcon,
  Dashboard as DashboardIcon,
  Assessment as KPIIcon,
  Timeline as TimelineIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import DashboardSummary from './DashboardSummary';

interface DashboardProps {
  onNavigate: (section: string) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [kpiData, setKpiData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // KPIデータを取得
  useEffect(() => {
    const fetchKPIData = async () => {
      try {
        setLoading(true);
        // シミュレートされたKPIデータ（実際のAPIに置き換え可能）
        await new Promise(resolve => setTimeout(resolve, 1000)); // ローディング演出
        
        const mockKPIData = {
          systemStatus: {
            uptime: '99.9%',
            activeUsers: 42,
            processedJobs: 1247,
            status: 'healthy'
          },
          inventoryMetrics: {
            totalItems: 15420,
            turnoverRate: 8.2,
            stockoutRisk: 12,
            optimalStock: 89
          },
          routingMetrics: {
            avgDeliveryTime: 2.3,
            fuelEfficiency: 85,
            co2Reduction: 15.7,
            routeOptimization: 94
          },
          operationalMetrics: {
            costSavings: 1234567,
            efficiencyGain: 23.5,
            errorReduction: 67.8,
            customerSatisfaction: 92
          },
          recentActivities: [
            { type: 'optimization', description: 'VRP最適化完了 - 15%コスト削減', timestamp: '10分前', status: 'success' },
            { type: 'analysis', description: 'ABC分析レポート生成完了', timestamp: '1時間前', status: 'success' },
            { type: 'alert', description: '在庫レベル警告 - 商品A123', timestamp: '2時間前', status: 'warning' },
            { type: 'schedule', description: 'シフト最適化スケジュール更新', timestamp: '3時間前', status: 'success' }
          ]
        };
        
        setKpiData(mockKPIData);
      } catch (err) {
        setError('KPIデータの取得に失敗しました');
      } finally {
        setLoading(false);
      }
    };

    fetchKPIData();
  }, []);

  const features = [
    {
      title: 'ABC分析',
      description: 'パレート分析と在庫分類',
      icon: <AnalyticsIcon />,
      section: 'analytics',
    },
    {
      title: 'EOQ計算',
      description: '経済的発注量の最適化',
      icon: <InventoryIcon />,
      section: 'inventory',
    },
    {
      title: 'CO2排出量',
      description: '輸送時の排出量計算',
      icon: <EcoIcon />,
      section: 'routing',
    },
    {
      title: '距離行列',
      description: '地点間の距離計算',
      icon: <RouteIcon />,
      section: 'routing',
    },
    {
      title: '在庫シミュレーション',
      description: 'モンテカルロ法による在庫ポリシーシミュレーション',
      icon: <SpeedIcon />,
      section: 'inventory',
    },
    {
      title: 'リスクプーリング',
      description: '需要集約とリスク分析',
      icon: <TrendingUpIcon />,
      section: 'analytics',
    },
    {
      title: 'ジョブショップスケジューリング',
      description: 'PyJobShopを使用した生産スケジュール最適化',
      icon: <FactoryIcon />,
      section: 'jobshop',
    },
    {
      title: 'シフト最適化',
      description: '勤務シフトスケジューリングの最適化',
      icon: <ScheduleIcon />,
      section: 'shift',
    },
    {
      title: 'スケジュールテンプレート',
      description: '頻繁に使用するスケジュール設定の保存・管理',
      icon: <TemplateIcon />,
      section: 'templates',
    },
    {
      title: '高度な配送計画 (PyVRP)',
      description: 'CVRP、VRPTW、MDVRP、PDVRP、PC-VRPの最適化',
      icon: <AdvancedVRPIcon />,
      section: 'advanced-vrp',
    },
    {
      title: 'PyVRP完全仕様',
      description: 'クライアントグループ、リロード機能、複数ルーティングプロファイル対応',
      icon: <AdvancedVRPIcon />,
      section: 'advanced-vrp-full',
    },
  ];

  const quickStats = [
    { label: 'アクティブモジュール', value: '10', color: '#2196f3' },
    { label: 'APIエンドポイント', value: '20+', color: '#ff9800' },
    { label: '分析タイプ', value: '4', color: '#4caf50' },
    { label: '最適化手法', value: '12', color: '#9c27b0' },
  ];

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          ダッシュボードデータを読み込み中...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 4 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Typography variant="h3" gutterBottom sx={{ mb: 4, fontWeight: 'bold' }}>
        SCMOPT2 - ダッシュボード
      </Typography>

      {/* System Status */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h6" color="primary">
                    稼働率
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {kpiData?.systemStatus.uptime}
                  </Typography>
                </Box>
                <CheckIcon sx={{ fontSize: 40, color: 'green' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h6" color="primary">
                    アクティブユーザー
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {kpiData?.systemStatus.activeUsers}
                  </Typography>
                </Box>
                <TrendingUpIcon sx={{ fontSize: 40, color: 'blue' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h6" color="primary">
                    処理済みジョブ
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {kpiData?.systemStatus.processedJobs?.toLocaleString()}
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, color: 'orange' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography variant="h6" color="primary">
                    コスト削減
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    ¥{kpiData?.operationalMetrics.costSavings?.toLocaleString()}
                  </Typography>
                </Box>
                <TrendingDownIcon sx={{ fontSize: 40, color: 'green' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* KPI Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                在庫管理KPI
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">総在庫アイテム数</Typography>
                  <Typography variant="h6">{kpiData?.inventoryMetrics.totalItems?.toLocaleString()}</Typography>
                </Box>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">在庫回転率</Typography>
                  <Typography variant="h6">{kpiData?.inventoryMetrics.turnoverRate}回/年</Typography>
                </Box>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>最適在庫水準達成率</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={kpiData?.inventoryMetrics.optimalStock} 
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" sx={{ mt: 1 }}>
                  {kpiData?.inventoryMetrics.optimalStock}%
                </Typography>
              </Box>
              <Chip 
                label={`欠品リスク: ${kpiData?.inventoryMetrics.stockoutRisk}%`}
                color={kpiData?.inventoryMetrics.stockoutRisk < 15 ? "success" : "warning"}
                size="small"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                配送最適化KPI
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="body2">平均配送時間</Typography>
                  <Typography variant="h6">{kpiData?.routingMetrics.avgDeliveryTime}日</Typography>
                </Box>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>燃費効率</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={kpiData?.routingMetrics.fuelEfficiency} 
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" sx={{ mt: 1 }}>
                  {kpiData?.routingMetrics.fuelEfficiency}%
                </Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>ルート最適化率</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={kpiData?.routingMetrics.routeOptimization} 
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" sx={{ mt: 1 }}>
                  {kpiData?.routingMetrics.routeOptimization}%
                </Typography>
              </Box>
              <Chip 
                label={`CO2削減: ${kpiData?.routingMetrics.co2Reduction}%`}
                color="success"
                size="small"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activities */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                最近のアクティビティ
              </Typography>
              <List>
                {kpiData?.recentActivities.map((activity: any, index: number) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {activity.status === 'success' && <CheckIcon color="success" />}
                        {activity.status === 'warning' && <WarningIcon color="warning" />}
                        {activity.status === 'error' && <ErrorIcon color="error" />}
                      </ListItemIcon>
                      <ListItemText
                        primary={activity.description}
                        secondary={activity.timestamp}
                      />
                      <Chip 
                        label={activity.type}
                        size="small"
                        variant="outlined"
                      />
                    </ListItem>
                    {index < kpiData.recentActivities.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                クイックアクション
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button 
                  variant="contained" 
                  startIcon={<AnalyticsIcon />}
                  onClick={() => onNavigate('analytics')}
                  fullWidth
                >
                  ABC分析実行
                </Button>
                <Button 
                  variant="contained" 
                  startIcon={<InventoryIcon />}
                  onClick={() => onNavigate('inventory')}
                  fullWidth
                >
                  EOQ計算
                </Button>
                <Button 
                  variant="contained" 
                  startIcon={<RouteIcon />}
                  onClick={() => onNavigate('routing')}
                  fullWidth
                >
                  ルート最適化
                </Button>
                <Button 
                  variant="contained" 
                  startIcon={<FactoryIcon />}
                  onClick={() => onNavigate('jobshop')}
                  fullWidth
                >
                  スケジューリング
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Access to Features */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        機能一覧
      </Typography>
      <Grid container spacing={2}>
        {features.map((feature, index) => (
          <Grid item key={index} xs={6} md={4} lg={3}>
            <Card 
              elevation={1} 
              sx={{ 
                cursor: 'pointer',
                '&:hover': {
                  elevation: 3,
                  transform: 'translateY(-2px)',
                  transition: 'all 0.3s ease-in-out',
                },
              }}
              onClick={() => onNavigate(feature.section)}
            >
              <CardContent sx={{ py: 2 }}>
                <Box display="flex" alignItems="center">
                  <Box sx={{ color: '#2196f3', mr: 1 }}>
                    {feature.icon}
                  </Box>
                  <Typography variant="body2" component="div">
                    {feature.title}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Dashboard;