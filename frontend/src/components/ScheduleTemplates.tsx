import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  CardActions,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Tabs,
  Tab,
  Collapse,
  Divider,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  ContentCopy as CopyIcon,
  PlayArrow as PlayIcon,
  GetApp as DownloadIcon,
  Publish as UploadIcon,
  Refresh as RefreshIcon,
  Star as StarIcon,
  Visibility as ViewIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
  Factory as FactoryIcon,
  Business as BusinessIcon,
  Settings as SettingsIcon,
  Storage as DataIcon,
  Category as CategoryIcon,
  PlayArrow as ExecuteIcon,
  Assessment as ResultsIcon,
  Timeline as MonitorIcon,
  Policy as PolicyIcon
} from '@mui/icons-material';
import apiClient from '../services/apiClient';

// Types
interface ScheduleTemplate {
  id: string;
  name: string;
  description?: string;
  category: string;
  problem_template: any;
  default_solver_config?: any;
  default_analysis_config?: any;
  tags: string[];
  created_at: string;
  updated_at: string;
  usage_count: number;
  is_public: boolean;
  created_by?: string;
}

interface TemplateListResponse {
  templates: ScheduleTemplate[];
  total_count: number;
  page: number;
  per_page: number;
}

interface TemplateUsageStats {
  template_id: string;
  usage_count: number;
  last_used?: string;
  average_solve_time?: number;
  success_rate: number;
}

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ScheduleTemplates: React.FC = () => {
  // State management
  const [tabValue, setTabValue] = useState(0);
  const [templates, setTemplates] = useState<ScheduleTemplate[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Search and filter
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [categories, setCategories] = useState<string[]>([]);
  const [showPublicOnly, setShowPublicOnly] = useState(true);
  
  // Dialogs
  const [templateDialog, setTemplateDialog] = useState(false);
  const [viewDialog, setViewDialog] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ScheduleTemplate | null>(null);
  
  // Popular templates
  const [popularTemplates, setPopularTemplates] = useState<ScheduleTemplate[]>([]);
  
  // Form data for template creation/editing
  const [templateForm, setTemplateForm] = useState({
    name: '',
    description: '',
    category: 'general',
    tags: [] as string[],
    is_public: true
  });

  useEffect(() => {
    loadTemplates();
    loadCategories();
    loadPopularTemplates();
  }, []);

  const loadTemplates = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (searchQuery) params.append('query', searchQuery);
      if (categoryFilter) params.append('category', categoryFilter);
      if (showPublicOnly) params.append('is_public', 'true');
      params.append('limit', '50');

      const response = await apiClient.get(`/templates?${params}`);
      setTemplates(response.data.templates);
    } catch (err: any) {
      setError('テンプレート一覧の取得に失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const loadCategories = async () => {
    try {
      const response = await apiClient.get('/templates/categories/list');
      setCategories(response.data);
    } catch (err) {
      console.error('カテゴリ取得に失敗しました:', err);
    }
  };

  const loadPopularTemplates = async () => {
    try {
      const response = await apiClient.get('/templates/popular/list?limit=6');
      setPopularTemplates(response.data);
    } catch (err) {
      console.error('人気テンプレート取得に失敗しました:', err);
    }
  };

  const handleSearch = () => {
    loadTemplates();
  };

  const handleDeleteTemplate = async (template: ScheduleTemplate) => {
    if (template.created_by === 'system') {
      setError('システムテンプレートは削除できません');
      return;
    }

    try {
      await apiClient.delete(`/templates/${template.id}`);
      setSuccess('テンプレートが削除されました');
      loadTemplates();
      setDeleteDialog(false);
    } catch (err: any) {
      setError('テンプレート削除に失敗しました: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleDuplicateTemplate = async (template: ScheduleTemplate) => {
    try {
      const newName = `${template.name} (コピー)`;
      await apiClient.post(`/templates/${template.id}/duplicate?new_name=${encodeURIComponent(newName)}`);
      setSuccess('テンプレートが複製されました');
      loadTemplates();
    } catch (err: any) {
      setError('テンプレート複製に失敗しました: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handleSolveFromTemplate = async (template: ScheduleTemplate) => {
    setLoading(true);
    try {
      const response = await apiClient.post(`/templates/${template.id}/solve`);
      // 結果をJobShopSchedulingコンポーネントに渡すか、結果表示ダイアログを開く
      setSuccess(`テンプレート "${template.name}" から問題を解決しました`);
      console.log('Solution:', response.data);
    } catch (err: any) {
      setError('問題解決に失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'manufacturing': return <FactoryIcon />;
      case 'project': return <BusinessIcon />;
      case 'general': return <ScheduleIcon />;
      default: return <ScheduleIcon />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'manufacturing': return 'primary';
      case 'project': return 'secondary';
      case 'general': return 'default';
      default: return 'default';
    }
  };

  const renderTemplateCard = (template: ScheduleTemplate) => (
    <Grid item xs={12} md={6} lg={4} key={template.id}>
      <Card 
        elevation={2} 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          '&:hover': {
            elevation: 4,
            transform: 'translateY(-2px)',
            transition: 'all 0.3s ease-in-out',
          }
        }}
      >
        <CardContent sx={{ flexGrow: 1 }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
            <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
              {template.name}
            </Typography>
            {template.created_by === 'system' && (
              <Chip label="公式" size="small" color="success" />
            )}
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: 40 }}>
            {template.description || 'テンプレートの説明がありません'}
          </Typography>
          
          <Box display="flex" alignItems="center" mb={2}>
            <Chip 
              icon={getCategoryIcon(template.category)}
              label={template.category}
              color={getCategoryColor(template.category) as any}
              size="small"
              sx={{ mr: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              使用回数: {template.usage_count}
            </Typography>
          </Box>
          
          <Box display="flex" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              マシン: {template.problem_template?.machines?.length || 0}台
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ ml: 2 }}>
              ジョブ: {template.problem_template?.jobs?.length || 0}件
            </Typography>
          </Box>
          
          {template.tags.length > 0 && (
            <Box sx={{ mt: 1 }}>
              {template.tags.map((tag, index) => (
                <Chip 
                  key={index} 
                  label={tag} 
                  size="small" 
                  variant="outlined" 
                  sx={{ mr: 0.5, mb: 0.5 }}
                />
              ))}
            </Box>
          )}
        </CardContent>
        
        <CardActions>
          <Tooltip title="実行">
            <IconButton 
              color="primary"
              onClick={() => handleSolveFromTemplate(template)}
              disabled={loading}
            >
              <PlayIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="詳細表示">
            <IconButton onClick={() => {
              setSelectedTemplate(template);
              setViewDialog(true);
            }}>
              <ViewIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="複製">
            <IconButton onClick={() => handleDuplicateTemplate(template)}>
              <CopyIcon />
            </IconButton>
          </Tooltip>
          
          {template.created_by !== 'system' && (
            <Tooltip title="削除">
              <IconButton 
                color="error"
                onClick={() => {
                  setSelectedTemplate(template);
                  setDeleteDialog(true);
                }}
              >
                <DeleteIcon />
              </IconButton>
            </Tooltip>
          )}
        </CardActions>
      </Card>
    </Grid>
  );

  return (
    <Box>
      <Typography variant="h3" gutterBottom>
        スケジュールテンプレート管理
      </Typography>
      
      <Typography variant="h6" color="text.secondary" gutterBottom>
        よく使用されるスケジュール問題を保存・再利用
      </Typography>

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
          <Tab label="システム設定" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="データ管理" icon={<DataIcon />} iconPosition="start" />
          <Tab label="カテゴリ管理" icon={<CategoryIcon />} iconPosition="start" />
          <Tab label="テンプレート実行" icon={<ExecuteIcon />} iconPosition="start" />
          <Tab label="結果分析" icon={<ResultsIcon />} iconPosition="start" />
          <Tab label="リアルタイム監視" icon={<MonitorIcon />} iconPosition="start" />
          <Tab label="ポリシー管理" icon={<PolicyIcon />} iconPosition="start" />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        <Typography variant="h6" gutterBottom>
          システム設定 - スケジュールテンプレート管理
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレート管理システムの基本設定と環境構成
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  テンプレート作成
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  新しいスケジュールテンプレートを作成
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setTemplateDialog(true)}
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  新規テンプレート作成
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  システム状態
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  現在のシステム状況とパフォーマンス
                </Typography>
                <Typography variant="body2">
                  登録テンプレート数: {templates.length}<br/>
                  カテゴリ数: {categories.length}<br/>
                  平均使用回数: {templates.length > 0 ? Math.round(templates.reduce((sum, t) => sum + t.usage_count, 0) / templates.length) : 0}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  システム設定
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  テンプレート管理の全般設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="デフォルトカテゴリ"
                      value="general"
                      disabled
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="最大テンプレート数"
                      value="1000"
                      disabled
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="キャッシュ有効期限"
                      value="24時間"
                      disabled
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6" gutterBottom>
          データ管理
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレートのインポート・エクスポート・検索機能
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  検索・フィルタ
                </Typography>
                <TextField
                  fullWidth
                  size="small"
                  label="テンプレート検索"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  sx={{ mb: 2 }}
                  InputProps={{
                    startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                  }}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                />
                <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                  <InputLabel>カテゴリ</InputLabel>
                  <Select
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    label="カテゴリ"
                  >
                    <MenuItem value="">すべて</MenuItem>
                    {categories.map((cat) => (
                      <MenuItem key={cat} value={cat}>{cat}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    onClick={handleSearch}
                    startIcon={<SearchIcon />}
                    disabled={loading}
                    size="small"
                  >
                    検索
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={loadTemplates}
                    startIcon={<RefreshIcon />}
                    disabled={loading}
                    size="small"
                  >
                    更新
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  データ操作
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  テンプレートのインポート・エクスポート
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<UploadIcon />}
                    fullWidth
                  >
                    テンプレートをインポート
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    fullWidth
                  >
                    テンプレートをエクスポート
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  テンプレート一覧 ({templates.length}件)
                </Typography>
                {loading ? (
                  <Box display="flex" justifyContent="center" p={4}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <Grid container spacing={2}>
                    {templates.slice(0, 6).map(renderTemplateCard)}
                  </Grid>
                )}
                
                {templates.length === 0 && !loading && (
                  <Box sx={{ p: 4, textAlign: 'center' }}>
                    <Typography variant="h6" color="text.secondary">
                      テンプレートが見つかりませんでした
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      検索条件を変更するか、新しいテンプレートを作成してください
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          カテゴリ管理
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレートカテゴリの管理と分類
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  カテゴリ一覧
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  {categories.map((category) => (
                    <Chip 
                      key={category} 
                      label={category} 
                      variant="outlined"
                      onClick={() => setCategoryFilter(category)}
                    />
                  ))}
                </Box>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  size="small"
                >
                  新しいカテゴリを追加
                </Button>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  人気テンプレート
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  よく使用されているテンプレート
                </Typography>
                <Grid container spacing={1}>
                  {popularTemplates.slice(0, 3).map(renderTemplateCard)}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <Typography variant="h6" gutterBottom>
          テンプレート実行
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          選択したテンプレートでスケジュール最適化を実行
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  実行可能テンプレート
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  テンプレートを選択して最適化を実行
                </Typography>
                
                {templates.length > 0 ? (
                  <Grid container spacing={2}>
                    {templates.slice(0, 6).map((template) => (
                      <Grid item xs={12} md={4} key={template.id}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {template.name}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              {template.description || 'テンプレートの説明'}
                            </Typography>
                            <Chip label={template.category} size="small" sx={{ mb: 2 }} />
                            <Box>
                              <Button
                                variant="contained"
                                startIcon={<PlayIcon />}
                                size="small"
                                fullWidth
                              >
                                実行
                              </Button>
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                ) : (
                  <Alert severity="info">
                    実行可能なテンプレートがありません。まずテンプレートを作成してください。
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        <Typography variant="h6" gutterBottom>
          結果分析
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレート実行結果の分析と比較
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', bgcolor: 'success.light' }}>
                <Typography variant="h4" color="white">
                  {templates.filter(t => t.usage_count > 0).length}
                </Typography>
                <Typography variant="subtitle2" color="white">使用済みテンプレート</Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', bgcolor: 'info.light' }}>
                <Typography variant="h4" color="white">
                  {templates.reduce((sum, t) => sum + t.usage_count, 0)}
                </Typography>
                <Typography variant="subtitle2" color="white">総実行回数</Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', bgcolor: 'warning.light' }}>
                <Typography variant="h4" color="white">
                  {Math.round((templates.filter(t => t.usage_count > 0).length / Math.max(templates.length, 1)) * 100)}%
                </Typography>
                <Typography variant="subtitle2" color="white">活用率</Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  パフォーマンス分析
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  テンプレートの使用頻度と効果の詳細分析
                </Typography>
                <Button
                  variant="outlined"
                  sx={{ mt: 2 }}
                  disabled
                >
                  詳細レポート生成（実装予定）
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={5}>
        <Typography variant="h6" gutterBottom>
          リアルタイム監視
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレート実行状況のリアルタイム監視
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  実行監視ダッシュボード
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  現在実行中のテンプレート処理状況
                </Typography>
                
                <Alert severity="info">
                  現在実行中のテンプレートはありません。
                </Alert>
                
                <Box sx={{ mt: 2 }}>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                  >
                    状況を更新
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={6}>
        <Typography variant="h6" gutterBottom>
          ポリシー管理
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          テンプレート管理ポリシーの設定とエクスポート
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  エクスポート・レポート
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    disabled={templates.length === 0}
                  >
                    全テンプレートをエクスポート
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    disabled={templates.length === 0}
                  >
                    使用統計レポート
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    disabled={categories.length === 0}
                  >
                    カテゴリ分析レポート
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  管理ポリシー
                </Typography>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  テンプレート管理の運用ポリシー
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="outlined"
                    disabled
                  >
                    ポリシー設定を保存
                  </Button>
                  <Button
                    variant="outlined"
                    disabled
                  >
                    ポリシー設定を読み込み
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Template Details Dialog */}
      <Dialog 
        open={viewDialog} 
        onClose={() => setViewDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          テンプレート詳細: {selectedTemplate?.name}
        </DialogTitle>
        <DialogContent>
          {selectedTemplate && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>基本情報</Typography>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>名前</TableCell>
                      <TableCell>{selectedTemplate.name}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>カテゴリ</TableCell>
                      <TableCell>{selectedTemplate.category}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>使用回数</TableCell>
                      <TableCell>{selectedTemplate.usage_count}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>作成日</TableCell>
                      <TableCell>{new Date(selectedTemplate.created_at).toLocaleDateString('ja-JP')}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>問題設定</Typography>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>マシン数</TableCell>
                      <TableCell>{selectedTemplate.problem_template?.machines?.length || 0}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>ジョブ数</TableCell>
                      <TableCell>{selectedTemplate.problem_template?.jobs?.length || 0}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>問題タイプ</TableCell>
                      <TableCell>{selectedTemplate.problem_template?.problem_type || 'N/A'}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>最適化目標</TableCell>
                      <TableCell>{selectedTemplate.problem_template?.optimization_objective || 'N/A'}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>説明</Typography>
                <Typography variant="body2">
                  {selectedTemplate.description || 'テンプレートの説明がありません'}
                </Typography>
              </Grid>
              
              {selectedTemplate.tags.length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>タグ</Typography>
                  <Box>
                    {selectedTemplate.tags.map((tag, index) => (
                      <Chip key={index} label={tag} size="small" sx={{ mr: 0.5 }} />
                    ))}
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialog(false)}>閉じる</Button>
          {selectedTemplate && (
            <Button 
              variant="contained"
              onClick={() => {
                handleSolveFromTemplate(selectedTemplate);
                setViewDialog(false);
              }}
              startIcon={<PlayIcon />}
            >
              実行
            </Button>
          )}
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle>テンプレート削除</DialogTitle>
        <DialogContent>
          <Typography>
            テンプレート "{selectedTemplate?.name}" を削除してもよろしいですか？
            この操作は取り消せません。
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>キャンセル</Button>
          <Button 
            color="error" 
            onClick={() => selectedTemplate && handleDeleteTemplate(selectedTemplate)}
          >
            削除
          </Button>
        </DialogActions>
      </Dialog>

      {/* Error and Success Messages */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}
    </Box>
  );
};

export default ScheduleTemplates;