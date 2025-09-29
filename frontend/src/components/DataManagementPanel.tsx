import React, { useState, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  IconButton,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  PictureAsPdf as PdfIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Save as SaveIcon,
  Undo as UndoIcon,
  CloudDownload as CloudDownloadIcon,
  Assessment as ReportIcon,
  TableChart as TableIcon,
  Route as RouteIcon,
} from '@mui/icons-material';

// Data import/export interfaces
interface ImportResult {
  success: boolean;
  data?: any;
  errors?: string[];
  warnings?: string[];
  summary?: {
    clients_imported: number;
    vehicles_imported: number;
    depots_imported: number;
  };
}

interface ExportOptions {
  format: 'csv' | 'excel' | 'json';
  include_routes: boolean;
  include_summary: boolean;
  include_constraints: boolean;
}

interface ReportOptions {
  include_map: boolean;
  include_schedule: boolean;
  include_costs: boolean;
  include_statistics: boolean;
  language: 'ja' | 'en';
}

interface DataManagementPanelProps {
  problemData: any;
  solution: any;
  onProblemDataChange: (data: any) => void;
  onError: (error: string) => void;
  onSuccess: (message: string) => void;
}

const DataManagementPanel: React.FC<DataManagementPanelProps> = ({
  problemData,
  solution,
  onProblemDataChange,
  onError,
  onSuccess
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [importing, setImporting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [generatingReport, setGeneratingReport] = useState(false);
  
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'excel',
    include_routes: true,
    include_summary: true,
    include_constraints: false
  });
  const [reportOptions, setReportOptions] = useState<ReportOptions>({
    include_map: true,
    include_schedule: true,
    include_costs: true,
    include_statistics: true,
    language: 'ja'
  });

  // Edit dialog states
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingType, setEditingType] = useState<'clients' | 'vehicles' | 'depots' | null>(null);
  const [editingIndex, setEditingIndex] = useState<number>(-1);
  const [editingData, setEditingData] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // File import functionality
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setImporting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Try to import as complete problem data first
      const response = await fetch('/api/data/import', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // If that fails, try to determine the data type from filename or content
        const filename = file.name.toLowerCase();
        let dataType = '';
        
        if (filename.includes('client') || filename.includes('顧客')) {
          dataType = 'clients';
        } else if (filename.includes('vehicle') || filename.includes('車両')) {
          dataType = 'vehicles';
        } else if (filename.includes('depot') || filename.includes('デポ')) {
          dataType = 'depots';
        }
        
        if (dataType) {
          // Reset FormData and re-append the file for the second request
          const newFormData = new FormData();
          newFormData.append('file', file);
          
          // Try importing as specific data type
          const specificResponse = await fetch(`/api/data/import/vrp/${dataType}`, {
            method: 'POST',
            body: newFormData,
          });
          
          if (specificResponse.ok) {
            const data = await specificResponse.json();
            const newProblemData = { ...problemData };
            
            if (dataType === 'clients') {
              newProblemData.clients = data;
              onProblemDataChange(newProblemData);
              onSuccess(`顧客データのインポートが完了しました: ${data.length}件`);
            } else if (dataType === 'vehicles') {
              newProblemData.vehicle_types = data;
              onProblemDataChange(newProblemData);
              onSuccess(`車両データのインポートが完了しました: ${data.length}タイプ`);
            } else if (dataType === 'depots') {
              newProblemData.depots = data;
              onProblemDataChange(newProblemData);
              onSuccess(`デポデータのインポートが完了しました: ${data.length}件`);
            }
            
            setImportResult({
              success: true,
              summary: {
                clients_imported: dataType === 'clients' ? data.length : 0,
                vehicles_imported: dataType === 'vehicles' ? data.length : 0,
                depots_imported: dataType === 'depots' ? data.length : 0
              }
            });
            return;
          }
        }
        
        throw new Error(`Import failed: ${response.status}`);
      }

      const result: ImportResult = await response.json();
      setImportResult(result);

      if (result.success) {
        onProblemDataChange(result.data);
        onSuccess(`データのインポートが完了しました。顧客: ${result.summary?.clients_imported || 0}件、車両: ${result.summary?.vehicles_imported || 0}件、デポ: ${result.summary?.depots_imported || 0}件`);
      } else {
        onError(`インポートエラー: ${result.errors?.join(', ')}`);
      }
    } catch (error: any) {
      onError(`ファイルのインポートに失敗しました: ${error.message}`);
    } finally {
      setImporting(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // CSV template download from backend
  const downloadTemplate = async (type: 'clients' | 'vehicles' | 'depots') => {
    try {
      const response = await fetch(`/api/data/templates/vrp/${type}/csv`);
      
      if (!response.ok) {
        throw new Error(`Failed to download template: ${response.status}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `vrp_${type}_template.csv`;
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      onSuccess(`${type === 'clients' ? '顧客' : type === 'vehicles' ? '車両' : 'デポ'}データテンプレートをダウンロードしました`);
    } catch (error: any) {
      onError(`テンプレートのダウンロードに失敗しました: ${error.message}`);
    }
  };

  // Data export functionality
  const handleExport = async () => {
    setExporting(true);
    try {
      const exportData = {
        problemData,
        solution: exportOptions.include_routes ? solution : undefined,
        options: exportOptions
      };

      const response = await fetch('/api/data/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData),
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      const extension = exportOptions.format === 'excel' ? 'xlsx' : 
                       exportOptions.format === 'csv' ? 'csv' : 'json';
      link.download = `vrp_data_${new Date().toISOString().split('T')[0]}.${extension}`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      onSuccess('データのエクスポートが完了しました');
      setExportDialogOpen(false);
    } catch (error: any) {
      onError(`エクスポートに失敗しました: ${error.message}`);
    } finally {
      setExporting(false);
    }
  };

  // PDF report generation
  const generateReport = async () => {
    setGeneratingReport(true);
    try {
      const reportData = {
        problemData,
        solution,
        options: reportOptions
      };

      const response = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData),
      });

      if (!response.ok) {
        throw new Error(`Report generation failed: ${response.status}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `vrp_report_${new Date().toISOString().split('T')[0]}.pdf`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      onSuccess('PDFレポートが生成されました');
      setReportDialogOpen(false);
    } catch (error: any) {
      onError(`レポート生成に失敗しました: ${error.message}`);
    } finally {
      setGeneratingReport(false);
    }
  };

  // Manual data editing
  const handleManualEdit = (type: 'clients' | 'vehicles' | 'depots', index: number) => {
    setEditingType(type);
    setEditingIndex(index);
    
    let dataToEdit = null;
    if (type === 'clients') {
      dataToEdit = { ...problemData.clients[index] };
    } else if (type === 'vehicles') {
      dataToEdit = { ...problemData.vehicle_types[index] };
    } else if (type === 'depots') {
      dataToEdit = { ...problemData.depots[index] };
    }
    
    setEditingData(dataToEdit);
    setEditDialogOpen(true);
  };

  // Save edited data
  const handleSaveEdit = () => {
    if (!editingType || editingIndex === -1 || !editingData) return;

    const newProblemData = { ...problemData };
    
    if (editingType === 'clients') {
      newProblemData.clients[editingIndex] = editingData;
    } else if (editingType === 'vehicles') {
      newProblemData.vehicle_types[editingIndex] = editingData;
    } else if (editingType === 'depots') {
      newProblemData.depots[editingIndex] = editingData;
    }
    
    onProblemDataChange(newProblemData);
    onSuccess(`${editingType === 'clients' ? '顧客' : editingType === 'vehicles' ? '車両' : 'デポ'}データを更新しました`);
    handleCloseEdit();
  };

  // Close edit dialog
  const handleCloseEdit = () => {
    setEditDialogOpen(false);
    setEditingType(null);
    setEditingIndex(-1);
    setEditingData(null);
  };

  // Update editing data field
  const updateEditingField = (field: string, value: any) => {
    setEditingData({
      ...editingData,
      [field]: value
    });
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <TableIcon color="primary" />
        <Typography variant="h6">データ管理機能</Typography>
      </Box>

      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} sx={{ mb: 2 }}>
        <Tab label="インポート/エクスポート" />
        <Tab label="データ編集" />
        <Tab label="レポート生成" />
      </Tabs>

      {/* Import/Export Tab */}
      {tabValue === 0 && (
        <Box>
          {/* Import Section */}
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">データインポート</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    ファイルからインポート
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                    <Button
                      variant="contained"
                      component="label"
                      startIcon={<UploadIcon />}
                      disabled={importing}
                    >
                      {importing ? 'インポート中...' : 'ファイル選択'}
                      <input
                        ref={fileInputRef}
                        type="file"
                        hidden
                        accept=".csv,.xlsx,.json"
                        onChange={handleFileUpload}
                      />
                    </Button>
                    {importing && <CircularProgress size={24} />}
                  </Box>
                  <Typography variant="caption" color="textSecondary">
                    対応形式: CSV, Excel (.xlsx), JSON
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    テンプレートダウンロード
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button
                      size="small"
                      onClick={() => downloadTemplate('clients')}
                      startIcon={<CloudDownloadIcon />}
                    >
                      顧客データテンプレート
                    </Button>
                    <Button
                      size="small"
                      onClick={() => downloadTemplate('vehicles')}
                      startIcon={<CloudDownloadIcon />}
                    >
                      車両データテンプレート
                    </Button>
                    <Button
                      size="small"
                      onClick={() => downloadTemplate('depots')}
                      startIcon={<CloudDownloadIcon />}
                    >
                      デポデータテンプレート
                    </Button>
                  </Box>
                </Grid>
              </Grid>

              {importResult && (
                <Box sx={{ mt: 2 }}>
                  <Alert severity={importResult.success ? 'success' : 'error'}>
                    <Typography variant="subtitle2">
                      インポート結果
                    </Typography>
                    {importResult.success ? (
                      <Box>
                        <Typography variant="body2">
                          • 顧客: {importResult.summary?.clients_imported || 0}件
                        </Typography>
                        <Typography variant="body2">
                          • 車両: {importResult.summary?.vehicles_imported || 0}タイプ
                        </Typography>
                        <Typography variant="body2">
                          • デポ: {importResult.summary?.depots_imported || 0}件
                        </Typography>
                        {(importResult.summary?.clients_imported || 0) > 0 && 
                         (importResult.summary?.vehicles_imported || 0) > 0 && 
                         (importResult.summary?.depots_imported || 0) > 0 && (
                          <Typography variant="body2" sx={{ mt: 1, fontWeight: 'bold' }}>
                            ✓ すべてのデータが揃いました。「SOLVE PROBLEM」ボタンで最適化を実行できます。
                          </Typography>
                        )}
                      </Box>
                    ) : (
                      <Box>
                        {importResult.errors?.map((error, index) => (
                          <Typography key={index} variant="body2" color="error">
                            • {error}
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </Alert>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>

          {/* Export Section */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">データエクスポート</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <Button
                  variant="contained"
                  onClick={() => setExportDialogOpen(true)}
                  startIcon={<DownloadIcon />}
                >
                  データエクスポート
                </Button>
                <Button
                  variant="outlined"
                  onClick={() => setReportDialogOpen(true)}
                  startIcon={<PdfIcon />}
                >
                  PDFレポート生成
                </Button>
              </Box>
              <Typography variant="caption" color="textSecondary">
                問題データ、解決結果、分析レポートをエクスポートできます
              </Typography>
            </AccordionDetails>
          </Accordion>
        </Box>
      )}

      {/* Data Editing Tab */}
      {tabValue === 1 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            手動データ編集
          </Typography>
          
          {/* Clients Table */}
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>顧客データ ({problemData?.clients?.length || 0}件)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400, overflow: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ minWidth: 50 }}>ID</TableCell>
                      <TableCell sx={{ minWidth: 150 }}>座標 (X, Y)</TableCell>
                      <TableCell sx={{ minWidth: 80 }}>配送量</TableCell>
                      <TableCell sx={{ minWidth: 80 }}>集荷量</TableCell>
                      <TableCell sx={{ minWidth: 100 }}>サービス時間</TableCell>
                      <TableCell sx={{ minWidth: 150 }}>時間窓</TableCell>
                      <TableCell sx={{ minWidth: 80 }}>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {problemData?.clients?.map((client: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>{index}</TableCell>
                        <TableCell>({client.x}, {client.y})</TableCell>
                        <TableCell>{client.delivery}</TableCell>
                        <TableCell>{client.pickup || 0}</TableCell>
                        <TableCell>{client.service_duration || 10}分</TableCell>
                        <TableCell>
                          {client.tw_early && client.tw_late ? (
                            `${Math.floor(client.tw_early/60).toString().padStart(2,'0')}:${(client.tw_early%60).toString().padStart(2,'0')} - ${Math.floor(client.tw_late/60).toString().padStart(2,'0')}:${(client.tw_late%60).toString().padStart(2,'0')}`
                          ) : (
                            '制限なし'
                          )}
                        </TableCell>
                        <TableCell>
                          <IconButton
                            size="small"
                            onClick={() => handleManualEdit('clients', index)}
                          >
                            <EditIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>

          {/* Vehicles Table */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>車両データ ({problemData?.vehicle_types?.length || 0}タイプ)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300, overflow: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ minWidth: 60 }}>タイプ</TableCell>
                      <TableCell sx={{ minWidth: 60 }}>台数</TableCell>
                      <TableCell sx={{ minWidth: 80 }}>容量</TableCell>
                      <TableCell sx={{ minWidth: 100 }}>開始デポ</TableCell>
                      <TableCell sx={{ minWidth: 120 }}>固定コスト</TableCell>
                      <TableCell sx={{ minWidth: 80 }}>操作</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {problemData?.vehicle_types?.map((vehicle: any, index: number) => (
                      <TableRow key={index}>
                        <TableCell>{index}</TableCell>
                        <TableCell>{vehicle.num_available}</TableCell>
                        <TableCell>{vehicle.capacity}</TableCell>
                        <TableCell>{vehicle.start_depot}</TableCell>
                        <TableCell>¥{vehicle.fixed_cost?.toLocaleString() || 0}</TableCell>
                        <TableCell>
                          <IconButton
                            size="small"
                            onClick={() => handleManualEdit('vehicles', index)}
                          >
                            <EditIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        </Box>
      )}

      {/* Report Generation Tab */}
      {tabValue === 2 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            レポート・分析機能
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }} variant="outlined">
                <Typography variant="subtitle1" gutterBottom>
                  <ReportIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  レポート生成
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="配送計画レポート"
                      secondary="ルート、コスト、統計情報を含む総合レポート"
                    />
                    <ListItemSecondaryAction>
                      <Button
                        size="small"
                        onClick={() => setReportDialogOpen(true)}
                        disabled={!solution}
                      >
                        生成
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="コスト分析レポート"
                      secondary="車両コスト、距離コストの詳細分析"
                    />
                    <ListItemSecondaryAction>
                      <Button size="small" disabled={!solution}>
                        生成
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="効率性分析"
                      secondary="車両利用率、時間効率の分析"
                    />
                    <ListItemSecondaryAction>
                      <Button size="small" disabled={!solution}>
                        生成
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                </List>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }} variant="outlined">
                <Typography variant="subtitle1" gutterBottom>
                  <RouteIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  ルート調整
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="手動ルート調整"
                      secondary="ドラッグ&ドロップでルートを手動調整"
                    />
                    <ListItemSecondaryAction>
                      <Button size="small" disabled={!solution}>
                        開始
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="制約違反チェック"
                      secondary="現在のルートの制約違反を確認"
                    />
                    <ListItemSecondaryAction>
                      <Button size="small" disabled={!solution}>
                        チェック
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemText
                      primary="ルート比較"
                      secondary="複数の解を比較分析"
                    />
                    <ListItemSecondaryAction>
                      <Button size="small" disabled>
                        準備中
                      </Button>
                    </ListItemSecondaryAction>
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      )}

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>データエクスポート設定</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>出力形式</InputLabel>
                <Select
                  value={exportOptions.format}
                  onChange={(e) => setExportOptions({
                    ...exportOptions,
                    format: e.target.value as any
                  })}
                >
                  <MenuItem value="excel">Excel (.xlsx)</MenuItem>
                  <MenuItem value="csv">CSV</MenuItem>
                  <MenuItem value="json">JSON</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2">含める内容:</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
                <label>
                  <input
                    type="checkbox"
                    checked={exportOptions.include_routes}
                    onChange={(e) => setExportOptions({
                      ...exportOptions,
                      include_routes: e.target.checked
                    })}
                  />
                  ルート情報
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={exportOptions.include_summary}
                    onChange={(e) => setExportOptions({
                      ...exportOptions,
                      include_summary: e.target.checked
                    })}
                  />
                  サマリー情報
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={exportOptions.include_constraints}
                    onChange={(e) => setExportOptions({
                      ...exportOptions,
                      include_constraints: e.target.checked
                    })}
                  />
                  制約情報
                </label>
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>キャンセル</Button>
          <Button
            onClick={handleExport}
            variant="contained"
            disabled={exporting}
            startIcon={exporting ? <CircularProgress size={16} /> : <DownloadIcon />}
          >
            {exporting ? 'エクスポート中...' : 'エクスポート'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Report Dialog */}
      <Dialog open={reportDialogOpen} onClose={() => setReportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>PDFレポート生成設定</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>言語</InputLabel>
                <Select
                  value={reportOptions.language}
                  onChange={(e) => setReportOptions({
                    ...reportOptions,
                    language: e.target.value as any
                  })}
                >
                  <MenuItem value="ja">日本語</MenuItem>
                  <MenuItem value="en">English</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="subtitle2">含める内容:</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
                <label>
                  <input
                    type="checkbox"
                    checked={reportOptions.include_map}
                    onChange={(e) => setReportOptions({
                      ...reportOptions,
                      include_map: e.target.checked
                    })}
                  />
                  地図表示
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={reportOptions.include_schedule}
                    onChange={(e) => setReportOptions({
                      ...reportOptions,
                      include_schedule: e.target.checked
                    })}
                  />
                  配送スケジュール
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={reportOptions.include_costs}
                    onChange={(e) => setReportOptions({
                      ...reportOptions,
                      include_costs: e.target.checked
                    })}
                  />
                  コスト分析
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={reportOptions.include_statistics}
                    onChange={(e) => setReportOptions({
                      ...reportOptions,
                      include_statistics: e.target.checked
                    })}
                  />
                  統計情報
                </label>
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDialogOpen(false)}>キャンセル</Button>
          <Button
            onClick={generateReport}
            variant="contained"
            disabled={generatingReport}
            startIcon={generatingReport ? <CircularProgress size={16} /> : <PdfIcon />}
          >
            {generatingReport ? '生成中...' : 'PDFレポート生成'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Data Dialog */}
      <Dialog open={editDialogOpen} onClose={handleCloseEdit} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingType === 'clients' ? '顧客データ編集' : 
           editingType === 'vehicles' ? '車両データ編集' : 'デポデータ編集'}
        </DialogTitle>
        <DialogContent>
          {editingData && editingType === 'clients' && (
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="X座標"
                  type="number"
                  value={editingData.x || 0}
                  onChange={(e) => updateEditingField('x', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Y座標"
                  type="number"
                  value={editingData.y || 0}
                  onChange={(e) => updateEditingField('y', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="配送量"
                  type="number"
                  value={editingData.delivery || 0}
                  onChange={(e) => updateEditingField('delivery', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="集荷量"
                  type="number"
                  value={editingData.pickup || 0}
                  onChange={(e) => updateEditingField('pickup', parseInt(e.target.value) || undefined)}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="サービス時間 (分)"
                  type="number"
                  value={editingData.service_duration || 10}
                  onChange={(e) => updateEditingField('service_duration', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="時間窓開始 (分)"
                  type="number"
                  value={editingData.tw_early || 480}
                  onChange={(e) => updateEditingField('tw_early', parseInt(e.target.value))}
                  fullWidth
                  helperText="0 = 00:00, 480 = 08:00"
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  label="時間窓終了 (分)"
                  type="number"
                  value={editingData.tw_late || 1080}
                  onChange={(e) => updateEditingField('tw_late', parseInt(e.target.value))}
                  fullWidth
                  helperText="1080 = 18:00"
                />
              </Grid>
            </Grid>
          )}

          {editingData && editingType === 'vehicles' && (
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="使用可能台数"
                  type="number"
                  value={editingData.num_available || 1}
                  onChange={(e) => updateEditingField('num_available', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="容量"
                  type="number"
                  value={editingData.capacity || 100}
                  onChange={(e) => updateEditingField('capacity', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="開始デポ"
                  type="number"
                  value={editingData.start_depot || 0}
                  onChange={(e) => updateEditingField('start_depot', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="終了デポ"
                  type="number"
                  value={editingData.end_depot || editingData.start_depot || 0}
                  onChange={(e) => updateEditingField('end_depot', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="固定コスト (円)"
                  type="number"
                  value={editingData.fixed_cost || 0}
                  onChange={(e) => updateEditingField('fixed_cost', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="最大稼働時間 (分)"
                  type="number"
                  value={editingData.max_duration || 480}
                  onChange={(e) => updateEditingField('max_duration', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="時間窓開始 (分)"
                  type="number"
                  value={editingData.tw_early || 480}
                  onChange={(e) => updateEditingField('tw_early', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="時間窓終了 (分)"
                  type="number"
                  value={editingData.tw_late || 1080}
                  onChange={(e) => updateEditingField('tw_late', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
            </Grid>
          )}

          {editingData && editingType === 'depots' && (
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="X座標"
                  type="number"
                  value={editingData.x || 0}
                  onChange={(e) => updateEditingField('x', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Y座標"
                  type="number"
                  value={editingData.y || 0}
                  onChange={(e) => updateEditingField('y', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="時間窓開始 (分)"
                  type="number"
                  value={editingData.tw_early || 480}
                  onChange={(e) => updateEditingField('tw_early', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="時間窓終了 (分)"
                  type="number"
                  value={editingData.tw_late || 1080}
                  onChange={(e) => updateEditingField('tw_late', parseInt(e.target.value))}
                  fullWidth
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEdit}>キャンセル</Button>
          <Button onClick={handleSaveEdit} variant="contained">
            保存
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default DataManagementPanel;