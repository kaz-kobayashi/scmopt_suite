import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Tab,
  Tabs,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Divider
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  CloudDownload as DownloadIcon,
  Description as FileIcon,
  TableView as ExcelIcon,
  DataObject as JsonIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  GetApp as TemplateIcon,
  Visibility as PreviewIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface DataImportExportProps {
  jobs?: any[];
  machines?: any[];
  currentSolution?: any;
  currentProblem?: any;
  events?: any[];
  onJobsImported?: (jobs: any[]) => void;
  onMachinesImported?: (machines: any[]) => void;
  onProblemImported?: (problem: any) => void;
}

interface ValidationResult {
  valid: boolean;
  job_count?: number;
  machine_count?: number;
  warnings: string[];
  errors: string[];
}

const DataImportExport: React.FC<DataImportExportProps> = ({
  jobs = [],
  machines = [],
  currentSolution,
  currentProblem,
  events = [],
  onJobsImported,
  onMachinesImported,
  onProblemImported
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [previewDialog, setPreviewDialog] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);

  // File upload handlers
  const handleJobsUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    
    const file = files[0];
    setLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      // Validation first
      const formData = new FormData();
      formData.append('file', file);
      
      const validationResponse = await fetch('http://localhost:8000/api/data/validate/jobs', {
        method: 'POST',
        body: formData
      });
      
      const validation: ValidationResult = await validationResponse.json();
      setValidationResult(validation);
      
      if (!validation.valid) {
        setErrorMessage(`バリデーションエラー: ${validation.errors.join(', ')}`);
        return;
      }
      
      // If valid, proceed with import
      const importResponse = await fetch('http://localhost:8000/api/data/import/jobs', {
        method: 'POST',
        body: formData
      });
      
      if (!importResponse.ok) {
        throw new Error('ジョブインポートに失敗しました');
      }
      
      const importedJobs = await importResponse.json();
      
      if (onJobsImported) {
        onJobsImported(importedJobs);
      }
      
      setSuccessMessage(`${importedJobs.length}個のジョブをインポートしました`);
      
    } catch (error) {
      setErrorMessage(`エラー: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [onJobsImported]);

  const handleMachinesUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    
    const file = files[0];
    setLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      // Validation first
      const formData = new FormData();
      formData.append('file', file);
      
      const validationResponse = await fetch('http://localhost:8000/api/data/validate/machines', {
        method: 'POST',
        body: formData
      });
      
      const validation: ValidationResult = await validationResponse.json();
      setValidationResult(validation);
      
      if (!validation.valid) {
        setErrorMessage(`バリデーションエラー: ${validation.errors.join(', ')}`);
        return;
      }
      
      // If valid, proceed with import
      const importResponse = await fetch('http://localhost:8000/api/data/import/machines', {
        method: 'POST',
        body: formData
      });
      
      if (!importResponse.ok) {
        throw new Error('マシンインポートに失敗しました');
      }
      
      const importedMachines = await importResponse.json();
      
      if (onMachinesImported) {
        onMachinesImported(importedMachines);
      }
      
      setSuccessMessage(`${importedMachines.length}個のマシンをインポートしました`);
      
    } catch (error) {
      setErrorMessage(`エラー: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [onMachinesImported]);

  const handleProblemUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    
    const file = files[0];
    setLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('http://localhost:8000/api/data/import/problem', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('問題インポートに失敗しました');
      }
      
      const importedProblem = await response.json();
      
      if (onProblemImported) {
        onProblemImported(importedProblem);
      }
      
      setSuccessMessage('問題定義をインポートしました');
      
    } catch (error) {
      setErrorMessage(`エラー: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [onProblemImported]);

  // Dropzone configurations
  const jobsDropzone = useDropzone({
    onDrop: handleJobsUpload,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const machinesDropzone = useDropzone({
    onDrop: handleMachinesUpload,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const problemDropzone = useDropzone({
    onDrop: handleProblemUpload,
    accept: {
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  // Export functions
  const exportSolutionCSV = async () => {
    if (!currentSolution) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/data/export/solution/csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentSolution)
      });
      
      if (!response.ok) throw new Error('CSV export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `solution_${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
      
      setSuccessMessage('CSVファイルをダウンロードしました');
    } catch (error) {
      setErrorMessage(`CSV出力エラー: ${error}`);
    }
  };

  const exportSolutionExcel = async () => {
    if (!currentSolution) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/data/export/solution/excel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentSolution)
      });
      
      if (!response.ok) throw new Error('Excel export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `solution_${new Date().toISOString().slice(0, 10)}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
      
      setSuccessMessage('Excelファイルをダウンロードしました');
    } catch (error) {
      setErrorMessage(`Excel出力エラー: ${error}`);
    }
  };

  const exportSolutionJSON = async () => {
    if (!currentSolution) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/data/export/solution/json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentSolution)
      });
      
      if (!response.ok) throw new Error('JSON export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `solution_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      window.URL.revokeObjectURL(url);
      
      setSuccessMessage('JSONファイルをダウンロードしました');
    } catch (error) {
      setErrorMessage(`JSON出力エラー: ${error}`);
    }
  };

  const downloadTemplate = async (type: 'jobs' | 'machines') => {
    try {
      const response = await fetch(`http://localhost:8000/api/data/templates/${type}/csv`);
      
      if (!response.ok) throw new Error('Template download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}_template.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
      
      setSuccessMessage(`${type === 'jobs' ? 'ジョブ' : 'マシン'}テンプレートをダウンロードしました`);
    } catch (error) {
      setErrorMessage(`テンプレートダウンロードエラー: ${error}`);
    }
  };

  const TabPanel = ({ children, value, index }: any) => (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  const DropzoneArea = ({ dropzone, title, description }: any) => (
    <Paper
      {...dropzone.getRootProps()}
      sx={{
        p: 3,
        border: '2px dashed',
        borderColor: dropzone.isDragActive ? 'primary.main' : 'grey.300',
        bgcolor: dropzone.isDragActive ? 'action.hover' : 'background.paper',
        cursor: 'pointer',
        textAlign: 'center',
        minHeight: 120,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.3s ease'
      }}
    >
      <input {...dropzone.getInputProps()} />
      <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {description}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
        CSV, Excel, JSON形式 (最大10MB)
      </Typography>
    </Paper>
  );

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        データインポート・エクスポート
      </Typography>

      {errorMessage && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setErrorMessage('')}>
          {errorMessage}
        </Alert>
      )}

      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage('')}>
          {successMessage}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Paper sx={{ width: '100%' }}>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} aria-label="data import export tabs">
          <Tab label="インポート" icon={<UploadIcon />} />
          <Tab label="エクスポート" icon={<DownloadIcon />} />
          <Tab label="テンプレート" icon={<TemplateIcon />} />
        </Tabs>

        {/* Import Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Jobs Import */}
            <Grid item xs={12} md={4}>
              <DropzoneArea
                dropzone={jobsDropzone}
                title="ジョブデータ"
                description="ジョブと操作情報をインポート"
              />
            </Grid>

            {/* Machines Import */}
            <Grid item xs={12} md={4}>
              <DropzoneArea
                dropzone={machinesDropzone}
                title="マシンデータ"
                description="マシン情報をインポート"
              />
            </Grid>

            {/* Problem Import */}
            <Grid item xs={12} md={4}>
              <DropzoneArea
                dropzone={problemDropzone}
                title="問題定義"
                description="完全な問題設定をインポート"
              />
            </Grid>

            {/* Current Data Summary */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    現在のデータ
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">ジョブ数</Typography>
                      <Typography variant="h6">{jobs.length}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">マシン数</Typography>
                      <Typography variant="h6">{machines.length}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">問題定義</Typography>
                      <Typography variant="h6">{currentProblem ? '有' : '無'}</Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="text.secondary">ソリューション</Typography>
                      <Typography variant="h6">{currentSolution ? '有' : '無'}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            {/* Validation Results */}
            {validationResult && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <CheckIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      バリデーション結果
                    </Typography>

                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Chip
                        label={validationResult.valid ? '有効' : '無効'}
                        color={validationResult.valid ? 'success' : 'error'}
                        icon={validationResult.valid ? <CheckIcon /> : <ErrorIcon />}
                      />
                      
                      {validationResult.job_count !== undefined && (
                        <Chip 
                          label={`ジョブ: ${validationResult.job_count}個`} 
                          sx={{ ml: 1 }}
                        />
                      )}
                      
                      {validationResult.machine_count !== undefined && (
                        <Chip 
                          label={`マシン: ${validationResult.machine_count}個`} 
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Box>

                    {validationResult.warnings.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center' }}>
                          <WarningIcon sx={{ mr: 1 }} color="warning" />
                          警告
                        </Typography>
                        <List dense>
                          {validationResult.warnings.map((warning, index) => (
                            <ListItem key={index}>
                              <ListItemText primary={warning} />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}

                    {validationResult.errors.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center' }}>
                          <ErrorIcon sx={{ mr: 1 }} color="error" />
                          エラー
                        </Typography>
                        <List dense>
                          {validationResult.errors.map((error, index) => (
                            <ListItem key={index}>
                              <ListItemText primary={error} />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        {/* Export Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            {/* Solution Export */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ソリューション出力
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    最適化結果をファイルに出力します
                  </Typography>

                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<FileIcon />}
                        onClick={exportSolutionCSV}
                        disabled={!currentSolution || loading}
                      >
                        CSV
                      </Button>
                    </Grid>
                    <Grid item xs={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<ExcelIcon />}
                        onClick={exportSolutionExcel}
                        disabled={!currentSolution || loading}
                      >
                        Excel
                      </Button>
                    </Grid>
                    <Grid item xs={4}>
                      <Button
                        fullWidth
                        variant="outlined"
                        startIcon={<JsonIcon />}
                        onClick={exportSolutionJSON}
                        disabled={!currentSolution || loading}
                      >
                        JSON
                      </Button>
                    </Grid>
                  </Grid>

                  {!currentSolution && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      ソリューションがありません。まず最適化を実行してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Problem Export */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    問題定義出力
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    現在の問題設定をファイルに出力します
                  </Typography>

                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<JsonIcon />}
                    disabled={!currentProblem || loading}
                    onClick={async () => {
                      if (!currentProblem) return;
                      
                      try {
                        const response = await fetch('http://localhost:8000/api/data/export/problem/json', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify(currentProblem)
                        });
                        
                        if (!response.ok) throw new Error('Problem export failed');
                        
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `problem_${new Date().toISOString().slice(0, 10)}.json`;
                        a.click();
                        window.URL.revokeObjectURL(url);
                        
                        setSuccessMessage('問題定義をダウンロードしました');
                      } catch (error) {
                        setErrorMessage(`問題出力エラー: ${error}`);
                      }
                    }}
                  >
                    JSON出力
                  </Button>

                  {!currentProblem && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      問題定義がありません。まず問題を設定してください。
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Events Export */}
            {events.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      イベント履歴出力
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      スケジュール変更イベントをCSVで出力します ({events.length}件)
                    </Typography>

                    <Button
                      variant="outlined"
                      startIcon={<FileIcon />}
                      onClick={async () => {
                        try {
                          const response = await fetch('http://localhost:8000/api/data/export/events/csv', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(events)
                          });
                          
                          if (!response.ok) throw new Error('Events export failed');
                          
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `events_${new Date().toISOString().slice(0, 10)}.csv`;
                          a.click();
                          window.URL.revokeObjectURL(url);
                          
                          setSuccessMessage('イベント履歴をダウンロードしました');
                        } catch (error) {
                          setErrorMessage(`イベント出力エラー: ${error}`);
                        }
                      }}
                      disabled={loading}
                    >
                      CSV出力
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        {/* Templates Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ジョブデータテンプレート
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    ジョブと操作のインポート用CSVテンプレート
                  </Typography>

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<DownloadIcon />}
                    onClick={() => downloadTemplate('jobs')}
                  >
                    ダウンロード
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    マシンデータテンプレート
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    マシン情報のインポート用CSVテンプレート
                  </Typography>

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={<DownloadIcon />}
                    onClick={() => downloadTemplate('machines')}
                  >
                    ダウンロード
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    サポートされているファイル形式
                  </Typography>
                  
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>データタイプ</TableCell>
                          <TableCell>インポート</TableCell>
                          <TableCell>エクスポート</TableCell>
                          <TableCell>ファイルサイズ制限</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>ジョブ</TableCell>
                          <TableCell>CSV, Excel, JSON</TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>10MB</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>マシン</TableCell>
                          <TableCell>CSV, Excel, JSON</TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>10MB</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>問題定義</TableCell>
                          <TableCell>JSON, Excel</TableCell>
                          <TableCell>JSON</TableCell>
                          <TableCell>10MB</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>ソリューション</TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>CSV, Excel, JSON</TableCell>
                          <TableCell>-</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>イベント履歴</TableCell>
                          <TableCell>-</TableCell>
                          <TableCell>CSV</TableCell>
                          <TableCell>-</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default DataImportExport;