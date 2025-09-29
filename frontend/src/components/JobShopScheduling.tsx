import React, { useState, useCallback, useEffect } from 'react';
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
  Tabs,
  Tab,
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
  Collapse,
  FormControlLabel,
  Switch,
  Slider,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Assessment as AssessmentIcon,
  Schedule as ScheduleIcon,
  Factory as FactoryIcon,
  Work as WorkIcon,
  Timeline as TimelineIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Settings as SettingsIcon,
  CloudDownload as DownloadIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import apiClient from '../services/apiClient';
import RealtimeScheduleManager from './RealtimeScheduleManager';
import DataImportExport from './DataImportExport';

// Types
interface Job {
  id: string;
  name: string;
  priority: number;
  weight: number;
  release_time: number;
  due_date?: number;
  deadline?: number;
  operations: Operation[];
}

interface Operation {
  id: string;
  job_id: string;
  machine_id?: string;
  duration: number;
  position_in_job: number;
  setup_time?: number;
  eligible_machines?: string[];
  earliest_start?: number;
  latest_finish?: number;
  skill_requirements?: string[];
}

interface Machine {
  id: string;
  name: string;
  capacity: number;
  available_from: number;
  available_until?: number;
  setup_matrix?: any;
  skills?: string[];
  maintenance_windows?: any[];
}

interface JobShopProblem {
  problem_type: string;
  jobs: Job[];
  machines: Machine[];
  resources?: any[];
  optimization_objective: string;
  time_horizon?: number;
  allow_preemption: boolean;
  setup_times_included: boolean;
  max_solve_time_seconds: number;
  optimality_gap_tolerance: number;
}

interface JobShopSolution {
  problem_type: string;
  job_schedules: any[];
  machine_schedules: any[];
  metrics: any;
  solution_status: string;
  gantt_chart_data?: any;
  critical_path?: string[];
  bottleneck_machines?: string[];
  improvement_suggestions?: string[];
  bottleneck_analysis?: any;
  advanced_kpis?: any;
}

interface MultiObjectiveWeights {
  makespan_weight: number;
  tardiness_weight: number;
  completion_time_weight: number;
  resource_cost_weight: number;
}

const JobShopScheduling: React.FC = () => {
  // State
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Problem definition state
  const [problemType, setProblemType] = useState<string>('job_shop');
  const [optimizationObjective, setOptimizationObjective] = useState<string>('makespan');
  const [maxSolveTime, setMaxSolveTime] = useState<number>(300);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [machines, setMachines] = useState<Machine[]>([]);
  
  // Multi-objective state
  const [multiObjective, setMultiObjective] = useState(false);
  const [paretoAnalysis, setParetoAnalysis] = useState(false);
  const [objectiveWeights, setObjectiveWeights] = useState<MultiObjectiveWeights>({
    makespan_weight: 1.0,
    tardiness_weight: 0.0,
    completion_time_weight: 0.0,
    resource_cost_weight: 0.0
  });
  
  // Advanced settings state
  const [allowPreemption, setAllowPreemption] = useState(false);
  const [setupTimesIncluded, setSetupTimesIncluded] = useState(false);
  const [timeHorizon, setTimeHorizon] = useState<number | undefined>(undefined);
  
  // Solution state
  const [solution, setSolution] = useState<JobShopSolution | null>(null);
  const [showSolution, setShowSolution] = useState(false);
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({
    machines: true,
    jobs: true,
    advanced: false,
    multiObjective: false,
    kpis: false
  });

  // Template state
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [saveTemplateDialogOpen, setSaveTemplateDialogOpen] = useState(false);
  const [availableTemplates, setAvailableTemplates] = useState<any[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');
  const [templateForm, setTemplateForm] = useState({
    name: '',
    description: '',
    category: 'general',
    tags: [] as string[],
    is_public: true
  });

  // Tab panel component
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
        id={`simple-tabpanel-${index}`}
        aria-labelledby={`simple-tab-${index}`}
        {...other}
      >
        {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
      </div>
    );
  }

  // Handlers
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Template functions
  const loadTemplateList = async () => {
    try {
      const response = await apiClient.get('/templates?limit=50');
      setAvailableTemplates(response.data.templates);
    } catch (err) {
      console.error('テンプレート一覧取得に失敗:', err);
    }
  };

  const loadFromTemplate = async () => {
    if (!selectedTemplateId) return;
    
    try {
      setLoading(true);
      const response = await apiClient.get(`/templates/${selectedTemplateId}`);
      const template = response.data;
      
      // テンプレートからデータを読み込み
      const problemTemplate = template.problem_template;
      
      if (problemTemplate.machines) {
        setMachines(problemTemplate.machines);
      }
      if (problemTemplate.jobs) {
        setJobs(problemTemplate.jobs);
      }
      if (problemTemplate.problem_type) {
        setProblemType(problemTemplate.problem_type);
      }
      if (problemTemplate.optimization_objective) {
        setOptimizationObjective(problemTemplate.optimization_objective);
      }
      if (problemTemplate.time_horizon) {
        setTimeHorizon(problemTemplate.time_horizon);
      }
      
      setSuccess(`テンプレート "${template.name}" を読み込みました`);
      setTemplateDialogOpen(false);
      
    } catch (err: any) {
      setError('テンプレート読み込みに失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const saveAsTemplate = async () => {
    try {
      setLoading(true);
      
      const templateData = {
        name: templateForm.name,
        description: templateForm.description,
        category: templateForm.category,
        problem_template: {
          problem_type: problemType,
          jobs: jobs,
          machines: machines,
          optimization_objective: optimizationObjective,
          time_horizon: timeHorizon,
          allow_preemption: allowPreemption,
          setup_times_included: setupTimesIncluded
        },
        default_solver_config: {
          solver_type: "pyjobshop",
          time_limit_seconds: maxSolveTime,
          optimization_objective: optimizationObjective
        },
        default_analysis_config: {
          include_critical_path: true,
          include_bottleneck_analysis: true,
          include_improvement_suggestions: true,
          include_gantt_chart: true
        },
        tags: templateForm.tags,
        is_public: templateForm.is_public
      };
      
      await apiClient.post('/templates', templateData);
      setSuccess('テンプレートが保存されました');
      setSaveTemplateDialogOpen(false);
      
      // フォームをリセット
      setTemplateForm({
        name: '',
        description: '',
        category: 'general',
        tags: [],
        is_public: true
      });
      
    } catch (err: any) {
      setError('テンプレート保存に失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Load templates on component mount
  useEffect(() => {
    loadTemplateList();
  }, []);

  const addMachine = () => {
    const newMachine: Machine = {
      id: `M${machines.length + 1}`,
      name: `Machine ${machines.length + 1}`,
      capacity: 1,
      available_from: 0
    };
    setMachines([...machines, newMachine]);
  };

  const removeMachine = (index: number) => {
    setMachines(machines.filter((_, i) => i !== index));
  };

  const updateMachine = (index: number, field: keyof Machine, value: any) => {
    const updated = machines.map((machine, i) => 
      i === index ? { ...machine, [field]: value } : machine
    );
    setMachines(updated);
  };

  const addJob = () => {
    const newJob: Job = {
      id: `J${jobs.length + 1}`,
      name: `Job ${jobs.length + 1}`,
      priority: 1,
      weight: 1.0,
      release_time: 0,
      operations: []
    };
    setJobs([...jobs, newJob]);
  };

  const removeJob = (index: number) => {
    setJobs(jobs.filter((_, i) => i !== index));
  };

  const updateJob = (index: number, field: keyof Job, value: any) => {
    const updated = jobs.map((job, i) => 
      i === index ? { ...job, [field]: value } : job
    );
    setJobs(updated);
  };

  const addOperation = (jobIndex: number) => {
    const job = jobs[jobIndex];
    const newOperation: Operation = {
      id: `${job.id}_O${job.operations.length + 1}`,
      job_id: job.id,
      machine_id: machines.length > 0 ? machines[0].id : undefined,
      duration: 1,
      position_in_job: job.operations.length
    };
    
    const updatedJob = {
      ...job,
      operations: [...job.operations, newOperation]
    };
    
    const updated = jobs.map((j, i) => i === jobIndex ? updatedJob : j);
    setJobs(updated);
  };

  const removeOperation = (jobIndex: number, operationIndex: number) => {
    const job = jobs[jobIndex];
    const updatedJob = {
      ...job,
      operations: job.operations.filter((_, i) => i !== operationIndex)
        .map((op, i) => ({ ...op, position_in_job: i }))
    };
    
    const updated = jobs.map((j, i) => i === jobIndex ? updatedJob : j);
    setJobs(updated);
  };

  const updateOperation = (jobIndex: number, operationIndex: number, field: keyof Operation, value: any) => {
    const job = jobs[jobIndex];
    const updatedOperations = job.operations.map((op, i) =>
      i === operationIndex ? { ...op, [field]: value } : op
    );
    
    const updatedJob = { ...job, operations: updatedOperations };
    const updated = jobs.map((j, i) => i === jobIndex ? updatedJob : j);
    setJobs(updated);
  };

  const loadSampleProblem = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get(`/jobshop/sample-problem/${problemType}`);
      const sampleProblem = response.data;
      
      setJobs(sampleProblem.jobs);
      setMachines(sampleProblem.machines);
      setOptimizationObjective(sampleProblem.optimization_objective);
      setMaxSolveTime(sampleProblem.max_solve_time_seconds);
      
      setSuccess('サンプル問題を読み込みました');
    } catch (err: any) {
      setError('サンプル問題の読み込みに失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const solveProblem = async () => {
    if (jobs.length === 0 || machines.length === 0) {
      setError('ジョブとマシンを追加してください');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const problem: JobShopProblem = {
        problem_type: problemType,
        jobs,
        machines,
        optimization_objective: optimizationObjective,
        time_horizon: timeHorizon,
        allow_preemption: allowPreemption,
        setup_times_included: setupTimesIncluded,
        max_solve_time_seconds: maxSolveTime,
        optimality_gap_tolerance: 0.01
      };

      let endpoint = '/jobshop/solve';
      let requestData: any = { problem };

      if (multiObjective) {
        const multiObjectiveProblem = {
          ...problem,
          objective_weights: objectiveWeights,
          pareto_analysis: paretoAnalysis
        };
        endpoint = paretoAnalysis ? '/jobshop/solve-pareto-analysis' : '/jobshop/solve-multi-objective';
        requestData = multiObjectiveProblem;
      }

      const response = await apiClient.post(endpoint, requestData);
      const solutionData = response.data;
      setSolution(solutionData);
      setShowSolution(true);
      setTabValue(1); // Switch to results tab
      setSuccess(`最適化が完了しました (Status: ${solutionData.solution_status})`);
    } catch (err: any) {
      setError('最適化に失敗しました: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const renderGanttChart = () => {
    if (!solution || !solution.gantt_chart_data) {
      return <Typography>ガントチャートデータがありません</Typography>;
    }

    // Enhanced Gantt chart with improved styling and interactivity
    const traces: any[] = [];
    const machineNames = Array.from(new Set(solution.machine_schedules.map((m: any) => m.machine_id)));
    
    // Color palette for jobs
    const jobColors: { [key: string]: string } = {};
    let colorIndex = 0;
    const colorPalette = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
      '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
      '#10AC84', '#EE5A24', '#0984E3', '#A29BFE', '#FD79A8'
    ];

    solution.machine_schedules.forEach((machine: any, machineIndex: number) => {
      machine.operations.forEach((op: any) => {
        // Assign consistent colors to jobs
        if (!jobColors[op.job_id]) {
          jobColors[op.job_id] = colorPalette[colorIndex % colorPalette.length];
          colorIndex++;
        }

        // Find job details for enhanced tooltips
        const jobDetail = jobs.find(j => j.id === op.job_id);
        const machineDetail = machines.find(m => m.id === machine.machine_id);
        
        // Calculate additional metrics
        const duration = op.end_time - op.start_time;
        const isLate = jobDetail?.due_date && op.end_time > jobDetail.due_date;
        const criticality = jobDetail?.priority || 1;
        
        traces.push({
          type: 'bar',
          orientation: 'h',
          name: `Job ${op.job_id}`,
          showlegend: false,
          y: [machine.machine_id],
          x: [duration],
          base: [op.start_time],
          text: [`${op.operation_id}`],
          textposition: 'inside',
          textfont: {
            color: 'white',
            size: 10,
            family: 'Arial, sans-serif'
          },
          customdata: [{
            job_id: op.job_id,
            operation_id: op.operation_id,
            machine_id: machine.machine_id,
            start_time: op.start_time,
            end_time: op.end_time,
            duration: duration,
            job_priority: jobDetail?.priority || 'N/A',
            job_weight: jobDetail?.weight || 'N/A',
            due_date: jobDetail?.due_date || 'なし',
            machine_name: machineDetail?.name || machine.machine_id,
            is_late: isLate,
            setup_time: op.setup_time || 0
          }],
          hovertemplate: 
            '<b>%{customdata.operation_id}</b><br>' +
            '<b>ジョブ:</b> %{customdata.job_id}<br>' +
            '<b>マシン:</b> %{customdata.machine_name}<br>' +
            '<b>開始時刻:</b> %{customdata.start_time}<br>' +
            '<b>完了時刻:</b> %{customdata.end_time}<br>' +
            '<b>処理時間:</b> %{customdata.duration}<br>' +
            '<b>セットアップ時間:</b> %{customdata.setup_time}<br>' +
            '<b>ジョブ優先度:</b> %{customdata.job_priority}<br>' +
            '<b>ジョブ重み:</b> %{customdata.job_weight}<br>' +
            '<b>期限:</b> %{customdata.due_date}<br>' +
            '<b>遅延:</b> %{customdata.is_late}<br>' +
            '<extra></extra>',
          marker: {
            color: jobColors[op.job_id],
            opacity: isLate ? 0.7 : 0.9,
            line: {
              color: isLate ? '#FF0000' : jobColors[op.job_id],
              width: isLate ? 3 : 1
            },
            // Add pattern for high priority jobs
            pattern: criticality >= 3 ? {
              shape: '/',
              bgcolor: jobColors[op.job_id],
              fgcolor: 'rgba(255,255,255,0.3)',
              size: 8,
              solidity: 0.3
            } : undefined
          }
        });
      });
    });

    // Add idle time visualization
    solution.machine_schedules.forEach((machine: any) => {
      const operations = machine.operations.sort((a: any, b: any) => a.start_time - b.start_time);
      
      operations.forEach((op: any, index: number) => {
        if (index < operations.length - 1) {
          const nextOp = operations[index + 1];
          const idleTime = nextOp.start_time - op.end_time;
          
          if (idleTime > 0) {
            traces.push({
              type: 'bar',
              orientation: 'h',
              name: 'Idle Time',
              showlegend: false,
              y: [machine.machine_id],
              x: [idleTime],
              base: [op.end_time],
              text: [''],
              hovertemplate: 
                '<b>アイドル時間</b><br>' +
                '<b>マシン:</b> %{y}<br>' +
                '<b>開始:</b> %{base}<br>' +
                '<b>時間:</b> %{x}<br>' +
                '<extra></extra>',
              marker: {
                color: 'rgba(128,128,128,0.3)',
                line: {
                  color: 'rgba(128,128,128,0.5)',
                  width: 1,
                  dash: 'dash'
                }
              }
            });
          }
        }
      });
    });

    // Create legend data for jobs
    const uniqueJobs = Object.keys(jobColors);
    const legendTraces = uniqueJobs.map(jobId => ({
      type: 'scatter',
      mode: 'markers',
      x: [null],
      y: [null],
      name: `Job ${jobId}`,
      marker: {
        color: jobColors[jobId],
        size: 10
      },
      showlegend: true
    }));

    return (
      <Box>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            インタラクティブガントチャート
          </Typography>
          <Box>
            <Button
              variant="outlined"
              size="small"
              startIcon={<DownloadIcon />}
              onClick={() => {
                // Export functionality could be added here
                console.log('Export functionality');
              }}
              sx={{ mr: 1 }}
            >
              エクスポート
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={() => {
                // Refresh/reset zoom functionality
                console.log('Reset zoom');
              }}
            >
              リセット
            </Button>
          </Box>
        </Box>
        
        <Box sx={{ border: '1px solid #e0e0e0', borderRadius: 1, overflow: 'hidden' }}>
          <Plot
            data={[...traces, ...legendTraces] as any}
            layout={{
              title: 'ジョブショップスケジュール - 詳細ガントチャート',
              xaxis: { 
                title: '時間 (時間単位)'
              },
              yaxis: { 
                title: 'マシン'
              },
              height: Math.max(400, machineNames.length * 50 + 150),
              plot_bgcolor: '#fafafa',
              paper_bgcolor: 'white',
              showlegend: true,
              margin: {
                l: 100,
                r: 50,
                b: 150,
                t: 80
              }
            } as any}
            style={{ width: '100%', height: `${Math.max(400, machineNames.length * 50 + 150)}px` }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
              toImageButtonOptions: {
                format: 'png' as const,
                filename: 'gantt_chart',
                height: Math.max(400, machineNames.length * 50 + 150),
                width: 1200,
                scale: 2
              }
            }}
          />
        </Box>
        
        {/* Chart Legend and Information */}
        <Box mt={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>チャート情報</Typography>
                <Typography variant="body2" color="text.secondary">
                  • 各バーはジョブの操作を表します<br/>
                  • 色は異なるジョブを示します<br/>
                  • ハッチングは高優先度ジョブです<br/>
                  • 赤枠は期限遅れを示します
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>インタラクション</Typography>
                <Typography variant="body2" color="text.secondary">
                  • バーにカーソルを合わせて詳細表示<br/>
                  • ズーム・パンで詳細確認<br/>
                  • 凡例クリックでジョブ表示/非表示<br/>
                  • ツールバーで画像エクスポート可能
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>統計情報</Typography>
                <Typography variant="body2" color="text.secondary">
                  • 総ジョブ数: {uniqueJobs.length}<br/>
                  • 総マシン数: {machineNames.length}<br/>
                  • 最大完了時間: {solution.metrics?.makespan || 'N/A'}<br/>
                  • 平均稼働率: {solution.machine_schedules.length > 0 ? 
                    ((solution.machine_schedules.reduce((sum: number, m: any) => sum + (m.utilization || 0), 0) / solution.machine_schedules.length) * 100).toFixed(1) : 'N/A'}%
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Box>
    );
  };

  const renderMetrics = () => {
    if (!solution) return null;

    const metrics = solution.metrics;
    
    return (
      <Grid container spacing={2}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                メイクスパン
              </Typography>
              <Typography variant="h4" component="h2" color="primary">
                {metrics.makespan}
              </Typography>
              <Typography color="textSecondary">
                時間単位
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                総遅延時間
              </Typography>
              <Typography variant="h4" component="h2" color="secondary">
                {metrics.total_tardiness}
              </Typography>
              <Typography color="textSecondary">
                時間単位
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                平均稼働率
              </Typography>
              <Typography variant="h4" component="h2" style={{ color: '#ff9800' }}>
                {(metrics.average_machine_utilization * 100).toFixed(1)}%
              </Typography>
              <Typography color="textSecondary">
                マシン稼働率
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                求解時間
              </Typography>
              <Typography variant="h4" component="h2" style={{ color: '#4caf50' }}>
                {metrics.solve_time_seconds.toFixed(2)}s
              </Typography>
              <Typography color="textSecondary">
                計算時間
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        <FactoryIcon sx={{ mr: 1 }} />
        ジョブショップ スケジューリング
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="job shop tabs">
          <Tab label="問題設定" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="ジョブ管理" icon={<WorkIcon />} iconPosition="start" />
          <Tab label="機械管理" icon={<FactoryIcon />} iconPosition="start" />
          <Tab label="スケジューリング" icon={<ScheduleIcon />} iconPosition="start" />
          <Tab label="最適化結果" icon={<AssessmentIcon />} iconPosition="start" />
          <Tab label="リアルタイム監視" icon={<TimelineIcon />} iconPosition="start" />
          <Tab label="データ管理" icon={<DownloadIcon />} iconPosition="start" />
        </Tabs>
      </Box>
        
        <TabPanel value={tabValue} index={0}>
          {/* Problem Configuration */}
          <Grid container spacing={3}>
            {/* Basic Settings */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    基本設定
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <FormControl fullWidth>
                        <InputLabel>問題タイプ</InputLabel>
                        <Select
                          value={problemType}
                          onChange={(e) => setProblemType(e.target.value)}
                        >
                          <MenuItem value="job_shop">ジョブショップ</MenuItem>
                          <MenuItem value="flexible_job_shop">フレキシブルジョブショップ</MenuItem>
                          <MenuItem value="flow_shop">フローショップ</MenuItem>
                          <MenuItem value="hybrid_flow_shop">ハイブリッドフローショップ</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <FormControl fullWidth>
                        <InputLabel>最適化目的</InputLabel>
                        <Select
                          value={optimizationObjective}
                          onChange={(e) => setOptimizationObjective(e.target.value)}
                          disabled={multiObjective}
                        >
                          <MenuItem value="makespan">メイクスパン最小化</MenuItem>
                          <MenuItem value="total_tardiness">総遅延時間最小化</MenuItem>
                          <MenuItem value="total_completion_time">総完了時間最小化</MenuItem>
                          <MenuItem value="weighted_tardiness">重み付き遅延最小化</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <TextField
                        fullWidth
                        label="最大求解時間 (秒)"
                        type="number"
                        value={maxSolveTime}
                        onChange={(e) => setMaxSolveTime(Number(e.target.value))}
                      />
                    </Grid>
                  </Grid>
                </CardContent>
                <CardActions>
                  <Button 
                    startIcon={<RefreshIcon />}
                    onClick={loadSampleProblem}
                    disabled={loading}
                    sx={{ mr: 1 }}
                  >
                    サンプル問題読み込み
                  </Button>
                  <Button
                    startIcon={<DownloadIcon />}
                    onClick={() => setTemplateDialogOpen(true)}
                    disabled={loading}
                    variant="outlined"
                  >
                    テンプレートから読み込み
                  </Button>
                  <Button
                    startIcon={<AddIcon />}
                    onClick={() => setSaveTemplateDialogOpen(true)}
                    disabled={loading || jobs.length === 0 || machines.length === 0}
                    variant="outlined"
                    sx={{ ml: 1 }}
                  >
                    テンプレート保存
                  </Button>
                </CardActions>
              </Card>
            </Grid>

            {/* Multi-objective Settings */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      マルチ目的最適化
                    </Typography>
                    <IconButton onClick={() => toggleSection('multiObjective')}>
                      {expandedSections.multiObjective ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={multiObjective}
                        onChange={(e) => setMultiObjective(e.target.checked)}
                      />
                    }
                    label="マルチ目的最適化を有効にする"
                  />
                  
                  <Collapse in={expandedSections.multiObjective && multiObjective}>
                    <Box mt={2}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={paretoAnalysis}
                            onChange={(e) => setParetoAnalysis(e.target.checked)}
                          />
                        }
                        label="Pareto前線分析を実行"
                      />
                      
                      {!paretoAnalysis && (
                        <Grid container spacing={2} mt={1}>
                          <Grid item xs={12} md={3}>
                            <Typography gutterBottom>メイクスパン重み: {objectiveWeights.makespan_weight}</Typography>
                            <Slider
                              value={objectiveWeights.makespan_weight}
                              onChange={(_, value) => setObjectiveWeights(prev => ({ ...prev, makespan_weight: value as number }))}
                              min={0}
                              max={1}
                              step={0.1}
                              valueLabelDisplay="auto"
                            />
                          </Grid>
                          <Grid item xs={12} md={3}>
                            <Typography gutterBottom>遅延重み: {objectiveWeights.tardiness_weight}</Typography>
                            <Slider
                              value={objectiveWeights.tardiness_weight}
                              onChange={(_, value) => setObjectiveWeights(prev => ({ ...prev, tardiness_weight: value as number }))}
                              min={0}
                              max={1}
                              step={0.1}
                              valueLabelDisplay="auto"
                            />
                          </Grid>
                          <Grid item xs={12} md={3}>
                            <Typography gutterBottom>完了時間重み: {objectiveWeights.completion_time_weight}</Typography>
                            <Slider
                              value={objectiveWeights.completion_time_weight}
                              onChange={(_, value) => setObjectiveWeights(prev => ({ ...prev, completion_time_weight: value as number }))}
                              min={0}
                              max={1}
                              step={0.1}
                              valueLabelDisplay="auto"
                            />
                          </Grid>
                          <Grid item xs={12} md={3}>
                            <Typography gutterBottom>リソースコスト重み: {objectiveWeights.resource_cost_weight}</Typography>
                            <Slider
                              value={objectiveWeights.resource_cost_weight}
                              onChange={(_, value) => setObjectiveWeights(prev => ({ ...prev, resource_cost_weight: value as number }))}
                              min={0}
                              max={1}
                              step={0.1}
                              valueLabelDisplay="auto"
                            />
                          </Grid>
                        </Grid>
                      )}
                    </Box>
                  </Collapse>
                </CardContent>
              </Card>
            </Grid>

            {/* Machines */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      <FactoryIcon sx={{ mr: 1 }} />
                      マシン ({machines.length})
                    </Typography>
                    <Button
                      startIcon={<AddIcon />}
                      onClick={addMachine}
                      size="small"
                    >
                      追加
                    </Button>
                    <IconButton onClick={() => toggleSection('machines')}>
                      {expandedSections.machines ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                  
                  <Collapse in={expandedSections.machines}>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>名前</TableCell>
                            <TableCell>容量</TableCell>
                            <TableCell>開始時刻</TableCell>
                            <TableCell>終了時刻</TableCell>
                            <TableCell>操作</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {machines.map((machine, index) => (
                            <TableRow key={machine.id}>
                              <TableCell>{machine.id}</TableCell>
                              <TableCell>
                                <TextField
                                  size="small"
                                  value={machine.name}
                                  onChange={(e) => updateMachine(index, 'name', e.target.value)}
                                />
                              </TableCell>
                              <TableCell>
                                <TextField
                                  size="small"
                                  type="number"
                                  value={machine.capacity}
                                  onChange={(e) => updateMachine(index, 'capacity', Number(e.target.value))}
                                />
                              </TableCell>
                              <TableCell>
                                <TextField
                                  size="small"
                                  type="number"
                                  value={machine.available_from}
                                  onChange={(e) => updateMachine(index, 'available_from', Number(e.target.value))}
                                />
                              </TableCell>
                              <TableCell>
                                <TextField
                                  size="small"
                                  type="number"
                                  value={machine.available_until || ''}
                                  onChange={(e) => updateMachine(index, 'available_until', e.target.value ? Number(e.target.value) : undefined)}
                                />
                              </TableCell>
                              <TableCell>
                                <IconButton
                                  onClick={() => removeMachine(index)}
                                  size="small"
                                  color="error"
                                >
                                  <DeleteIcon />
                                </IconButton>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Collapse>
                </CardContent>
              </Card>
            </Grid>

            {/* Jobs */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      <WorkIcon sx={{ mr: 1 }} />
                      ジョブ ({jobs.length})
                    </Typography>
                    <Button
                      startIcon={<AddIcon />}
                      onClick={addJob}
                      size="small"
                    >
                      追加
                    </Button>
                    <IconButton onClick={() => toggleSection('jobs')}>
                      {expandedSections.jobs ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                  
                  <Collapse in={expandedSections.jobs}>
                    {jobs.map((job, jobIndex) => (
                      <Card key={job.id} variant="outlined" sx={{ mb: 2 }}>
                        <CardContent>
                          <Grid container spacing={2} alignItems="center">
                            <Grid item xs={12} md={2}>
                              <TextField
                                size="small"
                                label="ID"
                                value={job.id}
                                onChange={(e) => updateJob(jobIndex, 'id', e.target.value)}
                              />
                            </Grid>
                            <Grid item xs={12} md={2}>
                              <TextField
                                size="small"
                                label="名前"
                                value={job.name}
                                onChange={(e) => updateJob(jobIndex, 'name', e.target.value)}
                              />
                            </Grid>
                            <Grid item xs={12} md={2}>
                              <TextField
                                size="small"
                                label="優先度"
                                type="number"
                                value={job.priority}
                                onChange={(e) => updateJob(jobIndex, 'priority', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12} md={2}>
                              <TextField
                                size="small"
                                label="重み"
                                type="number"
                                value={job.weight}
                                onChange={(e) => updateJob(jobIndex, 'weight', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12} md={2}>
                              <TextField
                                size="small"
                                label="納期"
                                type="number"
                                value={job.due_date || ''}
                                onChange={(e) => updateJob(jobIndex, 'due_date', e.target.value ? Number(e.target.value) : undefined)}
                              />
                            </Grid>
                            <Grid item xs={12} md={2}>
                              <Box display="flex" gap={1}>
                                <Button
                                  size="small"
                                  startIcon={<AddIcon />}
                                  onClick={() => addOperation(jobIndex)}
                                >
                                  工程追加
                                </Button>
                                <IconButton
                                  onClick={() => removeJob(jobIndex)}
                                  size="small"
                                  color="error"
                                >
                                  <DeleteIcon />
                                </IconButton>
                              </Box>
                            </Grid>
                          </Grid>
                          
                          {/* Operations */}
                          {job.operations.length > 0 && (
                            <Box mt={2}>
                              <Typography variant="subtitle2" gutterBottom>工程</Typography>
                              <TableContainer>
                                <Table size="small">
                                  <TableHead>
                                    <TableRow>
                                      <TableCell>ID</TableCell>
                                      <TableCell>マシン</TableCell>
                                      <TableCell>処理時間</TableCell>
                                      <TableCell>セットアップ時間</TableCell>
                                      <TableCell>操作</TableCell>
                                    </TableRow>
                                  </TableHead>
                                  <TableBody>
                                    {job.operations.map((operation, opIndex) => (
                                      <TableRow key={operation.id}>
                                        <TableCell>{operation.id}</TableCell>
                                        <TableCell>
                                          <FormControl size="small" sx={{ minWidth: 120 }}>
                                            <Select
                                              value={operation.machine_id || ''}
                                              onChange={(e) => updateOperation(jobIndex, opIndex, 'machine_id', e.target.value)}
                                            >
                                              {machines.map((machine) => (
                                                <MenuItem key={machine.id} value={machine.id}>
                                                  {machine.name}
                                                </MenuItem>
                                              ))}
                                            </Select>
                                          </FormControl>
                                        </TableCell>
                                        <TableCell>
                                          <TextField
                                            size="small"
                                            type="number"
                                            value={operation.duration}
                                            onChange={(e) => updateOperation(jobIndex, opIndex, 'duration', Number(e.target.value))}
                                            sx={{ width: 80 }}
                                          />
                                        </TableCell>
                                        <TableCell>
                                          <TextField
                                            size="small"
                                            type="number"
                                            value={operation.setup_time || 0}
                                            onChange={(e) => updateOperation(jobIndex, opIndex, 'setup_time', Number(e.target.value))}
                                            sx={{ width: 80 }}
                                          />
                                        </TableCell>
                                        <TableCell>
                                          <IconButton
                                            onClick={() => removeOperation(jobIndex, opIndex)}
                                            size="small"
                                            color="error"
                                          >
                                            <DeleteIcon />
                                          </IconButton>
                                        </TableCell>
                                      </TableRow>
                                    ))}
                                  </TableBody>
                                </Table>
                              </TableContainer>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Collapse>
                </CardContent>
              </Card>
            </Grid>

            {/* Advanced Settings */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      詳細設定
                    </Typography>
                    <IconButton onClick={() => toggleSection('advanced')}>
                      {expandedSections.advanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                  
                  <Collapse in={expandedSections.advanced}>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={allowPreemption}
                              onChange={(e) => setAllowPreemption(e.target.checked)}
                            />
                          }
                          label="プリエンプション許可"
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={setupTimesIncluded}
                              onChange={(e) => setSetupTimesIncluded(e.target.checked)}
                            />
                          }
                          label="セットアップ時間を含める"
                        />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <TextField
                          fullWidth
                          label="時間窓 (オプション)"
                          type="number"
                          value={timeHorizon || ''}
                          onChange={(e) => setTimeHorizon(e.target.value ? Number(e.target.value) : undefined)}
                        />
                      </Grid>
                    </Grid>
                  </Collapse>
                </CardContent>
              </Card>
            </Grid>

            {/* Solve Button */}
            <Grid item xs={12}>
              <Box display="flex" justifyContent="center" mt={2}>
                <Button
                  variant="contained"
                  size="large"
                  startIcon={loading ? <CircularProgress size={20} /> : <PlayIcon />}
                  onClick={solveProblem}
                  disabled={loading || jobs.length === 0 || machines.length === 0}
                  sx={{ minWidth: 200 }}
                >
                  {loading ? '最適化中...' : 'スケジュール最適化実行'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {/* Results */}
          {solution && (
            <Box>
              <Typography variant="h5" gutterBottom>
                <AssessmentIcon sx={{ mr: 1 }} />
                最適化結果
              </Typography>
              
              <Grid container spacing={3}>
                {/* Status and Metrics */}
                <Grid item xs={12}>
                  <Alert severity={solution.solution_status === 'OPTIMAL' ? 'success' : 'info'}>
                    ステータス: {solution.solution_status}
                  </Alert>
                </Grid>

                <Grid item xs={12}>
                  {renderMetrics()}
                </Grid>

                {/* Gantt Chart */}
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        <TimelineIcon sx={{ mr: 1 }} />
                        ガントチャート
                      </Typography>
                      {renderGanttChart()}
                    </CardContent>
                  </Card>
                </Grid>

                {/* Detailed Schedules */}
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        ジョブスケジュール
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>ジョブ</TableCell>
                              <TableCell>開始</TableCell>
                              <TableCell>完了</TableCell>
                              <TableCell>遅延</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {solution.job_schedules.map((job: any) => (
                              <TableRow key={job.job_id}>
                                <TableCell>{job.job_id}</TableCell>
                                <TableCell>{job.start_time}</TableCell>
                                <TableCell>{job.completion_time}</TableCell>
                                <TableCell>
                                  <Chip 
                                    label={job.tardiness}
                                    color={job.tardiness > 0 ? 'error' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        マシンスケジュール
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>マシン</TableCell>
                              <TableCell>稼働率</TableCell>
                              <TableCell>アイドル時間</TableCell>
                              <TableCell>操作数</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {solution.machine_schedules.map((machine: any) => (
                              <TableRow key={machine.machine_id}>
                                <TableCell>{machine.machine_id}</TableCell>
                                <TableCell>
                                  <Chip
                                    label={`${(machine.utilization * 100).toFixed(1)}%`}
                                    color={machine.utilization > 0.8 ? 'error' : machine.utilization > 0.5 ? 'warning' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                                <TableCell>{machine.idle_time}</TableCell>
                                <TableCell>{machine.operations.length}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Advanced Analysis Section */}
                {(solution.bottleneck_analysis || solution.advanced_kpis || solution.critical_path) && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          <AssessmentIcon sx={{ mr: 1 }} />
                          高度分析結果
                        </Typography>
                        
                        <Grid container spacing={3}>
                          {/* Critical Path Analysis */}
                          {solution.critical_path && solution.critical_path.length > 0 && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#fff3e0' }}>
                                <Typography variant="subtitle2" gutterBottom color="primary">
                                  <TimelineIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  クリティカルパス
                                </Typography>
                                <Typography variant="body2" gutterBottom>
                                  作業数: {solution.critical_path.length}
                                </Typography>
                                <Box sx={{ maxHeight: 150, overflowY: 'auto' }}>
                                  {solution.critical_path.map((op: string, index: number) => (
                                    <Chip
                                      key={index}
                                      label={op}
                                      size="small"
                                      color="warning"
                                      sx={{ m: 0.5 }}
                                    />
                                  ))}
                                </Box>
                              </Paper>
                            </Grid>
                          )}
                          
                          {/* Bottleneck Analysis */}
                          {solution.bottleneck_analysis && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#ffebee' }}>
                                <Typography variant="subtitle2" gutterBottom color="error">
                                  <FactoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  ボトルネック分析
                                </Typography>
                                {solution.bottleneck_analysis.analysis && (
                                  <>
                                    <Typography variant="body2" gutterBottom>
                                      ボトルネック数: {solution.bottleneck_analysis.bottleneck_machines?.length || 0}
                                    </Typography>
                                    {solution.bottleneck_analysis.analysis.utilization_bottlenecks && 
                                     solution.bottleneck_analysis.analysis.utilization_bottlenecks.length > 0 && (
                                      <Box sx={{ mb: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                          高稼働率ボトルネック:
                                        </Typography>
                                        {solution.bottleneck_analysis.analysis.utilization_bottlenecks.map((machine: string) => (
                                          <Chip
                                            key={machine}
                                            label={machine}
                                            size="small"
                                            color="error"
                                            sx={{ ml: 0.5 }}
                                          />
                                        ))}
                                      </Box>
                                    )}
                                    {solution.bottleneck_analysis.analysis.recommendations && 
                                     solution.bottleneck_analysis.analysis.recommendations.length > 0 && (
                                      <Alert severity="warning" sx={{ mt: 1 }}>
                                        {solution.bottleneck_analysis.analysis.recommendations[0]}
                                      </Alert>
                                    )}
                                  </>
                                )}
                              </Paper>
                            </Grid>
                          )}
                          
                          {/* Advanced KPIs */}
                          {solution.advanced_kpis && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#e8f5e8' }}>
                                <Typography variant="subtitle2" gutterBottom color="success.main">
                                  <AssessmentIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  総合評価
                                </Typography>
                                {solution.advanced_kpis.overall_score && (
                                  <>
                                    <Typography variant="h4" color="success.main" gutterBottom>
                                      {solution.advanced_kpis.overall_score.grade}
                                    </Typography>
                                    <Typography variant="body2" gutterBottom>
                                      総合スコア: {solution.advanced_kpis.overall_score.total_score?.toFixed(1)}/100
                                    </Typography>
                                    <LinearProgress
                                      variant="determinate"
                                      value={Math.min(solution.advanced_kpis.overall_score.total_score || 0, 100)}
                                      color="success"
                                      sx={{ mb: 1 }}
                                    />
                                    {solution.advanced_kpis.quality_metrics && (
                                      <Typography variant="caption" color="text.secondary">
                                        定時納期率: {((solution.advanced_kpis.quality_metrics.on_time_delivery_rate || 0) * 100).toFixed(1)}%
                                      </Typography>
                                    )}
                                  </>
                                )}
                              </Paper>
                            </Grid>
                          )}
                          
                          {/* Detailed KPIs */}
                          {solution.advanced_kpis && (
                            <Grid item xs={12}>
                              <Collapse in={expandedSections.kpis}>
                                <Grid container spacing={2}>
                                  {solution.advanced_kpis.efficiency_metrics && (
                                    <Grid item xs={12} md={3}>
                                      <Typography variant="subtitle2" gutterBottom>効率性指標</Typography>
                                      <Table size="small">
                                        <TableBody>
                                          <TableRow>
                                            <TableCell>スケジュール効率</TableCell>
                                            <TableCell>{((solution.advanced_kpis.efficiency_metrics.schedule_efficiency || 0) * 100).toFixed(1)}%</TableCell>
                                          </TableRow>
                                          <TableRow>
                                            <TableCell>負荷バランス指数</TableCell>
                                            <TableCell>{((solution.advanced_kpis.efficiency_metrics.load_balancing_index || 0) * 100).toFixed(1)}%</TableCell>
                                          </TableRow>
                                        </TableBody>
                                      </Table>
                                    </Grid>
                                  )}
                                  
                                  {solution.advanced_kpis.quality_metrics && (
                                    <Grid item xs={12} md={3}>
                                      <Typography variant="subtitle2" gutterBottom>品質指標</Typography>
                                      <Table size="small">
                                        <TableBody>
                                          <TableRow>
                                            <TableCell>定時納期率</TableCell>
                                            <TableCell>{((solution.advanced_kpis.quality_metrics.on_time_delivery_rate || 0) * 100).toFixed(1)}%</TableCell>
                                          </TableRow>
                                          <TableRow>
                                            <TableCell>平均遅延時間</TableCell>
                                            <TableCell>{(solution.advanced_kpis.quality_metrics.average_tardiness || 0).toFixed(1)}</TableCell>
                                          </TableRow>
                                        </TableBody>
                                      </Table>
                                    </Grid>
                                  )}
                                  
                                  {solution.advanced_kpis.resource_metrics && (
                                    <Grid item xs={12} md={3}>
                                      <Typography variant="subtitle2" gutterBottom>リソース指標</Typography>
                                      <Table size="small">
                                        <TableBody>
                                          <TableRow>
                                            <TableCell>稼働マシン率</TableCell>
                                            <TableCell>{((solution.advanced_kpis.resource_metrics.active_machine_ratio || 0) * 100).toFixed(1)}%</TableCell>
                                          </TableRow>
                                          <TableRow>
                                            <TableCell>平均作業数</TableCell>
                                            <TableCell>{(solution.advanced_kpis.resource_metrics.operations_per_machine || 0).toFixed(1)}</TableCell>
                                          </TableRow>
                                        </TableBody>
                                      </Table>
                                    </Grid>
                                  )}
                                  
                                  {solution.advanced_kpis.cost_metrics && (
                                    <Grid item xs={12} md={3}>
                                      <Typography variant="subtitle2" gutterBottom>コスト指標</Typography>
                                      <Table size="small">
                                        <TableBody>
                                          <TableRow>
                                            <TableCell>推定アイドルコスト</TableCell>
                                            <TableCell>¥{(solution.advanced_kpis.cost_metrics.estimated_idle_cost || 0).toLocaleString()}</TableCell>
                                          </TableRow>
                                          <TableRow>
                                            <TableCell>推定遅延ペナルティ</TableCell>
                                            <TableCell>¥{(solution.advanced_kpis.cost_metrics.estimated_tardiness_penalty || 0).toLocaleString()}</TableCell>
                                          </TableRow>
                                        </TableBody>
                                      </Table>
                                    </Grid>
                                  )}
                                </Grid>
                              </Collapse>
                              
                              <Box display="flex" justifyContent="center" mt={2}>
                                <Button
                                  variant="outlined"
                                  size="small"
                                  onClick={() => toggleSection('kpis')}
                                  startIcon={expandedSections.kpis ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                                >
                                  {expandedSections.kpis ? '詳細KPIを隠す' : '詳細KPIを表示'}
                                </Button>
                              </Box>
                            </Grid>
                          )}
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {/* Improvement Suggestions */}
                {solution.improvement_suggestions && solution.improvement_suggestions.length > 0 && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          改善提案
                        </Typography>
                        {solution.improvement_suggestions.map((suggestion: string, index: number) => (
                          <Alert key={index} severity="info" sx={{ mb: 1 }}>
                            {suggestion}
                          </Alert>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={4}>
          {/* Results */}
          {showSolution && solution && (
            <Box>
              {/* KPI Summary */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="primary">
                        {solution.metrics?.makespan?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        メイクスパン
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="success.main">
                        {solution.metrics?.total_completion_time?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        総完了時間
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="warning.main">
                        {solution.metrics?.average_flow_time?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        平均フロータイム
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="info.main">
                        {solution.metrics?.machine_utilization ? 
                          (solution.metrics.machine_utilization * 100).toFixed(1) + '%' : 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        機械稼働率
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Grid container spacing={3} sx={{ mt: 2 }}>
                {/* Gantt Chart */}
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        ガントチャート
                      </Typography>
                      {renderGanttChart()}
                    </CardContent>
                  </Card>
                </Grid>

                {/* Solution Metrics */}
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        ジョブスケジュール
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>ジョブ</TableCell>
                              <TableCell>開始時刻</TableCell>
                              <TableCell>完了時刻</TableCell>
                              <TableCell>総時間</TableCell>
                              <TableCell>遅延</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {solution.job_schedules.map((job: any) => (
                              <TableRow key={job.job_id}>
                                <TableCell>{job.job_id}</TableCell>
                                <TableCell>{job.start_time}</TableCell>
                                <TableCell>{job.completion_time}</TableCell>
                                <TableCell>{job.completion_time - job.start_time}</TableCell>
                                <TableCell>
                                  <Chip
                                    label={job.tardiness || 0}
                                    color={job.tardiness > 0 ? 'error' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Machine Utilization */}
                <Grid item xs={12} md={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        マシンスケジュール
                      </Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>マシン</TableCell>
                              <TableCell>稼働率</TableCell>
                              <TableCell>アイドル時間</TableCell>
                              <TableCell>操作数</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {solution.machine_schedules.map((machine: any) => (
                              <TableRow key={machine.machine_id}>
                                <TableCell>{machine.machine_id}</TableCell>
                                <TableCell>
                                  <Chip
                                    label={`${(machine.utilization * 100).toFixed(1)}%`}
                                    color={machine.utilization > 0.8 ? 'error' : machine.utilization > 0.5 ? 'warning' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                                <TableCell>{machine.idle_time}</TableCell>
                                <TableCell>{machine.operations.length}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Advanced Analysis Section */}
                {(solution.bottleneck_analysis || solution.advanced_kpis || solution.critical_path) && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          <AssessmentIcon sx={{ mr: 1 }} />
                          高度分析結果
                        </Typography>
                        
                        <Grid container spacing={3}>
                          {/* Critical Path Analysis */}
                          {solution.critical_path && solution.critical_path.length > 0 && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#fff3e0' }}>
                                <Typography variant="subtitle2" gutterBottom color="primary">
                                  <TimelineIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  クリティカルパス
                                </Typography>
                                <Typography variant="body2" gutterBottom>
                                  作業数: {solution.critical_path.length}
                                </Typography>
                                <Box sx={{ maxHeight: 150, overflowY: 'auto' }}>
                                  {solution.critical_path.map((op: string, index: number) => (
                                    <Chip
                                      key={index}
                                      label={op}
                                      size="small"
                                      color="warning"
                                      sx={{ m: 0.5 }}
                                    />
                                  ))}
                                </Box>
                              </Paper>
                            </Grid>
                          )}
                          
                          {/* Bottleneck Analysis */}
                          {solution.bottleneck_analysis && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#ffebee' }}>
                                <Typography variant="subtitle2" gutterBottom color="error">
                                  <FactoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  ボトルネック分析
                                </Typography>
                                {solution.bottleneck_analysis.analysis && (
                                  <>
                                    <Typography variant="body2" gutterBottom>
                                      ボトルネック数: {solution.bottleneck_analysis.bottleneck_machines?.length || 0}
                                    </Typography>
                                    {solution.bottleneck_analysis.analysis.utilization_bottlenecks && 
                                     solution.bottleneck_analysis.analysis.utilization_bottlenecks.length > 0 && (
                                      <Box sx={{ mb: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                          高稼働率ボトルネック:
                                        </Typography>
                                        {solution.bottleneck_analysis.analysis.utilization_bottlenecks.map((machine: string) => (
                                          <Chip
                                            key={machine}
                                            label={machine}
                                            size="small"
                                            color="error"
                                            sx={{ ml: 0.5 }}
                                          />
                                        ))}
                                      </Box>
                                    )}
                                    {solution.bottleneck_analysis.analysis.recommendations && 
                                     solution.bottleneck_analysis.analysis.recommendations.length > 0 && (
                                      <Alert severity="warning" sx={{ mt: 1 }}>
                                        {solution.bottleneck_analysis.analysis.recommendations[0]}
                                      </Alert>
                                    )}
                                  </>
                                )}
                              </Paper>
                            </Grid>
                          )}
                          
                          {/* Advanced KPIs */}
                          {solution.advanced_kpis && (
                            <Grid item xs={12} md={4}>
                              <Paper elevation={1} sx={{ p: 2, bgcolor: '#e8f5e8' }}>
                                <Typography variant="subtitle2" gutterBottom color="success.main">
                                  <AssessmentIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                                  総合評価
                                </Typography>
                                {solution.advanced_kpis.overall_score && (
                                  <>
                                    <Typography variant="h4" color="success.main" gutterBottom>
                                      {solution.advanced_kpis.overall_score.grade}
                                    </Typography>
                                    <Typography variant="body2" gutterBottom>
                                      総合スコア: {solution.advanced_kpis.overall_score.total_score?.toFixed(1)}/100
                                    </Typography>
                                    <LinearProgress
                                      variant="determinate"
                                      value={Math.min(solution.advanced_kpis.overall_score.total_score || 0, 100)}
                                      color="success"
                                      sx={{ mb: 1 }}
                                    />
                                    {solution.advanced_kpis.quality_metrics && (
                                      <Typography variant="caption" color="text.secondary">
                                        定時納期率: {((solution.advanced_kpis.quality_metrics.on_time_delivery_rate || 0) * 100).toFixed(1)}%
                                      </Typography>
                                    )}
                                  </>
                                )}
                              </Paper>
                            </Grid>
                          )}
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                )}

                {/* Improvement Suggestions */}
                {solution.improvement_suggestions && solution.improvement_suggestions.length > 0 && (
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          改善提案
                        </Typography>
                        {solution.improvement_suggestions.map((suggestion: string, index: number) => (
                          <Alert key={index} severity="info" sx={{ mb: 1 }}>
                            {suggestion}
                          </Alert>
                        ))}
                      </CardContent>
                    </Card>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={5}>
          {/* Realtime Schedule Management */}
          {solution ? (
            <RealtimeScheduleManager 
              solutionData={solution}
              onUpdate={() => {
                // Optionally refresh solution data when realtime updates occur
                setSuccess('スケジュールが更新されました');
              }}
            />
          ) : (
            <Alert severity="info">
              リアルタイム監視を開始するには、まず最適化を実行してください。
            </Alert>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={6}>
          {/* Data Import/Export */}
          <DataImportExport
            jobs={jobs}
            machines={machines}
            currentSolution={solution}
            currentProblem={{
              problem_type: problemType,
              jobs,
              machines,
              optimization_objective: optimizationObjective
            }}
            onJobsImported={(importedJobs) => {
              setJobs(importedJobs);
              setSuccess('ジョブデータをインポートしました');
            }}
            onMachinesImported={(importedMachines) => {
              setMachines(importedMachines);
              setSuccess('マシンデータをインポートしました');
            }}
            onProblemImported={(importedProblem) => {
              setJobs(importedProblem.jobs || []);
              setMachines(importedProblem.machines || []);
              setOptimizationObjective(importedProblem.optimization_objective || 'makespan');
              setProblemType(importedProblem.problem_type || 'job_shop');
              setSuccess('問題定義をインポートしました');
            }}
          />
        </TabPanel>

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

      {/* Template Load Dialog */}
      <Dialog 
        open={templateDialogOpen} 
        onClose={() => setTemplateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>テンプレートから読み込み</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            保存されたテンプレートを選択して、問題設定を読み込みます。
          </Typography>
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>テンプレートを選択</InputLabel>
            <Select
              value={selectedTemplateId}
              onChange={(e) => setSelectedTemplateId(e.target.value)}
              label="テンプレートを選択"
            >
              {availableTemplates.map((template) => (
                <MenuItem key={template.id} value={template.id}>
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {template.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {template.category} | マシン: {template.problem_template?.machines?.length || 0} | ジョブ: {template.problem_template?.jobs?.length || 0}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {selectedTemplateId && (
            <Alert severity="info">
              選択したテンプレートの設定が現在の問題設定を上書きします。
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialogOpen(false)}>キャンセル</Button>
          <Button 
            variant="contained"
            onClick={loadFromTemplate}
            disabled={!selectedTemplateId}
          >
            読み込み
          </Button>
        </DialogActions>
      </Dialog>

      {/* Template Save Dialog */}
      <Dialog 
        open={saveTemplateDialogOpen} 
        onClose={() => setSaveTemplateDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>テンプレートとして保存</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            現在の問題設定をテンプレートとして保存します。
          </Typography>
          
          <TextField
            fullWidth
            label="テンプレート名"
            value={templateForm.name}
            onChange={(e) => setTemplateForm(prev => ({ ...prev, name: e.target.value }))}
            sx={{ mb: 2 }}
            required
          />
          
          <TextField
            fullWidth
            label="説明"
            value={templateForm.description}
            onChange={(e) => setTemplateForm(prev => ({ ...prev, description: e.target.value }))}
            multiline
            rows={3}
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>カテゴリ</InputLabel>
            <Select
              value={templateForm.category}
              onChange={(e) => setTemplateForm(prev => ({ ...prev, category: e.target.value }))}
              label="カテゴリ"
            >
              <MenuItem value="general">一般</MenuItem>
              <MenuItem value="manufacturing">製造業</MenuItem>
              <MenuItem value="project">プロジェクト</MenuItem>
            </Select>
          </FormControl>
          
          <FormControlLabel
            control={
              <Switch
                checked={templateForm.is_public}
                onChange={(e) => setTemplateForm(prev => ({ ...prev, is_public: e.target.checked }))}
              />
            }
            label="公開テンプレートとして保存"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveTemplateDialogOpen(false)}>キャンセル</Button>
          <Button 
            variant="contained"
            onClick={saveAsTemplate}
            disabled={!templateForm.name.trim()}
          >
            保存
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default JobShopScheduling;