import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Tabs,
  Tab,
  Container,
  Button,
  Alert,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField
} from '@mui/material';
import { 
  Security as SecurityIcon, 
  CloudDownload as DownloadIcon, 
  Visibility as VisibilityIcon,
  Settings as SettingsIcon,
  DataObject as DataIcon,
  Assessment as AnalysisIcon,
  Timeline as MonitorIcon,
  Policy as PolicyIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

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
      id={`scrm-tabpanel-${index}`}
      aria-labelledby={`scrm-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `scrm-tab-${index}`,
    'aria-controls': `scrm-tabpanel-${index}`,
  };
}

const SCRMAnalysis: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleData, setSampleData] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [networkView, setNetworkView] = useState('supply_chain');
  const [analysisParams, setAnalysisParams] = useState({
    modelType: 'tts',
    timeHorizon: 10,
    riskLevel: 0.95,
    solver: 'pulp'
  });

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleGenerateSampleData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/scrm/generate-sample-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          benchmark_id: 'default',
          n_plants: 3,
          n_flex: 2,
          seed: 1
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSampleData(data);
      
      // Store session ID if provided
      if (data.session_id) {
        setSessionId(data.session_id);
      }
      
      console.log('Sample data generated:', data);
      
      // Move to the next tab after successful data generation
      setCurrentTab(1);
    } catch (err) {
      console.error('Error generating sample data:', err);
      setError('サンプルデータの生成に失敗しました。もう一度お試しください。');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
        {/* Header */}
        <Paper sx={{ mb: 3, p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <SecurityIcon sx={{ mr: 2, color: 'primary.main', fontSize: 32 }} />
            <Box>
              <Typography variant="h4" component="h1" gutterBottom>
                サプライチェインリスク管理 (SCRM)
              </Typography>
              <Typography variant="subtitle1" color="text.secondary">
                MERIODAS - MEta RIsk Oriented Disruption Analysis System
              </Typography>
            </Box>
          </Box>
          <Typography variant="body2" color="text.secondary">
            サプライチェインの途絶リスクを分析し、重要な拠点の特定と生存時間の計算を行います。
            BOM（部品構成表）、工場データ、輸送ネットワークを用いてリスク評価を実施します。
          </Typography>
        </Paper>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Tabs */}
        <Paper sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="SCRM analysis tabs">
              <Tab label="システム設定" icon={<SettingsIcon />} iconPosition="start" {...a11yProps(0)} />
              <Tab label="データ管理" icon={<DataIcon />} iconPosition="start" {...a11yProps(1)} />
              <Tab label="リスクモデル" icon={<SecurityIcon />} iconPosition="start" {...a11yProps(2)} disabled={!sampleData} />
              <Tab label="分析実行" icon={<AnalysisIcon />} iconPosition="start" {...a11yProps(3)} disabled={!sampleData} />
              <Tab label="結果可視化" icon={<VisibilityIcon />} iconPosition="start" {...a11yProps(4)} disabled={!sampleData} />
              <Tab label="リアルタイム監視" icon={<MonitorIcon />} iconPosition="start" {...a11yProps(5)} disabled={!sampleData} />
              <Tab label="ポリシー管理" icon={<StorageIcon />} iconPosition="start" {...a11yProps(6)} disabled={!sampleData} />
            </Tabs>
          </Box>

          {/* Tab 0: システム設定 */}
          <TabPanel value={currentTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      システム設定
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      SCRMシステムの基本設定を行います
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <TextField
                          fullWidth
                          label="ベンチマークID"
                          value="default"
                          disabled
                          helperText="使用するベンチマーク問題"
                        />
                      </Grid>
                      <Grid item xs={12} md={3}>
                        <TextField
                          fullWidth
                          label="工場数"
                          type="number"
                          value="3"
                          disabled
                          helperText="生成する工場数"
                        />
                      </Grid>
                      <Grid item xs={12} md={3}>
                        <TextField
                          fullWidth
                          label="柔軟性パラメータ"
                          type="number"
                          value="2"
                          disabled
                          helperText="n_flex パラメータ"
                        />
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      サンプルデータ生成
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      ベンチマーク問題からサンプルデータを自動生成します
                    </Typography>
                    
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                      onClick={handleGenerateSampleData}
                      disabled={loading}
                      sx={{ mb: 2 }}
                    >
                      {loading ? '生成中...' : 'サンプルデータを生成'}
                    </Button>

                    {sampleData && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        サンプルデータが正常に生成されました！
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 2: リスクモデル */}
          <TabPanel value={currentTab} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      リスクモデル設定
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      リスク分析のモデルとパラメータを設定してください
                    </Typography>
                    
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 3 }}>
                          <Typography variant="h6" gutterBottom>
                            最適化モデル選択
                          </Typography>
                          
                          <FormControl fullWidth sx={{ mb: 2 }}>
                            <InputLabel>最適化モデル</InputLabel>
                            <Select 
                              value={analysisParams.modelType} 
                              label="最適化モデル"
                              onChange={(e) => setAnalysisParams({...analysisParams, modelType: e.target.value})}
                            >
                              <MenuItem value="tts">Time-to-Survival (TTS)</MenuItem>
                              <MenuItem value="expected_value">期待値最小化モデル</MenuItem>
                              <MenuItem value="cvar">CVaR (条件付きリスク値)</MenuItem>
                            </Select>
                          </FormControl>
                          
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            TTS: 各ノードの生存時間を最大化<br/>
                            期待値: 複数シナリオの期待値最小化<br/>
                            CVaR: リスク回避的な最適化
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 3 }}>
                          <Typography variant="h6" gutterBottom>
                            分析設定
                          </Typography>
                          
                          <TextField
                            fullWidth
                            label="時間地平線"
                            type="number"
                            value={analysisParams.timeHorizon}
                            onChange={(e) => setAnalysisParams({...analysisParams, timeHorizon: parseInt(e.target.value)})}
                            InputProps={{ inputProps: { min: 1, max: 50 } }}
                            sx={{ mb: 2 }}
                          />
                          
                          <TextField
                            fullWidth
                            label="リスクレベル (CVaR用)"
                            type="number"
                            value={analysisParams.riskLevel}
                            onChange={(e) => setAnalysisParams({...analysisParams, riskLevel: parseFloat(e.target.value)})}
                            InputProps={{ 
                              inputProps: { min: 0.1, max: 1.0, step: 0.05 } 
                            }}
                            sx={{ mb: 2 }}
                          />
                          
                          <FormControl fullWidth>
                            <InputLabel>ソルバー</InputLabel>
                            <Select 
                              value={analysisParams.solver} 
                              label="ソルバー"
                              onChange={(e) => setAnalysisParams({...analysisParams, solver: e.target.value})}
                            >
                              <MenuItem value="pulp">PuLP (デフォルト)</MenuItem>
                              <MenuItem value="gurobi">Gurobi</MenuItem>
                              <MenuItem value="cplex">CPLEX</MenuItem>
                            </Select>
                          </FormControl>
                        </Paper>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 3: 分析実行 */}
          <TabPanel value={currentTab} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <AnalysisIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      分析実行
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      設定されたパラメータでSCRM分析を実行します
                    </Typography>
                    
                    <Box sx={{ textAlign: 'center' }}>
                      <Button
                        variant="contained"
                        size="large"
                        onClick={async () => {
                          setAnalysisRunning(true);
                          setError(null);
                          
                          try {
                            // Call the real API endpoint based on model type
                            const endpoint = analysisParams.modelType === 'tts' ? '/api/scrm/analyze-tts' :
                                           analysisParams.modelType === 'expected_value' ? '/api/scrm/analyze-expected-value' :
                                           '/api/scrm/analyze-cvar';
                            
                            const response = await fetch(endpoint, {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json',
                              },
                              body: JSON.stringify({
                                model_type: analysisParams.modelType,
                                time_horizon: analysisParams.timeHorizon,
                                risk_level: analysisParams.riskLevel,
                                solver: analysisParams.solver,
                                session_id: sessionId,  // Include session ID
                                use_sample_data: true  // Use the generated sample data
                              })
                            });
      
                            if (!response.ok) {
                              throw new Error(`HTTP error! status: ${response.status}`);
                            }
      
                            const results = await response.json();
                            setAnalysisResults(results);
                            setCurrentTab(4); // 結果可視化タブに移動
                            
                          } catch (err) {
                            console.error('Analysis error:', err);
                            setError('分析の実行に失敗しました。パラメータを確認してもう一度お試しください。');
                          } finally {
                            setAnalysisRunning(false);
                          }
                        }}
                        disabled={analysisRunning || !sampleData}
                        startIcon={analysisRunning ? <CircularProgress size={20} /> : <AnalysisIcon />}
                        sx={{ minWidth: 200 }}
                      >
                        {analysisRunning ? '分析実行中...' : '分析を実行'}
                      </Button>
                      
                      {!sampleData && (
                        <Alert severity="warning" sx={{ mt: 2 }}>
                          まずシステム設定タブでサンプルデータを生成してください。
                        </Alert>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 4: 結果可視化 */}
          <TabPanel value={currentTab} index={4}>
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box>
                  <Typography variant="h6" gutterBottom>
                    ネットワーク可視化
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    サプライチェーンネットワークとリスク分析結果
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <FormControl size="small" sx={{ minWidth: 150 }}>
                    <InputLabel>表示モード</InputLabel>
                    <Select 
                      value={networkView} 
                      label="表示モード"
                      onChange={(e) => setNetworkView(e.target.value)}
                    >
                      <MenuItem value="supply_chain">サプライチェーン</MenuItem>
                      <MenuItem value="risk_heatmap">リスクヒートマップ</MenuItem>
                      <MenuItem value="tts_analysis">TTS分析</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              </Box>

              {analysisResults ? (
                <Grid container spacing={3}>
                  <Grid item xs={12} lg={8}>
                    <Paper sx={{ p: 2, height: 500 }}>
                      <Typography variant="h6" gutterBottom>
                        {networkView === 'supply_chain' && 'サプライチェーンネットワーク'}
                        {networkView === 'risk_heatmap' && 'リスクヒートマップ'}
                        {networkView === 'tts_analysis' && 'TTS分析結果'}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80%' }}>
                        {networkView === 'supply_chain' && (
                          <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
                            <svg width="100%" height="400" style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}>
                              {/* Generate simple network visualization based on analysis results */}
                              {(() => {
                                // Create simplified network nodes for visualization
                                const nodes = analysisResults.survival_time ? 
                                  analysisResults.survival_time.map((tts: number, idx: number) => ({
                                    id: `node_${idx}`,
                                    name: `ノード${idx + 1}`,
                                    x: (idx % 4) * 2,
                                    y: Math.floor(idx / 4) * 2,
                                    tts: tts,
                                    type: idx < 2 ? 'supplier' : idx < 4 ? 'plant' : 'customer'
                                  })) : 
                                  [
                                    { id: 'node_0', name: 'ノード1', x: 0, y: 0, tts: 8.5, type: 'supplier' },
                                    { id: 'node_1', name: 'ノード2', x: 2, y: 0, tts: 12.3, type: 'plant' },
                                    { id: 'node_2', name: 'ノード3', x: 4, y: 0, tts: 6.8, type: 'customer' }
                                  ];
                                
                                const edges = nodes.length > 1 ? 
                                  nodes.slice(0, -1).map((node: any, idx: number) => ({
                                    from: `node_${idx}`,
                                    to: `node_${idx + 1}`,
                                    weight: 100 + idx * 20
                                  })) : [];

                                return (
                                  <>
                                    {/* Draw edges first */}
                                    {edges.map((edge: any, i: number) => {
                                      const fromNode = nodes.find((n: any) => n.id === edge.from);
                                      const toNode = nodes.find((n: any) => n.id === edge.to);
                                      if (!fromNode || !toNode) return null;
                                      const x1 = (fromNode.x + 1) * 100 + 50;
                                      const y1 = (fromNode.y + 1) * 80 + 100;
                                      const x2 = (toNode.x + 1) * 100 + 50;
                                      const y2 = (toNode.y + 1) * 80 + 100;
                                      return (
                                        <g key={i}>
                                          <line
                                            x1={x1}
                                            y1={y1}
                                            x2={x2}
                                            y2={y2}
                                            stroke="#666"
                                            strokeWidth="2"
                                          />
                                          <text
                                            x={(x1 + x2) / 2}
                                            y={(y1 + y2) / 2 - 5}
                                            textAnchor="middle"
                                            fontSize="10"
                                            fill="#666"
                                          >
                                            {edge.weight}
                                          </text>
                                        </g>
                                      );
                                    })}
                                    
                                    {/* Draw nodes */}
                                    {nodes.map((node: any) => {
                                      const x = (node.x + 1) * 100 + 50;
                                      const y = (node.y + 1) * 80 + 100;
                                      const color = node.type === 'supplier' ? '#2196F3' : 
                                                  node.type === 'plant' ? '#4CAF50' : '#FF9800';
                                      return (
                                        <g key={node.id}>
                                          <circle
                                            cx={x}
                                            cy={y}
                                            r="25"
                                            fill={color}
                                            stroke="white"
                                            strokeWidth="3"
                                          />
                                          <text
                                            x={x}
                                            y={y + 35}
                                            textAnchor="middle"
                                            fontSize="10"
                                            fontWeight="bold"
                                          >
                                            {node.name}
                                          </text>
                                          {node.tts && (
                                            <text
                                              x={x}
                                              y={y}
                                              textAnchor="middle"
                                              fontSize="8"
                                              fill="white"
                                              fontWeight="bold"
                                            >
                                              {node.tts.toFixed(1)}
                                            </text>
                                          )}
                                        </g>
                                      );
                                    })}
                                  </>
                                );
                              })()}
                            </svg>
                            
                            {/* Legend */}
                            <Box sx={{ mt: 2, display: 'flex', gap: 3, justifyContent: 'center' }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#2196F3' }} />
                                <Typography variant="body2">サプライヤー</Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#4CAF50' }} />
                                <Typography variant="body2">工場</Typography>
                              </Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: '#FF9800' }} />
                                <Typography variant="body2">顧客</Typography>
                              </Box>
                            </Box>
                          </Box>
                        )}
                        
                        {networkView === 'risk_heatmap' && (
                          <Box sx={{ width: '100%', height: '100%' }}>
                            <svg width="100%" height="400" style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}>
                              {(() => {
                                // Generate risk visualization based on survival time
                                const nodes = analysisResults.survival_time ? 
                                  analysisResults.survival_time.map((tts: number, idx: number) => {
                                    const maxTts = analysisResults.max_survival_time || Math.max(...analysisResults.survival_time);
                                    const minTts = analysisResults.min_survival_time || Math.min(...analysisResults.survival_time);
                                    const normalizedRisk = 1 - ((tts - minTts) / (maxTts - minTts)); // Inverse: lower survival = higher risk
                                    return {
                                      id: `node_${idx}`,
                                      name: `ノード${idx + 1}`,
                                      x: (idx % 4) * 2,
                                      y: Math.floor(idx / 4) * 2,
                                      risk: isNaN(normalizedRisk) ? 0.5 : normalizedRisk,
                                      tts: tts
                                    };
                                  }) : [];

                                return nodes.map((node: any) => {
                                  const x = (node.x + 1) * 100 + 50;
                                  const y = (node.y + 1) * 80 + 100;
                                  const riskIntensity = node.risk;
                                  const red = Math.floor(255 * riskIntensity);
                                  const green = Math.floor(255 * (1 - riskIntensity));
                                  const color = `rgb(${red}, ${green}, 0)`;
                                  const size = 20 + riskIntensity * 30;
                                  return (
                                    <g key={node.id}>
                                      <circle
                                        cx={x}
                                        cy={y}
                                        r={size}
                                        fill={color}
                                        stroke="white"
                                        strokeWidth="2"
                                        opacity="0.8"
                                      />
                                      <text
                                        x={x}
                                        y={y}
                                        textAnchor="middle"
                                        fontSize="10"
                                        fontWeight="bold"
                                        fill="white"
                                      >
                                        {(node.risk * 100).toFixed(0)}%
                                      </text>
                                      <text
                                        x={x}
                                        y={y + size + 15}
                                        textAnchor="middle"
                                        fontSize="11"
                                        fontWeight="bold"
                                      >
                                        {node.name}
                                      </text>
                                    </g>
                                  );
                                });
                              })()}
                            </svg>
                            
                            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                              <Typography variant="body2" color="text.secondary">
                                リスクレベル: 低 🟢 → 中 🟡 → 高 🔴 (生存時間の逆数でリスクを算出)
                              </Typography>
                            </Box>
                          </Box>
                        )}
                        
                        {networkView === 'tts_analysis' && (
                          <Box sx={{ width: '100%' }}>
                            {analysisResults.model_type === 'tts' && analysisResults.survival_time ? (
                              <Grid container spacing={2}>
                                {analysisResults.survival_time.map((tts: number, index: number) => (
                                  <Grid item xs={6} key={index}>
                                    <Card sx={{ height: 120 }}>
                                      <CardContent>
                                        <Typography variant="h6" gutterBottom>ノード {index + 1}</Typography>
                                        <Typography variant="h3" 
                                          color={tts > 10 ? 'success.main' : tts > 5 ? 'warning.main' : 'error.main'}
                                          sx={{ fontWeight: 'bold' }}>
                                          {tts.toFixed(1)}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                          日間生存可能
                                        </Typography>
                                        
                                        {/* Progress bar showing TTS relative to max */}
                                        <Box sx={{ mt: 1, width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 8 }}>
                                          <Box 
                                            sx={{ 
                                              width: `${Math.min(100, (tts / (analysisResults.max_survival_time || 20)) * 100)}%`, 
                                              bgcolor: tts > 10 ? 'success.main' : tts > 5 ? 'warning.main' : 'error.main',
                                              height: '100%',
                                              borderRadius: 1
                                            }} 
                                          />
                                        </Box>
                                      </CardContent>
                                    </Card>
                                  </Grid>
                                ))}
                              </Grid>
                            ) : (
                              <Alert severity="info">
                                {analysisResults.model_type === 'expected_value' ? '期待値最小化モデルの結果表示' : 
                                 analysisResults.model_type === 'cvar' ? 'CVaRモデルの結果表示' : 
                                 'TTS分析結果がありません'}
                              </Alert>
                            )}
                          </Box>
                        )}
                      </Box>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} lg={4}>
                    <Paper sx={{ p: 2, height: 500, overflowY: 'auto' }}>
                      <Typography variant="h6" gutterBottom>
                        <VisibilityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                        分析サマリー
                      </Typography>
                      
                      <Card sx={{ mb: 2 }}>
                        <CardContent>
                          <Typography variant="subtitle2" gutterBottom>
                            ネットワーク構成
                          </Typography>
                          <Typography variant="body2">
                            ノード数: {analysisResults.total_nodes || '計算中'}<br/>
                            モデル: {analysisResults.model_type === 'tts' ? 'TTS' : 
                                   analysisResults.model_type === 'expected_value' ? '期待値' : 'CVaR'}
                          </Typography>
                        </CardContent>
                      </Card>
                      
                      <Card sx={{ mb: 2 }}>
                        <CardContent>
                          <Typography variant="subtitle2" gutterBottom>
                            最高リスクノード
                          </Typography>
                          <Typography variant="body2">
                            {analysisResults.model_type === 'tts' && analysisResults.critical_nodes ? 
                              `${analysisResults.critical_nodes[0]?.[0] || '計算中'}` : 
                              '分析結果待ち'}
                            <br/>
                            最小TTS: {analysisResults.min_survival_time?.toFixed(1) || 'N/A'} 日
                          </Typography>
                        </CardContent>
                      </Card>
                      
                      <Card>
                        <CardContent>
                          <Typography variant="subtitle2" gutterBottom>
                            平均TTS
                          </Typography>
                          <Typography variant="body2">
                            {analysisResults.average_survival_time?.toFixed(1) || 'N/A'} 日
                          </Typography>
                        </CardContent>
                      </Card>
                    </Paper>
                  </Grid>
                </Grid>
              ) : (
                <Alert severity="info">
                  分析を実行してからネットワーク可視化を表示します
                </Alert>
              )}
            </Box>
          </TabPanel>

          {/* Tab 3: Analysis Results */}
          <TabPanel value={currentTab} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <VisibilityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      結果可視化
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      SCRM分析結果のネットワーク可視化とグラフ
                    </Typography>
                    
                    {analysisResults ? (
                      <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Typography variant="h6">ネットワーク可視化</Typography>
                          <FormControl size="small" sx={{ minWidth: 150 }}>
                            <InputLabel>表示モード</InputLabel>
                            <Select 
                              value={networkView} 
                              label="表示モード"
                              onChange={(e) => setNetworkView(e.target.value)}
                            >
                              <MenuItem value="supply_chain">サプライチェーン</MenuItem>
                              <MenuItem value="risk_heatmap">リスクヒートマップ</MenuItem>
                              <MenuItem value="tts_analysis">TTS分析</MenuItem>
                            </Select>
                          </FormControl>
                        </Box>
                        
                        <Paper sx={{ p: 2, height: 400, mb: 3 }}>
                          {/* Simplified visualization content */}
                          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                            <Typography variant="body1" color="text.secondary">
                              {networkView === 'supply_chain' && 'サプライチェーンネットワーク表示'}
                              {networkView === 'risk_heatmap' && 'リスクヒートマップ表示'}
                              {networkView === 'tts_analysis' && 'TTS分析結果表示'}
                            </Typography>
                          </Box>
                        </Paper>
                        
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>TTS分析</Typography>
                              {analysisResults.survival_time?.map((tts: number, index: number) => (
                                <Box key={index} sx={{ 
                                  display: 'flex', 
                                  justifyContent: 'space-between', 
                                  p: 1, 
                                  bgcolor: tts > 10 ? 'success.light' : tts > 5 ? 'warning.light' : 'error.light',
                                  borderRadius: 1, 
                                  mb: 1 
                                }}>
                                  <Typography variant="body2">ノード {index + 1}</Typography>
                                  <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                                    {tts.toFixed(1)}日
                                  </Typography>
                                </Box>
                              ))}
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>分析サマリー</Typography>
                              <Grid container spacing={1}>
                                <Grid item xs={12}>
                                  <Typography variant="body2">総ノード数: {analysisResults.total_nodes || 0}</Typography>
                                </Grid>
                                <Grid item xs={12}>
                                  <Typography variant="body2">平均TTS: {analysisResults.average_survival_time?.toFixed(1) || 'N/A'} 日</Typography>
                                </Grid>
                                <Grid item xs={12}>
                                  <Typography variant="body2">最小TTS: {analysisResults.min_survival_time?.toFixed(1) || 'N/A'} 日</Typography>
                                </Grid>
                              </Grid>
                            </Paper>
                          </Grid>
                        </Grid>
                      </Box>
                    ) : (
                      <Alert severity="info">
                        分析を実行してから結果可視化を表示します
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 5: リアルタイム監視 */}
          <TabPanel value={currentTab} index={5}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <MonitorIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      リアルタイム監視
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      サプライチェーンリスクのリアルタイム監視とアラート
                    </Typography>
                    
                    {analysisResults ? (
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                            <Typography variant="h4" color="white">
                              {analysisResults.survival_time?.filter((tts: number) => tts > 10).length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">安全なノード</Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                            <Typography variant="h4" color="white">
                              {analysisResults.survival_time?.filter((tts: number) => tts > 5 && tts <= 10).length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">注意が必要なノード</Typography>
                          </Paper>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <Paper elevation={1} sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light' }}>
                            <Typography variant="h4" color="white">
                              {analysisResults.survival_time?.filter((tts: number) => tts <= 5).length || 0}
                            </Typography>
                            <Typography variant="subtitle2" color="white">高リスクノード</Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    ) : (
                      <Alert severity="info">
                        リアルタイム監視を開始するには、まず分析を実行してください。
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 6: ポリシー管理 */}
          <TabPanel value={currentTab} index={6}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                      ポリシー管理
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      リスク管理ポリシーの設定とエクスポート機能
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="h6" gutterBottom>
                            リスク闾値設定
                          </Typography>
                          <TextField
                            fullWidth
                            label="高リスク闾値 (TTS)"
                            type="number"
                            value="5"
                            sx={{ mb: 2 }}
                          />
                          <TextField
                            fullWidth
                            label="中リスク闾値 (TTS)"
                            type="number"
                            value="10"
                          />
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="h6" gutterBottom>
                            エクスポート機能
                          </Typography>
                          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Button
                              variant="outlined"
                              startIcon={<DownloadIcon />}
                              disabled={!analysisResults}
                            >
                              分析結果をエクスポート
                            </Button>
                            <Button
                              variant="outlined"
                              startIcon={<DownloadIcon />}
                              disabled={!analysisResults}
                            >
                              リスクレポート生成
                            </Button>
                          </Box>
                        </Paper>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Paper>
      </Box>
    </Container>
  );
};

export default SCRMAnalysis;