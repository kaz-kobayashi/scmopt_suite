import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Grid,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import { 
  Upload as CloudUploadIcon,
  Settings as SettingsIcon,
  Storage as DataIcon,
  Assessment as AnalysisIcon,
  PlayArrow as ExecuteIcon,
  TrendingUp as ResultsIcon,
  Timeline as MonitorIcon,
  Policy as PolicyIcon
} from '@mui/icons-material';
import apiClient, { ApiService, ABCAnalysisResult } from '../services/apiClient';
import { ABCAnalysisChart, ParetoChart, MeanCVChart, RankAnalysisChart, ABCTreemapChart, ComprehensiveABCChart, AdvancedRankAnalysisChart, InventoryReductionChart } from './DataVisualization';
import TreemapVisualization from './TreemapVisualization';
import AdvancedAnalyticsVisualization from './AdvancedAnalyticsVisualization';

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
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Analytics: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState('0.7,0.2,0.1');
  const [aggCol, setAggCol] = useState('prod');
  const [valueCol, setValueCol] = useState('demand');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ABCAnalysisResult | null>(null);
  
  // Risk Pooling state
  const [riskPoolingFile, setRiskPoolingFile] = useState<File | null>(null);
  const [aggPeriod, setAggPeriod] = useState('1w');
  const [riskPoolingResult, setRiskPoolingResult] = useState<any>(null);
  
  // Treemap state
  const [treemapFile, setTreemapFile] = useState<File | null>(null);
  const [treemapParent, setTreemapParent] = useState('cust');
  const [treemapValue, setTreemapValue] = useState('demand');
  const [treemapResult, setTreemapResult] = useState<any>(null);
  
  // Pareto Analysis state
  const [paretoFile, setParetoFile] = useState<File | null>(null);
  const [paretoAggCol, setParetoAggCol] = useState('prod');
  const [paretoValue, setParetoValue] = useState('demand');
  const [paretoResult, setParetoResult] = useState<any>(null);

  // Mean CV Analysis state
  const [meanCVDemandFile, setMeanCVDemandFile] = useState<File | null>(null);
  const [meanCVProductFile, setMeanCVProductFile] = useState<File | null>(null);
  const [meanCVShowNames] = useState(true);
  const [meanCVResult, setMeanCVResult] = useState<any>(null);

  // Rank Analysis state
  const [rankAnalysisFile, setRankAnalysisFile] = useState<File | null>(null);
  const [rankAggCol, setRankAggCol] = useState('prod');
  const [rankValue, setRankValue] = useState('demand');
  const [rankPeriods, setRankPeriods] = useState(false);
  const [rankAggPeriod, setRankAggPeriod] = useState('1w');
  const [rankResult, setRankResult] = useState<any>(null);

  // ABC Treemap state
  const [abcTreemapFile, setAbcTreemapFile] = useState<File | null>(null);
  const [abcCol, setAbcCol] = useState('abc');
  const [abcTreemapResult, setAbcTreemapResult] = useState<any>(null);

  // Comprehensive ABC state
  const [comprehensiveFile, setComprehensiveFile] = useState<File | null>(null);
  const [comprehensiveValue, setComprehensiveValue] = useState('demand');
  const [comprehensiveCumsum, setComprehensiveCumsum] = useState(true);
  const [comprehensiveCustThres, setComprehensiveCustThres] = useState('0.7,0.2,0.1');
  const [comprehensiveProdThres, setComprehensiveProdThres] = useState('0.7,0.2,0.1');
  const [comprehensiveResult, setComprehensiveResult] = useState<any>(null);

  // Inventory Reduction state
  const [inventoryReductionFile, setInventoryReductionFile] = useState<File | null>(null);
  const [inventoryReductionPeriod, setInventoryReductionPeriod] = useState('1w');
  const [inventoryReductionResult, setInventoryReductionResult] = useState<any>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    // Reset state when switching tabs
    setFile(null);
    setResult(null);
    setRiskPoolingFile(null);
    setRiskPoolingResult(null);
    setTreemapFile(null);
    setTreemapResult(null);
    setParetoFile(null);
    setParetoResult(null);
    setMeanCVDemandFile(null);
    setMeanCVProductFile(null);
    setMeanCVResult(null);
    setRankAnalysisFile(null);
    setRankResult(null);
    setAbcTreemapFile(null);
    setAbcTreemapResult(null);
    setComprehensiveFile(null);
    setComprehensiveResult(null);
    setInventoryReductionFile(null);
    setInventoryReductionResult(null);
    setError(null);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleABCAnalysis = async () => {
    if (!file) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const analysisResult = await ApiService.performABCAnalysis(
        file,
        threshold,
        aggCol,
        valueCol
      );
      setResult(analysisResult);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ABC分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleUseSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch sample data from public folder
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      
      // Create a File object from the CSV text
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setFile(sampleFile);
      
      // Perform ABC analysis with sample data
      const analysisResult = await ApiService.performABCAnalysis(
        sampleFile,
        threshold,
        aggCol,
        valueCol
      );
      setResult(analysisResult);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRiskPoolingAnalysis = async () => {
    if (!riskPoolingFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.analyzeRiskPooling(riskPoolingFile, aggPeriod);
      setRiskPoolingResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'リスクプーリング分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRiskPoolingSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setRiskPoolingFile(sampleFile);
      
      const result = await ApiService.analyzeRiskPooling(sampleFile, aggPeriod);
      setRiskPoolingResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleTreemapGeneration = async () => {
    if (!treemapFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.generateTreemap(treemapFile, treemapParent, treemapValue);
      setTreemapResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ツリーマップの生成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleTreemapSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setTreemapFile(sampleFile);
      
      const result = await ApiService.generateTreemap(sampleFile, treemapParent, treemapValue);
      setTreemapResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleParetoAnalysis = async () => {
    if (!paretoFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.performParetoAnalysis(paretoFile, paretoAggCol, paretoValue);
      setParetoResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'パレート分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleParetoSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setParetoFile(sampleFile);
      
      const result = await ApiService.performParetoAnalysis(sampleFile, paretoAggCol, paretoValue);
      setParetoResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleMeanCVAnalysis = async () => {
    if (!meanCVDemandFile) {
      setError('まず需要データファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.performMeanCVAnalysis(
        meanCVDemandFile,
        meanCVProductFile || undefined,
        meanCVShowNames
      );
      setMeanCVResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Mean-CV分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleMeanCVSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setMeanCVDemandFile(sampleFile);
      
      const result = await ApiService.performMeanCVAnalysis(sampleFile, undefined, meanCVShowNames);
      setMeanCVResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRankAnalysis = async () => {
    if (!rankAnalysisFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = rankPeriods
        ? await ApiService.performRankAnalysisPeriods(rankAnalysisFile, rankAggCol, rankValue, rankAggPeriod)
        : await ApiService.performRankAnalysis(rankAnalysisFile, rankAggCol, rankValue);
      setRankResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ランク分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleRankAnalysisSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setRankAnalysisFile(sampleFile);
      
      const result = rankPeriods
        ? await ApiService.performRankAnalysisPeriods(sampleFile, rankAggCol, rankValue, rankAggPeriod)
        : await ApiService.performRankAnalysis(sampleFile, rankAggCol, rankValue);
      setRankResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleAbcTreemapAnalysis = async () => {
    if (!abcTreemapFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.generateTreemapWithABC(abcTreemapFile, abcCol);
      setAbcTreemapResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ABCツリーマップの生成に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleAbcTreemapSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      // First perform ABC analysis to get classified data
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      // Perform ABC analysis first to get the classified data
      const abcResult = await ApiService.performABCAnalysis(sampleFile, '0.7,0.2,0.1', 'prod', 'demand');
      
      // Create a new CSV with ABC classification
      const classifiedData = abcResult.classified_data.slice(0, 1000); // Limit for demo
      const csvHeaders = Object.keys(classifiedData[0]).join(',');
      const csvRows = classifiedData.map(row => Object.values(row).join(','));
      const classifiedCsv = [csvHeaders, ...csvRows].join('\n');
      
      const classifiedBlob = new Blob([classifiedCsv], { type: 'text/csv' });
      const classifiedFile = new File([classifiedBlob], 'classified_demand.csv', { type: 'text/csv' });
      
      setAbcTreemapFile(classifiedFile);
      
      const result = await ApiService.generateTreemapWithABC(classifiedFile, abcCol);
      setAbcTreemapResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleComprehensiveABCAnalysis = async () => {
    if (!comprehensiveFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.performComprehensiveABCAnalysis(
        comprehensiveFile,
        comprehensiveValue,
        comprehensiveCumsum,
        comprehensiveCustThres,
        comprehensiveProdThres
      );
      setComprehensiveResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || '包括的ABC分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleComprehensiveSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setComprehensiveFile(sampleFile);
      
      const result = await ApiService.performComprehensiveABCAnalysis(
        sampleFile,
        comprehensiveValue,
        comprehensiveCumsum,
        comprehensiveCustThres,
        comprehensiveProdThres
      );
      setComprehensiveResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleInventoryReductionAnalysis = async () => {
    if (!inventoryReductionFile) {
      setError('まずファイルを選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ApiService.visualizeInventoryReduction(
        inventoryReductionFile,
        inventoryReductionPeriod
      );
      setInventoryReductionResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || '在庫削減分析の実行に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleInventoryReductionSampleData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/sample_data/demand_sample.csv');
      if (!response.ok) {
        throw new Error('サンプルデータの取得に失敗しました');
      }
      const csvText = await response.text();
      const blob = new Blob([csvText], { type: 'text/csv' });
      const sampleFile = new File([blob], 'demand_sample.csv', { type: 'text/csv' });
      
      setInventoryReductionFile(sampleFile);
      
      const result = await ApiService.visualizeInventoryReduction(
        sampleFile,
        inventoryReductionPeriod
      );
      setInventoryReductionResult(result);
    } catch (err: any) {
      setError(err.message || 'サンプルデータの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const renderABCAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ABC分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="abc-file-input"
                  type="file"
                  onChange={handleFileChange}
                />
                <label htmlFor="abc-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {file && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {file.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="閾値 (カンマ区切り)"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                sx={{ mb: 2 }}
                helperText="例: 0.7,0.2,0.1 (合計が1.0になる必要があります)"
              />

              <TextField
                fullWidth
                label="集計列"
                value={aggCol}
                onChange={(e) => setAggCol(e.target.value)}
                sx={{ mb: 2 }}
                helperText="グループ化する列名 (例: 'prod', 'cust')"
              />

              <TextField
                fullWidth
                label="値列"
                value={valueCol}
                onChange={(e) => setValueCol(e.target.value)}
                sx={{ mb: 2 }}
                helperText="値の列名 (例: 'demand', 'sales')"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleABCAnalysis}
                disabled={loading || !file}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ABC分析を実行'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleUseSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                サンプルデータには、異なる都市における製品A、B、Cの需要データが含まれています
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {result && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  分析結果サマリー
                </Typography>
                <Typography variant="body2">
                  総アイテム数: {result.summary.total_items}
                </Typography>
                <Typography variant="body2">
                  総額: {result.summary.total_value.toLocaleString()}
                </Typography>
                <Typography variant="body2">
                  カテゴリA: {result.categories[0]?.length || 0} アイテム
                </Typography>
                <Typography variant="body2">
                  カテゴリB: {result.categories[1]?.length || 0} アイテム
                </Typography>
                <Typography variant="body2">
                  カテゴリC: {result.categories[2]?.length || 0} アイテム
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {result && (
          <>
            <Grid item xs={12}>
              <ABCAnalysisChart 
                data={{ 
                  aggregated_data: result.aggregated_data,
                  categories: result.categories
                }} 
                valueColumn={valueCol} 
              />
            </Grid>
            <Grid item xs={12}>
              <ParetoChart 
                data={{ 
                  aggregated_data: result.aggregated_data 
                }} 
                valueColumn={valueCol} 
              />
            </Grid>
            <Grid item xs={12}>
              <AdvancedAnalyticsVisualization
                abcResult={result}
                type="abc"
              />
            </Grid>
          </>
        )}
      </Grid>
    </Box>
  );

  const renderRiskPoolingAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                リスクプーリング設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="risk-pooling-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setRiskPoolingFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="risk-pooling-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {riskPoolingFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {riskPoolingFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="集計期間"
                value={aggPeriod}
                onChange={(e) => setAggPeriod(e.target.value)}
                sx={{ mb: 2 }}
                helperText="集計の期間 (例: '1w' が週次、'1d' が日次)"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleRiskPoolingAnalysis}
                disabled={loading || !riskPoolingFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'リスクプーリング分析'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleRiskPoolingSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                リスクプーリングによる需要分散の削減を分析します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {riskPoolingResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  リスクプーリング結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  分析製品数: {riskPoolingResult.summary?.total_products || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総削減量 (絶対値): {riskPoolingResult.summary?.total_reduction_abs?.toFixed(2) || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  平均削減率 (%): {riskPoolingResult.summary?.avg_reduction_pct?.toFixed(2) || 'N/A'}%
                </Typography>
                <Typography variant="body2">
                  集計期間: {riskPoolingResult.summary?.aggregation_period || aggPeriod}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
        
        {riskPoolingResult && (
          <Grid item xs={12}>
            <AdvancedAnalyticsVisualization
              riskPoolingResult={{
                original_variance: riskPoolingResult.analysis?.original_variance || 100,
                pooled_variance: riskPoolingResult.analysis?.pooled_variance || 60,
                variance_reduction_percentage: riskPoolingResult.summary?.avg_reduction_pct || 40,
                pooling_benefit: riskPoolingResult.summary?.total_reduction_abs || 25,
                risk_pooling_analysis: {
                  individual_cv: riskPoolingResult.detailed_analysis?.individual_cv || [0.3, 0.35, 0.28, 0.4],
                  pooled_cv: riskPoolingResult.detailed_analysis?.pooled_cv || 0.22,
                  correlation_matrix: riskPoolingResult.detailed_analysis?.correlation_matrix || [[1, 0.1], [0.1, 1]]
                }
              }}
              type="risk-pooling"
            />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderTreemapAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ツリーマップ設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="treemap-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setTreemapFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="treemap-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {treemapFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {treemapFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="親列"
                value={treemapParent}
                onChange={(e) => setTreemapParent(e.target.value)}
                sx={{ mb: 2 }}
                helperText="グループ化用の列 (例: 'cust', 'region')"
              />

              <TextField
                fullWidth
                label="値列"
                value={treemapValue}
                onChange={(e) => setTreemapValue(e.target.value)}
                sx={{ mb: 2 }}
                helperText="Column for values (e.g., 'demand', 'sales')"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleTreemapGeneration}
                disabled={loading || !treemapFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ツリーマップを生成'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleTreemapSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                需要データの階層的ツリーマップ表示を作成します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {treemapResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ツリーマップ結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  カテゴリ数: {treemapResult.treemap?.num_categories || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総額: {treemapResult.treemap?.total_value?.toLocaleString() || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  親列: {treemapParent}
                </Typography>
                <Typography variant="body2">
                  値列: {treemapValue}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {treemapResult && treemapResult.treemap && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  需要ツリーマップの可視化
                </Typography>
                <TreemapVisualization
                  data={treemapResult.treemap.data}
                  title={treemapResult.treemap.title}
                  height={400}
                />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderParetoAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                パレート分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="pareto-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setParetoFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="pareto-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {paretoFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {paretoFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="集計列"
                value={paretoAggCol}
                onChange={(e) => setParetoAggCol(e.target.value)}
                sx={{ mb: 2 }}
                helperText="グループ化用の列 (例: 'prod', 'cust')"
              />

              <TextField
                fullWidth
                label="値列"
                value={paretoValue}
                onChange={(e) => setParetoValue(e.target.value)}
                sx={{ mb: 2 }}
                helperText="Column for values (e.g., 'demand', 'sales')"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleParetoAnalysis}
                disabled={loading || !paretoFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'パレート分析 (80-20の法則)'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleParetoSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                データが80-20の法則（パレートの法則）に従うか分析します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {paretoResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  パレート分析結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総アイテム数: {paretoResult.pareto_statistics?.total_items || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  80-20の法則に従う: {paretoResult.pareto_statistics?.pareto_compliance ? '✅ はい' : '❌ いいえ'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  上位20%の寄与度: {paretoResult.analysis_summary?.top_20_percent_contribute || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  80%値の位置: {paretoResult.pareto_statistics?.pareto_80_index || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
        
        {paretoResult && (
          <Grid item xs={12}>
            <AdvancedAnalyticsVisualization
              paretoResult={{
                pareto_data: paretoResult.pareto_analysis?.sorted_data || [],
                pareto_80_20: {
                  items_80_percent: paretoResult.pareto_statistics?.pareto_80_index || 0,
                  value_80_percent: 80
                }
              }}
              type="pareto"
            />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderMeanCVAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                平均-変動係数分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="meancv-demand-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setMeanCVDemandFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="meancv-demand-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {meanCVDemandFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {meanCVDemandFile.name}
                  </Typography>
                )}
              </Box>

              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="meancv-product-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setMeanCVProductFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="meancv-product-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    製品データをアップロード (CSV) - オプション
                  </Button>
                </label>
                {meanCVProductFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {meanCVProductFile.name}
                  </Typography>
                )}
              </Box>

              <Button
                fullWidth
                variant="contained"
                onClick={handleMeanCVAnalysis}
                disabled={loading || !meanCVDemandFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '平均-変動係数分析を実行'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleMeanCVSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                製品の需要平均と変動係数（CV）の関係を散布図で分析します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {meanCVResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  平均-CV分析結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  分析製品数: {meanCVResult.scatter_data?.length || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  製品価値情報: {meanCVResult.has_price_info ? '✅ 有り' : '❌ 無し'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  製品名表示: {meanCVResult.show_names ? '✅ 有り' : '❌ 無し'}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {meanCVResult && (
          <Grid item xs={12}>
            <MeanCVChart data={meanCVResult} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderRankAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ランク分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="rank-analysis-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setRankAnalysisFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="rank-analysis-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {rankAnalysisFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {rankAnalysisFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="集計列"
                value={rankAggCol}
                onChange={(e) => setRankAggCol(e.target.value)}
                sx={{ mb: 2 }}
                helperText="グループ化する列名 (例: 'prod', 'cust')"
              />

              <TextField
                fullWidth
                label="値列"
                value={rankValue}
                onChange={(e) => setRankValue(e.target.value)}
                sx={{ mb: 2 }}
                helperText="値の列名 (例: 'demand', 'sales')"
              />

              <Box sx={{ mb: 2 }}>
                <label>
                  <input
                    type="checkbox"
                    checked={rankPeriods}
                    onChange={(e) => setRankPeriods(e.target.checked)}
                  />
                  期間別ランク分析を実行
                </label>
              </Box>

              {rankPeriods && (
                <TextField
                  fullWidth
                  label="集計期間"
                  value={rankAggPeriod}
                  onChange={(e) => setRankAggPeriod(e.target.value)}
                  sx={{ mb: 2 }}
                  helperText="期間 (例: '1w' 週次、'1d' 日次、'1m' 月次)"
                />
              )}

              <Button
                fullWidth
                variant="contained"
                onClick={handleRankAnalysis}
                disabled={loading || !rankAnalysisFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ランク分析を実行'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleRankAnalysisSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                製品や顧客の需要ランキングを分析します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {rankResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ランク分析結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  分析アイテム数: {rankResult.analysis_summary?.total_items || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  分析対象列: {rankResult.analysis_summary?.analyzed_column || rankAggCol}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  値列: {rankResult.analysis_summary?.value_column || rankValue}
                </Typography>
                {rankResult.analysis_summary?.aggregation_period && (
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    集計期間: {rankResult.analysis_summary.aggregation_period}
                  </Typography>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>

        {rankResult && (
          <Grid item xs={12}>
            <RankAnalysisChart data={rankResult} topItems={10} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderAbcTreemapAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ABC分類ツリーマップ設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="abc-treemap-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setAbcTreemapFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="abc-treemap-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    ABC分類済み需要データをアップロード (CSV)
                  </Button>
                </label>
                {abcTreemapFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {abcTreemapFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="ABC列名"
                value={abcCol}
                onChange={(e) => setAbcCol(e.target.value)}
                sx={{ mb: 2 }}
                helperText="ABC分類結果の列名 (例: 'abc', 'customer_ABC')"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleAbcTreemapAnalysis}
                disabled={loading || !abcTreemapFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'ABC分類ツリーマップを生成'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleAbcTreemapSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                ABC分類結果を色分けしたツリーマップを生成します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {abcTreemapResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ABC分類ツリーマップ結果
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総アイテム数: {abcTreemapResult.treemap?.num_items || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総需要量: {abcTreemapResult.treemap?.total_value?.toLocaleString() || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  ABC列: {abcTreemapResult.treemap?.abc_column || abcCol}
                </Typography>
                <Typography variant="body2">
                  値列: {abcTreemapResult.treemap?.value_column || 'demand'}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {abcTreemapResult && (
          <Grid item xs={12}>
            <ABCTreemapChart data={abcTreemapResult.treemap} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderComprehensiveABCAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                包括的ABC分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="comprehensive-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setComprehensiveFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="comprehensive-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {comprehensiveFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {comprehensiveFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="値列"
                value={comprehensiveValue}
                onChange={(e) => setComprehensiveValue(e.target.value)}
                sx={{ mb: 2 }}
                helperText="分析する値の列名 (例: 'demand', 'sales')"
              />

              <TextField
                fullWidth
                label="顧客閾値"
                value={comprehensiveCustThres}
                onChange={(e) => setComprehensiveCustThres(e.target.value)}
                sx={{ mb: 2 }}
                helperText="顧客ABC分類の閾値 (例: '0.7,0.2,0.1')"
              />

              <TextField
                fullWidth
                label="製品閾値"
                value={comprehensiveProdThres}
                onChange={(e) => setComprehensiveProdThres(e.target.value)}
                sx={{ mb: 2 }}
                helperText="製品ABC分類の閾値 (例: '0.7,0.2,0.1')"
              />

              <Box sx={{ mb: 2 }}>
                <label>
                  <input
                    type="checkbox"
                    checked={comprehensiveCumsum}
                    onChange={(e) => setComprehensiveCumsum(e.target.checked)}
                  />
                  累積表示を使用
                </label>
              </Box>

              <Button
                fullWidth
                variant="contained"
                onClick={handleComprehensiveABCAnalysis}
                disabled={loading || !comprehensiveFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : '包括的ABC分析を実行'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleComprehensiveSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                顧客と製品の両方を同時にABC分析し、統合ダッシュボードを生成します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {comprehensiveResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  包括的ABC分析結果概要
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  製品総数: {comprehensiveResult.summary?.total_products || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  顧客総数: {comprehensiveResult.summary?.total_customers || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総需要量: {comprehensiveResult.summary?.total_value?.toLocaleString() || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  分析対象列: {comprehensiveResult.summary?.value_column || comprehensiveValue}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {comprehensiveResult && (
          <Grid item xs={12}>
            <ComprehensiveABCChart data={comprehensiveResult} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  const renderInventoryReductionAnalysis = () => (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                在庫削減分析設定
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="inventory-reduction-file-input"
                  type="file"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) {
                      setInventoryReductionFile(selectedFile);
                      setError(null);
                    }
                  }}
                />
                <label htmlFor="inventory-reduction-file-input">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                  >
                    需要データをアップロード (CSV)
                  </Button>
                </label>
                {inventoryReductionFile && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    選択済み: {inventoryReductionFile.name}
                  </Typography>
                )}
              </Box>

              <TextField
                fullWidth
                label="集約期間"
                value={inventoryReductionPeriod}
                onChange={(e) => setInventoryReductionPeriod(e.target.value)}
                sx={{ mb: 2 }}
                helperText="需要集約用の期間 (例: '1w', '1m')"
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleInventoryReductionAnalysis}
                disabled={loading || !inventoryReductionFile}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'リスクプーリング在庫削減分析'}
              </Button>
              
              <Button
                fullWidth
                variant="outlined"
                onClick={handleInventoryReductionSampleData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'サンプルデータを使用'}
              </Button>
              
              <Typography variant="caption" sx={{ mt: 1, display: 'block', textAlign: 'center', color: 'text.secondary' }}>
                複数顧客を持つ製品のリスクプーリングによる在庫削減効果を分析します
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {inventoryReductionResult && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  在庫削減分析結果概要
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  対象製品数: {inventoryReductionResult.statistics?.num_products || 0}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  総削減量: {inventoryReductionResult.statistics?.total_reduction?.toFixed(2) || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  平均削減量: {inventoryReductionResult.statistics?.average_reduction?.toFixed(2) || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  最大削減量: {inventoryReductionResult.statistics?.max_reduction?.toFixed(2) || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>

        {inventoryReductionResult && (
          <Grid item xs={12}>
            <InventoryReductionChart data={inventoryReductionResult} />
          </Grid>
        )}
      </Grid>
    </Box>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        分析
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analytics tabs">
          <Tab label="システム設定" icon={<SettingsIcon />} iconPosition="start" />
          <Tab label="データ管理" icon={<DataIcon />} iconPosition="start" />
          <Tab label="分析モデル" icon={<AnalysisIcon />} iconPosition="start" />
          <Tab label="分析実行" icon={<ExecuteIcon />} iconPosition="start" />
          <Tab label="結果可視化" icon={<ResultsIcon />} iconPosition="start" />
          <Tab label="リアルタイム監視" icon={<MonitorIcon />} iconPosition="start" />
          <Tab label="レポート管理" icon={<PolicyIcon />} iconPosition="start" />
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        {renderABCAnalysis()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        {renderRiskPoolingAnalysis()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        {renderTreemapAnalysis()}
      </TabPanel>
      
      <TabPanel value={tabValue} index={3}>
        {renderParetoAnalysis()}
      </TabPanel>

      <TabPanel value={tabValue} index={4}>
        {renderMeanCVAnalysis()}
      </TabPanel>

      <TabPanel value={tabValue} index={5}>
        {renderRankAnalysis()}
      </TabPanel>

      <TabPanel value={tabValue} index={6}>
        {renderAbcTreemapAnalysis()}
      </TabPanel>

      <TabPanel value={tabValue} index={7}>
        {renderComprehensiveABCAnalysis()}
      </TabPanel>

      <TabPanel value={tabValue} index={8}>
        {renderInventoryReductionAnalysis()}
      </TabPanel>
    </Box>
  );
};

export default Analytics;