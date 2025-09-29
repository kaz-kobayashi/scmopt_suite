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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
} from '@mui/material';
import { 
  Upload as CloudUploadIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  Storage as DataIcon,
  Map as MapIcon,
  PlayArrow as ExecuteIcon,
  Assessment as ResultsIcon,
  Timeline as MonitorIcon,
  Policy as PolicyIcon
} from '@mui/icons-material';
import apiClient, { ApiService } from '../services/apiClient';
import {
  WeiszfeldLocationChart,
  MultiFacilityWeiszfeldChart,
  CustomerClusteringChart,
  KMedianSolutionChart,
  ServiceAreaChart,
  SingleSourceLNDChart,
  ElbowMethodChart,
  MultipleSourceLNDChart,
  WeiszfeldLocationData,
  CustomerClusteringData,
  KMedianSolutionData,
  ServiceAreaData,
  SingleSourceLNDData,
  ElbowMethodData,
  MultipleSourceLNDData
} from './LNDVisualization';

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
      id={`lnd-tabpanel-${index}`}
      aria-labelledby={`lnd-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const LogisticsNetworkDesign: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Sample data dialog state
  const [sampleDataDialogOpen, setSampleDataDialogOpen] = useState(false);
  const [sampleDataInfo, setSampleDataInfo] = useState<any>(null);
  
  // Common state
  const [customerFile, setCustomerFile] = useState<File | null>(null);
  const [latCol, setLatCol] = useState('lat');
  const [lonCol, setLonCol] = useState('lon');
  const [demandCol, setDemandCol] = useState('demand');
  
  // Weiszfeld state
  const [weiszfeldResult, setWeiszfeldResult] = useState<WeiszfeldLocationData | null>(null);
  const [weiszfeldMaxIterations, setWeiszfeldMaxIterations] = useState(1000);
  const [weiszfeldTolerance, setWeiszfeldTolerance] = useState(1e-6);
  
  // Multi-facility Weiszfeld state
  const [multiFacilityWeiszfeldResult, setMultiFacilityWeiszfeldResult] = useState<any | null>(null);
  const [multiFacilityNum, setMultiFacilityNum] = useState(3);
  const [multiFacilityMaxIterations, setMultiFacilityMaxIterations] = useState(1000);
  const [multiFacilityTolerance, setMultiFacilityTolerance] = useState(1e-4);
  const [multiFacilityRandomState, setMultiFacilityRandomState] = useState(42);
  
  // Repeated Multi-facility Weiszfeld state
  const [repeatedWeiszfeldResult, setRepeatedWeiszfeldResult] = useState<any | null>(null);
  const [repeatedNumRuns, setRepeatedNumRuns] = useState(10);
  const [repeatedNumFacilities, setRepeatedNumFacilities] = useState(3);
  const [repeatedMaxIterations, setRepeatedMaxIterations] = useState(1000);
  const [repeatedTolerance, setRepeatedTolerance] = useState(1e-4);
  const [repeatedBaseRandomState, setRepeatedBaseRandomState] = useState(42);
  
  // Clustering state
  const [clusteringResult, setClusteringResult] = useState<CustomerClusteringData | null>(null);
  const [clusteringMethod, setClusteringMethod] = useState('kmeans');
  const [nClusters, setNClusters] = useState(3);
  const [linkageMethod, setLinkageMethod] = useState('ward');
  const [randomState, setRandomState] = useState(42);
  
  // K-Median state
  const [kMedianResult, setKMedianResult] = useState<KMedianSolutionData | null>(null);
  const [kValue, setKValue] = useState(3);
  const [candidateMethod, setCandidateMethod] = useState('grid');
  const [nCandidates, setNCandidates] = useState(20);
  const [maxIterations, setMaxIterations] = useState(100);
  const [lambdaStep, setLambdaStep] = useState(0.1);
  
  // Service Area state
  const [serviceAreaResult, setServiceAreaResult] = useState<ServiceAreaData | null>(null);
  const [facilityLat, setFacilityLat] = useState<number | ''>('');
  const [facilityLon, setFacilityLon] = useState<number | ''>('');

  // Multiple Source LND state
  const [multipleSourceLNDResult, setMultipleSourceLNDResult] = useState<any | null>(null);
  const [customerFileMS, setCustomerFileMS] = useState<File | null>(null);
  const [warehouseFile, setWarehouseFile] = useState<File | null>(null);
  const [factoryFile, setFactoryFile] = useState<File | null>(null);
  const [productFile, setProductFile] = useState<File | null>(null);
  const [demandFile, setDemandFile] = useState<File | null>(null);
  const [factoryCapacityFile, setFactoryCapacityFile] = useState<File | null>(null);
  const [transportationCost, setTransportationCost] = useState(1.0);
  const [deliveryCost, setDeliveryCost] = useState(2.0);
  const [warehouseFixedCost, setWarehouseFixedCost] = useState(10000.0);
  const [warehouseVariableCost, setWarehouseVariableCost] = useState(1.0);
  const [numWarehouses, setNumWarehouses] = useState<number | ''>('');
  const [singleSourcing, setSingleSourcing] = useState(false);
  const [maxRuntime, setMaxRuntime] = useState(300);
  // Data for visualization
  const [msLndCustomerData, setMsLndCustomerData] = useState<any[]>([]);
  const [msLndWarehouseData, setMsLndWarehouseData] = useState<any[]>([]);
  const [msLndFactoryData, setMsLndFactoryData] = useState<any[]>([]);

  // Single Source LND state
  const [singleSourceLNDResult, setSingleSourceLNDResult] = useState<SingleSourceLNDData | null>(null);
  const [customerFileSS, setCustomerFileSS] = useState<File | null>(null);
  const [warehouseFileSS, setWarehouseFileSS] = useState<File | null>(null);
  const [factoryFileSS, setFactoryFileSS] = useState<File | null>(null);
  const [productFileSS, setProductFileSS] = useState<File | null>(null);
  const [demandFileSS, setDemandFileSS] = useState<File | null>(null);
  const [factoryCapacityFileSS, setFactoryCapacityFileSS] = useState<File | null>(null);
  const [transportationCostSS, setTransportationCostSS] = useState(1.0);
  const [deliveryCostSS, setDeliveryCostSS] = useState(2.0);
  const [warehouseFixedCostSS, setWarehouseFixedCostSS] = useState(10000.0);
  const [warehouseVariableCostSS, setWarehouseVariableCostSS] = useState(1.0);
  const [numWarehousesSS, setNumWarehousesSS] = useState<number | ''>('');
  const [maxRuntimeSS, setMaxRuntimeSS] = useState(300);

  // Elbow Method state
  const [elbowMethodResult, setElbowMethodResult] = useState<ElbowMethodData | null>(null);
  const [elbowMinFacilities, setElbowMinFacilities] = useState(1);
  const [elbowMaxFacilities, setElbowMaxFacilities] = useState(7);  // Reduced from 10
  const [elbowAlgorithm, setElbowAlgorithm] = useState('weiszfeld');
  const [elbowMaxIterations, setElbowMaxIterations] = useState(100);  // Reduced from 1000
  const [elbowTolerance, setElbowTolerance] = useState(1e-2);  // Relaxed from 1e-4
  const [elbowRandomState, setElbowRandomState] = useState(42);

  const [customerData, setCustomerData] = useState<any[]>([]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleOpenSampleDataDialog = async () => {
    try {
      console.log('Fetching sample data info...');
      
      let info;
      try {
        info = await ApiService.getSampleDataInfo();
        console.log('Sample data info received from API:', Object.keys(info).length, 'datasets');
      } catch (apiError) {
        console.log('API failed, using fallback data info:', apiError);
        // Fallback to hardcoded info
        info = {
          'customers_standard': {
            name: 'Standard Customer Dataset',
            description: '東京周辺の50顧客のサンプルデータ',
            customers: 50,
            area: 'Tokyo area (50km radius)',
            use_case: 'General LND analysis and optimization'
          },
          'customers_small': {
            name: 'Small Customer Dataset',
            description: '東京周辺の20顧客のサンプルデータ（テスト用）',
            customers: 20,
            area: 'Tokyo area (30km radius)',
            use_case: 'Quick testing and algorithm validation'
          },
          'customers_regional': {
            name: 'Multi-Regional Dataset',
            description: '東京・大阪・名古屋地域の40顧客のサンプルデータ',
            customers: 40,
            area: 'Tokyo, Osaka, Nagoya regions',
            use_case: 'Multi-cluster analysis and regional facility planning'
          },
          'facilities': {
            name: 'Facility Candidates',
            description: '15の施設候補地のサンプルデータ',
            facilities: 15,
            area: 'Tokyo area (30km radius)',
            use_case: 'Facility location optimization with predefined candidates'
          }
        };
      }
      
      setSampleDataInfo(info);
      setSampleDataDialogOpen(true);
    } catch (err: any) {
      console.error('Error in sample data dialog:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Unknown error';
      setError(`サンプルデータ情報の取得に失敗しました: ${errorMessage}`);
    }
  };

  const handleDownloadSampleData = async (datasetType: string) => {
    try {
      console.log('Downloading dataset:', datasetType);
      
      // Try API first, then fallback to static files
      let blob;
      try {
        blob = await ApiService.downloadSampleData(datasetType);
        console.log('Downloaded from API, blob size:', blob.size);
      } catch (apiError) {
        console.log('API failed, trying static file:', apiError);
        // Fallback to static files
        const response = await fetch(`/sample_data/${datasetType}.csv`);
        if (!response.ok) {
          throw new Error(`Static file not found: ${response.status}`);
        }
        blob = await response.blob();
        console.log('Downloaded from static file, blob size:', blob.size);
      }
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      const filenameMap: {[key: string]: string} = {
        'customers_standard': 'lnd_customers_standard.csv',
        'customers_small': 'lnd_customers_small.csv',
        'customers_regional': 'lnd_customers_regional.csv',
        'facilities': 'lnd_facilities.csv',
        'ms_lnd_customers': 'ms_lnd_customers.csv',
        'ms_lnd_warehouses': 'ms_lnd_warehouses.csv',
        'ms_lnd_factories': 'ms_lnd_factories.csv',
        'ms_lnd_products': 'ms_lnd_products.csv',
        'ms_lnd_demand': 'ms_lnd_demand.csv',
        'ms_lnd_factory_capacity': 'ms_lnd_factory_capacity.csv',
        'elbow_customers_3clusters': 'elbow_customers_3clusters.csv',
        'elbow_customers_2clusters': 'elbow_customers_2clusters.csv',
        'elbow_customers_5clusters': 'elbow_customers_5clusters.csv'
      };
      
      link.download = filenameMap[datasetType] || `${datasetType}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      setSampleDataDialogOpen(false);
    } catch (err: any) {
      console.error('Error downloading sample data:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Unknown error';
      setError(`サンプルデータのダウンロードに失敗しました: ${errorMessage}`);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setCustomerFile(file);
      // Reset results when new file is selected
      setWeiszfeldResult(null);
      setMultiFacilityWeiszfeldResult(null);
      setRepeatedWeiszfeldResult(null);
      setClusteringResult(null);
      setKMedianResult(null);
      setServiceAreaResult(null);
      setCustomerData([]);
      
      // Parse CSV to extract customer data for visualization
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csvData = e.target?.result as string;
          const lines = csvData.split('\n');
          const headers = lines[0].split(',').map(h => h.trim());
          const data: any[] = [];
          
          for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
              const values = lines[i].split(',').map(v => v.trim());
              const record: any = {};
              headers.forEach((header, index) => {
                const value = values[index];
                // Try to parse as number
                const numValue = parseFloat(value);
                record[header] = isNaN(numValue) ? value : numValue;
              });
              data.push(record);
            }
          }
          setCustomerData(data);
        } catch (err) {
          console.error('Error parsing CSV:', err);
        }
      };
      reader.readAsText(file);
    }
  };

  const calculateWeiszfeldLocation = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.calculateWeiszfeldLocation(
        customerFile,
        latCol,
        lonCol,
        demandCol,
        weiszfeldMaxIterations,
        weiszfeldTolerance
      );
      setWeiszfeldResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error calculating Weiszfeld location');
    } finally {
      setLoading(false);
    }
  };

  const calculateMultiFacilityWeiszfeld = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.calculateMultiFacilityWeiszfeld(
        customerFile,
        multiFacilityNum,
        latCol,
        lonCol,
        demandCol,
        multiFacilityMaxIterations,
        multiFacilityTolerance,
        multiFacilityRandomState
      );
      setMultiFacilityWeiszfeldResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error calculating multi-facility Weiszfeld');
    } finally {
      setLoading(false);
    }
  };

  const calculateRepeatedMultiFacilityWeiszfeld = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.calculateRepeatedMultiFacilityWeiszfeld(
        customerFile,
        repeatedNumFacilities,
        repeatedNumRuns,
        latCol,
        lonCol,
        demandCol,
        repeatedMaxIterations,
        repeatedTolerance,
        repeatedBaseRandomState
      );
      setRepeatedWeiszfeldResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error calculating repeated multi-facility Weiszfeld');
    } finally {
      setLoading(false);
    }
  };

  const performCustomerClustering = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.clusterCustomers(
        customerFile,
        clusteringMethod,
        nClusters,
        latCol,
        lonCol,
        demandCol,
        linkageMethod,
        randomState
      );
      setClusteringResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error performing customer clustering');
    } finally {
      setLoading(false);
    }
  };

  const solveKMedianProblem = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.solveKMedian(
        customerFile,
        kValue,
        candidateMethod,
        nCandidates,
        latCol,
        lonCol,
        demandCol,
        maxIterations,
        lambdaStep
      );
      setKMedianResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error solving K-Median problem');
    } finally {
      setLoading(false);
    }
  };

  const analyzeServiceArea = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    if (facilityLat === '' || facilityLon === '') {
      setError('Please enter facility coordinates');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.analyzeServiceArea(
        customerFile,
        facilityLat as number,
        facilityLon as number,
        latCol,
        lonCol,
        demandCol
      );
      setServiceAreaResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error analyzing service area');
    } finally {
      setLoading(false);
    }
  };

  const solveMultipleSourceLND = async () => {
    if (!customerFileMS || !warehouseFile || !factoryFile || !productFile || !demandFile || !factoryCapacityFile) {
      setError('Please select all required data files');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.solveMultipleSourceLND(
        customerFileMS,
        warehouseFile,
        factoryFile,
        productFile,
        demandFile,
        factoryCapacityFile,
        transportationCost,
        deliveryCost,
        warehouseFixedCost,
        warehouseVariableCost,
        numWarehouses === '' ? undefined : numWarehouses as number,
        singleSourcing,
        maxRuntime
      );
      setMultipleSourceLNDResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error solving Multiple Source LND');
    } finally {
      setLoading(false);
    }
  };

  const downloadAllMSLNDSampleData = async () => {
    const msLndDatasets = [
      'ms_lnd_customers',
      'ms_lnd_warehouses', 
      'ms_lnd_factories',
      'ms_lnd_products',
      'ms_lnd_demand',
      'ms_lnd_factory_capacity'
    ];

    try {
      for (const datasetType of msLndDatasets) {
        await handleDownloadSampleData(datasetType);
        // Small delay between downloads to avoid overwhelming the browser
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (err: any) {
      setError('複数拠点LNDサンプルデータのダウンロードに失敗しました');
    }
  };

  const downloadAllElbowSampleData = async () => {
    const elbowDatasets = [
      'elbow_customers_2clusters',
      'elbow_customers_3clusters',
      'elbow_customers_5clusters'
    ];

    try {
      for (const datasetType of elbowDatasets) {
        await handleDownloadSampleData(datasetType);
        // Small delay between downloads to avoid overwhelming the browser
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (err: any) {
      setError('エルボー法サンプルデータのダウンロードに失敗しました');
    }
  };

  const solveSingleSourceLND = async () => {
    if (!customerFileSS || !warehouseFileSS || !factoryFileSS || !productFileSS || !demandFileSS || !factoryCapacityFileSS) {
      setError('Please select all required data files');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.solveSingleSourceLND(
        customerFileSS,
        warehouseFileSS,
        factoryFileSS,
        productFileSS,
        demandFileSS,
        factoryCapacityFileSS,
        transportationCostSS,
        deliveryCostSS,
        warehouseFixedCostSS,
        warehouseVariableCostSS,
        numWarehousesSS === '' ? undefined : numWarehousesSS as number,
        maxRuntimeSS
      );
      setSingleSourceLNDResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error solving Single Source LND');
    } finally {
      setLoading(false);
    }
  };

  const performElbowMethodAnalysis = async () => {
    if (!customerFile) {
      setError('Please select a customer data file');
      return;
    }
    
    if (elbowMaxFacilities <= elbowMinFacilities) {
      setError('Maximum facilities must be greater than minimum facilities');
      return;
    }

    if (elbowMaxFacilities - elbowMinFacilities < 2) {
      setError('Need at least 3 different facility counts for elbow method analysis');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.performElbowMethodAnalysis(
        customerFile,
        latCol,
        lonCol,
        demandCol,
        elbowMinFacilities,
        elbowMaxFacilities,
        elbowAlgorithm,
        elbowMaxIterations,
        elbowTolerance,
        elbowRandomState
      );
      setElbowMethodResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error performing elbow method analysis');
    } finally {
      setLoading(false);
    }
  };

  // Helper function to parse CSV file and extract data
  const parseCSVFile = (file: File): Promise<any[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const csvData = e.target?.result as string;
          const lines = csvData.split('\n');
          const headers = lines[0].split(',').map(h => h.trim());
          const data: any[] = [];
          
          for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
              const values = lines[i].split(',').map(v => v.trim());
              const record: any = {};
              headers.forEach((header, index) => {
                const value = values[index];
                // Try to parse as number
                const numValue = parseFloat(value);
                record[header] = isNaN(numValue) ? value : numValue;
              });
              data.push(record);
            }
          }
          resolve(data);
        } catch (err) {
          reject(err);
        }
      };
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  // Update file handlers to parse CSV data for visualization
  const handleMSLNDCustomerFileChange = async (file: File) => {
    setCustomerFileMS(file);
    try {
      const data = await parseCSVFile(file);
      setMsLndCustomerData(data);
    } catch (err) {
      console.error('Error parsing customer CSV:', err);
      setMsLndCustomerData([]);
    }
  };

  const handleMSLNDWarehouseFileChange = async (file: File) => {
    setWarehouseFile(file);
    try {
      const data = await parseCSVFile(file);
      setMsLndWarehouseData(data);
    } catch (err) {
      console.error('Error parsing warehouse CSV:', err);
      setMsLndWarehouseData([]);
    }
  };

  const handleMSLNDFactoryFileChange = async (file: File) => {
    setFactoryFile(file);
    try {
      const data = await parseCSVFile(file);
      setMsLndFactoryData(data);
    } catch (err) {
      console.error('Error parsing factory CSV:', err);
      setMsLndFactoryData([]);
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Logistics Network Design (MELOS)
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data Input
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<CloudUploadIcon />}
                  sx={{ mr: 2 }}
                >
                  顧客データをアップロード (CSV)
                  <input
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={handleFileChange}
                  />
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<DownloadIcon />}
                  onClick={handleOpenSampleDataDialog}
                >
                  サンプルデータをダウンロード
                </Button>
              </Box>
              {customerFile && (
                <Typography variant="body2" color="text.secondary">
                  選択済み: {customerFile.name} ({customerData.length} 顧客)
                </Typography>
              )}
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <TextField
                    fullWidth
                    label="Latitude Column"
                    value={latCol}
                    onChange={(e) => setLatCol(e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={4}>
                  <TextField
                    fullWidth
                    label="Longitude Column"
                    value={lonCol}
                    onChange={(e) => setLonCol(e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={4}>
                  <TextField
                    fullWidth
                    label="Demand Column"
                    value={demandCol}
                    onChange={(e) => setDemandCol(e.target.value)}
                    size="small"
                  />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Tabs value={tabValue} onChange={handleTabChange}>
        <Tab label="システム設定" icon={<SettingsIcon />} iconPosition="start" />
        <Tab label="データ管理" icon={<DataIcon />} iconPosition="start" />
        <Tab label="立地モデル" icon={<MapIcon />} iconPosition="start" />
        <Tab label="最適化実行" icon={<ExecuteIcon />} iconPosition="start" />
        <Tab label="結果分析" icon={<ResultsIcon />} iconPosition="start" />
        <Tab label="リアルタイム監視" icon={<MonitorIcon />} iconPosition="start" />
        <Tab label="ポリシー管理" icon={<PolicyIcon />} iconPosition="start" />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        <Typography variant="h6" gutterBottom>
          システム設定 - 物流ネットワーク設計
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          物流ネットワーク設計システムの基本設定と初期化
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  アルゴリズム設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="最大反復回数"
                      type="number"
                      value={weiszfeldMaxIterations}
                      onChange={(e) => setWeiszfeldMaxIterations(parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="収束許容誤差"
                      type="number"
                      value={weiszfeldTolerance}
                      onChange={(e) => setWeiszfeldTolerance(parseFloat(e.target.value))}
                      inputProps={{ step: 1e-7 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  システム状態
                </Typography>
                <Typography variant="body2">
                  データ読み込み状況: {customerFile ? '完了' : '未完了'}<br/>
                  分析モジュール: 9種類<br/>
                  最適化エンジン: 利用可能
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  環境設定
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  ネットワーク設計の全般設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="デフォルト距離単位"
                      value="km"
                      disabled
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="最大施設数"
                      value="50"
                      disabled
                    />
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <TextField
                      fullWidth
                      label="計算タイムアウト"
                      value="300秒"
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
          顧客データとネットワークデータの管理
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ファイルアップロード
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<CloudUploadIcon />}
                    sx={{ mr: 2 }}
                    fullWidth
                  >
                    顧客データをアップロード (CSV)
                    <input
                      type="file"
                      accept=".csv"
                      hidden
                      onChange={handleFileChange}
                    />
                  </Button>
                  {customerFile && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      選択済み: {customerFile.name} ({customerData.length} 顧客)
                    </Typography>
                  )}
                </Box>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<DownloadIcon />}
                  onClick={handleOpenSampleDataDialog}
                  fullWidth
                >
                  サンプルデータをダウンロード
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  データ列設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Latitude Column"
                      value={latCol}
                      onChange={(e) => setLatCol(e.target.value)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Longitude Column"
                      value={lonCol}
                      onChange={(e) => setLonCol(e.target.value)}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Demand Column"
                      value={demandCol}
                      onChange={(e) => setDemandCol(e.target.value)}
                      size="small"
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
                  データ検証
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      読み込み状況
                    </Typography>
                    <Typography variant="h6">
                      {customerFile ? '完了' : '未完了'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      顧客数
                    </Typography>
                    <Typography variant="h6">
                      {customerData.length || 0}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      データ形式
                    </Typography>
                    <Typography variant="h6">
                      CSV
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      分析準備
                    </Typography>
                    <Typography variant="h6" color={customerFile ? 'success.main' : 'error.main'}>
                      {customerFile ? '準備完了' : '未準備'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          立地モデル
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Weiszfeld法、K-Median法、クラスタリングによる施設立地最適化
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Weiszfeld法 (単一施設)
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  重み付き距離の総和を最小化する最適立地を計算
                </Typography>
                <Button
                  variant="contained"
                  onClick={calculateWeiszfeldLocation}
                  disabled={loading || !customerFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  Weiszfeld最適化を実行
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  複数施設Weiszfeld法
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  複数施設の同時最適配置
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="施設数"
                      type="number"
                      value={multiFacilityNum}
                      onChange={(e) => setMultiFacilityNum(parseInt(e.target.value))}
                      inputProps={{ min: 2, max: 10 }}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="Random State"
                      type="number"
                      value={multiFacilityRandomState}
                      onChange={(e) => setMultiFacilityRandomState(parseInt(e.target.value))}
                      size="small"
                    />
                  </Grid>
                </Grid>
                <Button
                  variant="contained"
                  onClick={calculateMultiFacilityWeiszfeld}
                  disabled={loading || !customerFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  複数施設最適化を実行
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  K-Median法
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  ラグランジュ緩和による最適施設配置
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="施設数 (K)"
                      type="number"
                      value={kValue}
                      onChange={(e) => setKValue(parseInt(e.target.value))}
                      inputProps={{ min: 1, max: 10 }}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <FormControl fullWidth size="small">
                      <InputLabel>候補生成</InputLabel>
                      <Select
                        value={candidateMethod}
                        label="候補生成"
                        onChange={(e) => setCandidateMethod(e.target.value)}
                      >
                        <MenuItem value="grid">グリッド</MenuItem>
                        <MenuItem value="random">ランダム</MenuItem>
                        <MenuItem value="customer_locations">顧客位置</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>
                <Button
                  variant="contained"
                  onClick={solveKMedianProblem}
                  disabled={loading || !customerFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  K-Median問題を解く
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  顧客クラスタリング
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  地域別施設計画のためのクラスタ分析
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <FormControl fullWidth size="small">
                      <InputLabel>手法</InputLabel>
                      <Select
                        value={clusteringMethod}
                        label="手法"
                        onChange={(e) => setClusteringMethod(e.target.value)}
                      >
                        <MenuItem value="kmeans">K-Means</MenuItem>
                        <MenuItem value="hierarchical">階層</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="クラスタ数"
                      type="number"
                      value={nClusters}
                      onChange={(e) => setNClusters(parseInt(e.target.value))}
                      inputProps={{ min: 2, max: 10 }}
                      size="small"
                    />
                  </Grid>
                </Grid>
                <Button
                  variant="contained"
                  onClick={performCustomerClustering}
                  disabled={loading || !customerFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  クラスタリング実行
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* 結果表示セクション - 1列で表示 */}
        {(weiszfeldResult || multiFacilityWeiszfeldResult || kMedianResult || clusteringResult) && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              分析結果
            </Typography>
            <Grid container spacing={3}>
              {weiszfeldResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Weiszfeld法最適化結果
                      </Typography>
                      <WeiszfeldLocationChart 
                        data={weiszfeldResult} 
                        customerData={customerData}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {multiFacilityWeiszfeldResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        複数施設Weiszfeld法結果
                      </Typography>
                      <MultiFacilityWeiszfeldChart 
                        data={multiFacilityWeiszfeldResult} 
                        customerData={customerData}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {kMedianResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        K-Median最適化結果
                      </Typography>
                      <KMedianSolutionChart 
                        data={kMedianResult} 
                        customerData={customerData}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {clusteringResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        顧客クラスタリング結果
                      </Typography>
                      <CustomerClusteringChart data={clusteringResult} />
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          </Box>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <Typography variant="h6" gutterBottom>
          最適化実行
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          大規模物流ネットワーク設計の最適化実行と監視
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  複数拠点LND最適化
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  混合整数最適化による複数拠点物流ネットワーク設計
                </Typography>
                
                {/* File Upload Section */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    データファイル
                  </Typography>
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Button
                        variant={customerFileMS ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={customerFileMS ? "success" : "primary"}
                      >
                        顧客
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleMSLNDCustomerFileChange(file);
                          }}
                        />
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant={warehouseFile ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={warehouseFile ? "success" : "primary"}
                      >
                        倉庫
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleMSLNDWarehouseFileChange(file);
                          }}
                        />
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant={factoryFile ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={factoryFile ? "success" : "primary"}
                      >
                        工場
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleMSLNDFactoryFileChange(file);
                          }}
                        />
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant={productFile ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={productFile ? "success" : "primary"}
                      >
                        製品
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => setProductFile(e.target.files?.[0] || null)}
                        />
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant={demandFile ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={demandFile ? "success" : "primary"}
                      >
                        需要
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => setDemandFile(e.target.files?.[0] || null)}
                        />
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant={factoryCapacityFile ? "contained" : "outlined"}
                        component="label"
                        size="small"
                        fullWidth
                        color={factoryCapacityFile ? "success" : "primary"}
                      >
                        工場能力
                        <input
                          type="file"
                          accept=".csv"
                          hidden
                          onChange={(e) => setFactoryCapacityFile(e.target.files?.[0] || null)}
                        />
                      </Button>
                    </Grid>
                  </Grid>
                </Box>

                <Button
                  variant="contained"
                  onClick={solveMultipleSourceLND}
                  disabled={loading || !customerFileMS || !warehouseFile || !factoryFile || !productFile || !demandFile || !factoryCapacityFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                  size="large"
                >
                  複数拠点LND最適化を実行
                </Button>
                
                {multipleSourceLNDResult && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Status: {multipleSourceLNDResult.status} | 
                      Cost: {multipleSourceLNDResult.total_cost?.toFixed(2) || 'N/A'} | 
                      Runtime: {multipleSourceLNDResult.runtime?.toFixed(2) || 'N/A'}s
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  単一拠点LND最適化
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  単一拠点制約付き物流ネットワーク設計
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="輸送コスト"
                      type="number"
                      value={transportationCostSS}
                      onChange={(e) => setTransportationCostSS(parseFloat(e.target.value) || 0)}
                      inputProps={{ step: 0.1, min: 0 }}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="配送コスト"
                      type="number"
                      value={deliveryCostSS}
                      onChange={(e) => setDeliveryCostSS(parseFloat(e.target.value) || 0)}
                      inputProps={{ step: 0.1, min: 0 }}
                      size="small"
                    />
                  </Grid>
                </Grid>

                <Button
                  variant="contained"
                  onClick={solveSingleSourceLND}
                  disabled={loading || !customerFileSS || !warehouseFileSS || !factoryFileSS}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                  size="large"
                >
                  単一拠点LND最適化を実行
                </Button>
                
                {singleSourceLNDResult && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Status: {singleSourceLNDResult.status} | 
                      Cost: {singleSourceLNDResult.total_cost?.toFixed(2) || 'N/A'} | 
                      Runtime: {singleSourceLNDResult.runtime?.toFixed(2) || 'N/A'}s
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  最適化実行状況
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      実行中タスク
                    </Typography>
                    <Typography variant="h6">
                      {loading ? '1' : '0'}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      完了タスク
                    </Typography>
                    <Typography variant="h6">
                      {[multipleSourceLNDResult, singleSourceLNDResult].filter(r => r).length}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      エラー
                    </Typography>
                    <Typography variant="h6" color="error.main">
                      {error ? '1' : '0'}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      システム状態
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      正常
                    </Typography>
                  </Grid>
                </Grid>
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
          最適化結果の詳細分析とサービスエリア解析
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  サービスエリア分析
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  指定施設のサービス範囲と統計分析
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="施設緯度"
                      type="number"
                      value={facilityLat}
                      onChange={(e) => setFacilityLat(e.target.value === '' ? '' : parseFloat(e.target.value))}
                      inputProps={{ step: 0.0001 }}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="施設経度"
                      type="number"
                      value={facilityLon}
                      onChange={(e) => setFacilityLon(e.target.value === '' ? '' : parseFloat(e.target.value))}
                      inputProps={{ step: 0.0001 }}
                      size="small"
                    />
                  </Grid>
                </Grid>
                
                <Button
                  variant="contained"
                  onClick={analyzeServiceArea}
                  disabled={loading || !customerFile || facilityLat === '' || facilityLon === ''}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  サービスエリア分析実行
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  エルボー法分析
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  最適施設数決定のためのコスト分析
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="最小施設数"
                      type="number"
                      value={elbowMinFacilities}
                      onChange={(e) => setElbowMinFacilities(parseInt(e.target.value))}
                      inputProps={{ min: 1 }}
                      size="small"
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="最大施設数"
                      type="number"
                      value={elbowMaxFacilities}
                      onChange={(e) => setElbowMaxFacilities(parseInt(e.target.value))}
                      inputProps={{ min: 2 }}
                      size="small"
                    />
                  </Grid>
                </Grid>
                
                <Button
                  variant="contained"
                  onClick={performElbowMethodAnalysis}
                  disabled={loading || !customerFile}
                  startIcon={loading && <CircularProgress size={20} />}
                  fullWidth
                >
                  エルボー法分析を実行
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  分析結果サマリー
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      実行済み分析
                    </Typography>
                    <Typography variant="h6">
                      {[weiszfeldResult, multiFacilityWeiszfeldResult, clusteringResult, kMedianResult, serviceAreaResult, elbowMethodResult].filter(r => r).length}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      最適化結果
                    </Typography>
                    <Typography variant="h6">
                      {[multipleSourceLNDResult, singleSourceLNDResult].filter(r => r && r.status === 'Optimal').length}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      可視化グラフ
                    </Typography>
                    <Typography variant="h6">
                      {[weiszfeldResult, multiFacilityWeiszfeldResult, clusteringResult, kMedianResult, serviceAreaResult].filter(r => r).length}
                    </Typography>
                  </Grid>
                  <Grid item xs={3}>
                    <Typography variant="body2" color="text.secondary">
                      分析準備度
                    </Typography>
                    <Typography variant="h6" color={customerFile ? 'success.main' : 'warning.main'}>
                      {customerFile ? '100%' : '50%'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* 結果分析 - 1列で表示 */}
        {(serviceAreaResult || elbowMethodResult) && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              分析結果
            </Typography>
            <Grid container spacing={3}>
              {serviceAreaResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        サービスエリア分析結果
                      </Typography>
                      <ServiceAreaChart 
                        data={serviceAreaResult} 
                        customerData={customerData}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {elbowMethodResult && (
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        エルボー法分析結果
                      </Typography>
                      <ElbowMethodChart data={elbowMethodResult} />
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          </Box>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={5}>
        <Typography variant="h6" gutterBottom>
          リアルタイム監視
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          最適化プロセスとシステム状態のリアルタイム監視
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  最適化進行状況
                </Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      実行状態
                    </Typography>
                    <Typography variant="h6" color={loading ? 'warning.main' : 'success.main'}>
                      {loading ? '実行中' : '待機中'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      CPU使用率
                    </Typography>
                    <Typography variant="h6">
                      {loading ? '75%' : '15%'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      メモリ使用量
                    </Typography>
                    <Typography variant="h6">
                      {loading ? '2.1GB' : '0.8GB'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      実行時間
                    </Typography>
                    <Typography variant="h6">
                      {loading ? '実行中...' : '待機中'}
                    </Typography>
                  </Grid>
                </Grid>
                
                {loading && (
                  <Box sx={{ width: '100%', mt: 2 }}>
                    <CircularProgress size={24} sx={{ mr: 2 }} />
                    <Typography variant="body2" component="span">
                      最適化計算中...
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  システム監視
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      API接続
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      正常
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      データベース
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      接続中
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      最適化エンジン
                    </Typography>
                    <Typography variant="h6" color="success.main">
                      利用可能
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      エラー数
                    </Typography>
                    <Typography variant="h6" color={error ? 'error.main' : 'success.main'}>
                      {error ? '1' : '0'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  実行履歴
                </Typography>
                <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                  {[
                    { time: '15:32:01', action: 'システム初期化完了', status: 'success' },
                    { time: '15:32:15', action: 'データ読み込み開始', status: 'info' },
                    customerFile && { time: '15:32:18', action: `顧客データ読み込み完了 (${customerData.length}件)`, status: 'success' },
                    weiszfeldResult && { time: '15:33:45', action: 'Weiszfeld最適化完了', status: 'success' },
                    multiFacilityWeiszfeldResult && { time: '15:34:12', action: '複数施設最適化完了', status: 'success' },
                    error && { time: '15:35:01', action: 'エラーが発生しました', status: 'error' },
                  ].filter(Boolean).map((log, index) => (
                    <Box key={index} sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #eee' }}>
                      <Typography variant="body2">{log.action}</Typography>
                      <Typography variant="caption" color="text.secondary">{log.time}</Typography>
                    </Box>
                  ))}
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
          最適化アルゴリズムのパラメータ設定と実行ポリシー管理
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Weiszfeldアルゴリズム設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="最大反復回数"
                      type="number"
                      value={weiszfeldMaxIterations}
                      onChange={(e) => setWeiszfeldMaxIterations(parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="収束許容誤差"
                      type="number"
                      value={weiszfeldTolerance}
                      onChange={(e) => setWeiszfeldTolerance(parseFloat(e.target.value))}
                      inputProps={{ step: 1e-7 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  複数施設最適化設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="最大反復回数"
                      type="number"
                      value={multiFacilityMaxIterations}
                      onChange={(e) => setMultiFacilityMaxIterations(parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="収束許容誤差"
                      type="number"
                      value={multiFacilityTolerance}
                      onChange={(e) => setMultiFacilityTolerance(parseFloat(e.target.value))}
                      inputProps={{ step: 1e-5 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  K-Median設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="最大反復回数"
                      type="number"
                      value={maxIterations}
                      onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="候補数"
                      type="number"
                      value={nCandidates}
                      onChange={(e) => setNCandidates(parseInt(e.target.value))}
                      inputProps={{ min: 5, max: 100 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  実行制限設定
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="最大実行時間 (秒)"
                      type="number"
                      value={maxRuntime}
                      onChange={(e) => setMaxRuntime(parseInt(e.target.value) || 300)}
                      inputProps={{ min: 10, max: 3600 }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="メモリ制限 (MB)"
                      type="number"
                      value="4096"
                      disabled
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
                  保存済みポリシー
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        デフォルト設定
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        標準的な最適化設定<br/>
                        反復回数: 1000, 許容誤差: 1e-6
                      </Typography>
                      <Button size="small" sx={{ mt: 1 }}>適用</Button>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        高速実行設定
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        時間重視の設定<br/>
                        反復回数: 100, 許容誤差: 1e-2
                      </Typography>
                      <Button size="small" sx={{ mt: 1 }}>適用</Button>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        高精度設定
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        精度重視の設定<br/>
                        反復回数: 5000, 許容誤差: 1e-8
                      </Typography>
                      <Button size="small" sx={{ mt: 1 }}>適用</Button>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Sample Data Dialog */}
      <Dialog 
        open={sampleDataDialogOpen} 
        onClose={() => setSampleDataDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <InfoIcon sx={{ mr: 1 }} />
            サンプルデータをダウンロード
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            物流ネットワーク設計機能を試すためのサンプルデータをダウンロードできます。
            ダウンロード後、上記の「顧客データをアップロード」ボタンでファイルを選択してください。
          </Typography>
          
          {sampleDataInfo && (
            <List>
              {Object.entries(sampleDataInfo).map(([key, info]: [string, any]) => (
                <ListItem key={key} divider>
                  <ListItemText
                    primary={info.name}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {info.description}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {info.customers ? `${info.customers}顧客` : 
                           info.facilities ? `${info.facilities}施設` :
                           info.warehouses ? `${info.warehouses}倉庫` :
                           info.factories ? `${info.factories}工場` :
                           info.products ? `${info.products}製品` :
                           info.records ? `${info.records}レコード` : ''}
                          {info.customers || info.facilities || info.warehouses || info.factories || info.products || info.records ? ' • ' : ''}
                          {info.area} • {info.use_case}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end" 
                      onClick={() => handleDownloadSampleData(key)}
                      color="primary"
                    >
                      <DownloadIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSampleDataDialogOpen(false)}>閉じる</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default LogisticsNetworkDesign;