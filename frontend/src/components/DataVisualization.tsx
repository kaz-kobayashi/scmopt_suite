import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Card, CardContent, Typography, Grid } from '@mui/material';

interface DataVisualizationProps {
  title: string;
  data: any;
  layout?: any;
  config?: any;
}

const DataVisualization: React.FC<DataVisualizationProps> = ({
  title,
  data,
  layout = {},
  config = { responsive: true, displayModeBar: false },
}) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6" component="h3" gutterBottom>
          {title}
        </Typography>
        <Box>
          <Plot
            data={data}
            layout={{
              autosize: true,
              ...layout,
            }}
            config={config}
            style={{ width: '100%', height: '400px' }}
            useResizeHandler={true}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export interface ABCChartData {
  aggregated_data: Array<{
    [key: string]: string | number;
    abc: string;
    rank: number;
  }>;
  categories: {
    [key: number]: string[];
  };
}

export const ABCAnalysisChart: React.FC<{ data: ABCChartData; valueColumn: string }> = ({
  data,
  valueColumn,
}) => {
  const aggregatedData = data.aggregated_data;
  const categories = ['A', 'B', 'C'];
  
  const chartData = categories.map((category, index) => {
    const categoryItems = aggregatedData.filter(item => item.abc === category);
    const values = categoryItems.map(item => item[valueColumn] as number);
    const labels = categoryItems.map(item => item.prod || item.cust || `Item ${index + 1}`);
    
    return {
      x: labels,
      y: values,
      name: `Category ${category}`,
      type: 'bar' as const,
      marker: {
        color: category === 'A' ? '#2196f3' : category === 'B' ? '#ff9800' : '#4caf50',
      },
    };
  });

  const layout = {
    title: 'ABC Analysis Distribution',
    xaxis: { title: 'Items' },
    yaxis: { title: valueColumn.charAt(0).toUpperCase() + valueColumn.slice(1) },
    barmode: 'group',
    height: 400,
  };

  return <DataVisualization title="ABC Analysis" data={chartData} layout={layout} />;
};

export interface ParetoChartData {
  aggregated_data: Array<{
    [key: string]: string | number | undefined;
    cum_pct?: number;
    abc: string;
  }>;
}

export const ParetoChart: React.FC<{ data: ParetoChartData; valueColumn: string }> = ({
  data,
  valueColumn,
}) => {
  const sortedData = data.aggregated_data.sort((a, b) => (b[valueColumn] as number) - (a[valueColumn] as number));
  
  const barTrace = {
    x: sortedData.map(item => item.prod || item.cust || item.name),
    y: sortedData.map(item => item[valueColumn] as number),
    name: valueColumn.charAt(0).toUpperCase() + valueColumn.slice(1),
    type: 'bar' as const,
    yaxis: 'y',
    marker: {
      color: sortedData.map(item => {
        const abc = item.abc as string;
        return abc === 'A' ? '#2196f3' : abc === 'B' ? '#ff9800' : '#4caf50';
      }),
    },
  };

  const lineTrace = {
    x: sortedData.map(item => item.prod || item.cust || item.name),
    y: sortedData.map(item => (item.cum_pct || 0) * 100),
    name: 'Cumulative %',
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    yaxis: 'y2',
    line: { color: '#e91e63' },
  };

  const layout = {
    title: 'Pareto Chart',
    xaxis: { title: 'Items' },
    yaxis: {
      title: valueColumn.charAt(0).toUpperCase() + valueColumn.slice(1),
      side: 'left' as const,
    },
    yaxis2: {
      title: 'Cumulative Percentage (%)',
      side: 'right' as const,
      overlaying: 'y',
      range: [0, 100],
    },
    height: 400,
  };

  return <DataVisualization title="Pareto Analysis" data={[barTrace, lineTrace]} layout={layout} />;
};

export interface InventorySimulationData {
  simulation_results: {
    average_cost_per_period: number;
    cost_standard_deviation: number;
    cost_range: {
      min: number;
      max: number;
    };
    confidence_interval_95: {
      lower: number;
      upper: number;
    };
    average_inventory_level: number;
  };
}

export const InventorySimulationChart: React.FC<{ data: InventorySimulationData }> = ({ data }) => {
  const { simulation_results } = data;
  
  const costData = {
    x: ['Average Cost', 'Min Cost', 'Max Cost', 'CI Lower', 'CI Upper'],
    y: [
      simulation_results.average_cost_per_period,
      simulation_results.cost_range.min,
      simulation_results.cost_range.max,
      simulation_results.confidence_interval_95.lower,
      simulation_results.confidence_interval_95.upper,
    ],
    type: 'bar' as const,
    name: 'Cost Analysis',
    marker: { color: '#3f51b5' },
  };

  const layout = {
    title: 'Inventory Simulation Results',
    xaxis: { title: 'Cost Metrics' },
    yaxis: { title: 'Cost per Period' },
    height: 400,
  };

  return <DataVisualization title="Inventory Simulation" data={[costData]} layout={layout} />;
};

export interface CO2Data {
  emissions_calculation: {
    fuel_consumption_L_per_ton_km: number;
    co2_emissions_g_per_ton_km: number;
    fuel_type: string;
    vehicle_capacity_tons: number;
    loading_rate: number;
  };
  annual_estimates: {
    estimated_annual_fuel_consumption_L: number;
    estimated_annual_co2_emissions_kg: number;
  };
  optimization_suggestions: {
    efficiency_score: number;
  };
}

export const CO2EmissionsChart: React.FC<{ data: CO2Data }> = ({ data }) => {
  const { emissions_calculation, annual_estimates, optimization_suggestions } = data;
  
  const chartData = [
    {
      labels: ['Fuel Consumption (L/ton-km)', 'CO2 Emissions (g/ton-km)', 'Efficiency Score'],
      values: [
        emissions_calculation.fuel_consumption_L_per_ton_km,
        emissions_calculation.co2_emissions_g_per_ton_km / 1000, // Convert to kg for better scale
        optimization_suggestions.efficiency_score,
      ],
      type: 'bar' as const,
      name: 'Emissions Metrics',
      marker: { 
        color: ['#ff6b6b', '#ffa726', '#66bb6a'],
      },
    }
  ];

  const layout = {
    title: `${emissions_calculation.fuel_type} Vehicle Emissions Analysis`,
    xaxis: { title: 'Metrics' },
    yaxis: { title: 'Values' },
    height: 400,
  };

  return <DataVisualization title="CO2 Emissions Analysis" data={chartData} layout={layout} />;
};

export interface MeanCVData {
  scatter_data: Array<{
    x: number;
    y: number;
    prod: string;
    size: number;
    color?: number;
  }>;
  x_label: string;
  y_label: string;
  title: string;
  show_names: boolean;
  has_price_info: boolean;
}

export const MeanCVChart: React.FC<{ data: MeanCVData }> = ({ data }) => {
  const { scatter_data, x_label, y_label, title, show_names, has_price_info } = data;
  
  const scatterTrace = {
    x: scatter_data.map(item => item.x),
    y: scatter_data.map(item => item.y),
    text: show_names ? scatter_data.map(item => item.prod) : undefined,
    textposition: 'middle right' as const,
    mode: show_names ? ('markers+text' as const) : ('markers' as const),
    type: 'scatter' as const,
    marker: {
      size: scatter_data.map(item => Math.max(item.size / 100, 5)), // Scale size appropriately
      color: has_price_info ? scatter_data.map(item => item.color || 1) : undefined,
      colorscale: has_price_info ? 'Viridis' : undefined,
      showscale: has_price_info,
      colorbar: has_price_info ? {
        title: 'Product Value'
      } : undefined,
    },
    name: 'Products',
    hovertemplate: '<b>%{text}</b><br>' +
                  `${x_label}: %{x}<br>` +
                  `${y_label}: %{y}<br>` +
                  '<extra></extra>',
    hovertext: scatter_data.map(item => item.prod),
  };

  const layout = {
    title: title,
    xaxis: { 
      title: x_label,
      type: 'log' as const,
    },
    yaxis: { 
      title: y_label,
    },
    height: 400,
  };

  return <DataVisualization title={title} data={[scatterTrace]} layout={layout} />;
};

export interface RankAnalysisData {
  rank_analysis?: { [key: string]: number };
  rank_analysis_periods?: { [key: string]: (number | null)[] };
  analysis_summary: {
    total_items: number;
    analyzed_column: string;
    value_column: string;
    aggregation_period?: string;
  };
}

export const RankAnalysisChart: React.FC<{ data: RankAnalysisData; topItems?: number }> = ({ 
  data, 
  topItems = 10 
}) => {
  const { rank_analysis, rank_analysis_periods, analysis_summary } = data;
  
  // Handle simple rank analysis (single period)
  if (rank_analysis && !rank_analysis_periods) {
    const sortedItems = Object.entries(rank_analysis)
      .sort(([,a], [,b]) => a - b)
      .slice(0, topItems);
    
    const barTrace = {
      x: sortedItems.map(([item,]) => item),
      y: sortedItems.map(([, rank]) => rank),
      type: 'bar' as const,
      name: 'Rank',
      marker: { color: '#3f51b5' },
    };

    const layout = {
      title: `Rank Analysis - Top ${topItems} Items`,
      xaxis: { title: analysis_summary.analyzed_column },
      yaxis: { 
        title: 'Rank',
        autorange: 'reversed' as const, // Lower rank (better) at top
      },
      height: 400,
    };

    return <DataVisualization title="Rank Analysis" data={[barTrace]} layout={layout} />;
  }
  
  // Handle periodic rank analysis
  if (rank_analysis_periods) {
    // Get top items by latest rank
    const latestRanks = Object.entries(rank_analysis_periods)
      .map(([item, ranks]) => {
        const lastRank = ranks[ranks.length - 1];
        return [item, lastRank] as [string, number | null];
      })
      .filter(([, rank]) => rank !== null)
      .sort(([, a], [, b]) => (a as number) - (b as number))
      .slice(0, topItems);
    
    const traces = latestRanks.map(([item, ]) => {
      const ranks = rank_analysis_periods[item];
      return {
        x: ranks.map((_, index) => `Period ${index + 1}`),
        y: ranks,
        name: item,
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        line: { width: 2 },
        marker: { size: 6 },
      };
    });

    const layout = {
      title: `Rank Evolution Over Time - Top ${topItems} Items`,
      xaxis: { title: 'Time Period' },
      yaxis: { 
        title: 'Rank',
        autorange: 'reversed' as const, // Lower rank (better) at top
      },
      height: 400,
    };

    return <DataVisualization title="Rank Analysis Over Time" data={traces} layout={layout} />;
  }

  return (
    <Card elevation={2}>
      <CardContent>
        <Typography variant="h6">Rank Analysis</Typography>
        <Typography variant="body2" color="text.secondary">
          No rank data available
        </Typography>
      </CardContent>
    </Card>
  );
};

export interface ABCTreemapData {
  data: Array<{
    label: string;
    value: number;
    abc_category: string;
    color: string;
    parent: string;
    prod: string;
    cust: string;
    percentage: number;
  }>;
  title: string;
  total_value: number;
  abc_column: string;
  value_column: string;
  num_items: number;
  color_mapping: { [key: string]: string };
}

export const ABCTreemapChart: React.FC<{ data: ABCTreemapData }> = ({ data }) => {
  const { data: treeData, title, color_mapping } = data;
  
  // Group data by ABC category for better visualization
  const groupedData = treeData.reduce((acc, item) => {
    const category = item.abc_category;
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(item);
    return acc;
  }, {} as { [key: string]: typeof treeData });

  // Create treemap trace
  const treemapTrace = {
    type: 'treemap' as const,
    labels: treeData.map(item => item.label),
    values: treeData.map(item => item.value),
    parents: treeData.map(item => item.parent),
    text: treeData.map(item => 
      `${item.label}<br>ABC: ${item.abc_category}<br>Value: ${item.value.toFixed(0)}<br>Percentage: ${item.percentage.toFixed(1)}%`
    ),
    textinfo: 'label+value',
    hovertemplate: '<b>%{label}</b><br>' +
                  'ABC Category: %{customdata.abc_category}<br>' +
                  'Value: %{value}<br>' +
                  'Percentage: %{customdata.percentage:.1f}%<br>' +
                  '<extra></extra>',
    customdata: treeData.map(item => ({
      abc_category: item.abc_category,
      percentage: item.percentage
    })),
    marker: {
      colors: treeData.map(item => item.color),
      colorbar: {
        title: 'ABC Category',
        tickvals: Object.keys(color_mapping).map((_, i) => i),
        ticktext: Object.keys(color_mapping)
      }
    },
    pathbar: {
      visible: true
    }
  };

  const layout = {
    title: title,
    font: { size: 12 },
    height: 500,
    annotations: [
      {
        text: `Total Items: ${data.num_items} | Total Value: ${data.total_value.toLocaleString()}`,
        showarrow: false,
        x: 0,
        y: -0.1,
        xref: 'paper',
        yref: 'paper',
        font: { size: 10, color: 'gray' }
      }
    ]
  };

  return <DataVisualization title={title} data={[treemapTrace]} layout={layout} />;
};

export interface ComprehensiveABCData {
  prod_chart: {
    x: number[];
    y: number[];
    labels: string[];
    colors: string[];
    type: string;
    cumulative: boolean;
    title: string;
  };
  cust_chart: {
    x: number[];
    y: number[];
    labels: string[];
    colors: string[];
    type: string;
    cumulative: boolean;
    title: string;
  };
  summary: {
    total_products: number;
    total_customers: number;
    total_value: number;
    prod_thresholds: number[];
    cust_thresholds: number[];
    value_column: string;
  };
  category_prod: { [key: number]: string[] };
  category_cust: { [key: number]: string[] };
}

export const ComprehensiveABCChart: React.FC<{ data: ComprehensiveABCData }> = ({ data }) => {
  const { prod_chart, cust_chart, summary, category_prod, category_cust } = data;
  
  // Create product chart
  const prodTrace = {
    x: prod_chart.labels,
    y: prod_chart.y,
    type: 'bar' as const,
    marker: {
      color: prod_chart.colors
    },
    name: 'Products',
    hovertemplate: '<b>%{x}</b><br>' +
                  (prod_chart.cumulative ? 'Cumulative %: %{y:.1%}<br>' : 'Value: %{y}<br>') +
                  '<extra></extra>',
  };

  const prodLayout = {
    title: prod_chart.title,
    xaxis: { title: 'Products', tickangle: -45 },
    yaxis: { 
      title: prod_chart.cumulative ? 'Cumulative Percentage' : summary.value_column,
      tickformat: prod_chart.cumulative ? '.0%' : undefined
    },
    height: 400,
  };

  // Create customer chart
  const custTrace = {
    x: cust_chart.labels,
    y: cust_chart.y,
    type: 'bar' as const,
    marker: {
      color: cust_chart.colors
    },
    name: 'Customers',
    hovertemplate: '<b>%{x}</b><br>' +
                  (cust_chart.cumulative ? 'Cumulative %: %{y:.1%}<br>' : 'Value: %{y}<br>') +
                  '<extra></extra>',
  };

  const custLayout = {
    title: cust_chart.title,
    xaxis: { title: 'Customers', tickangle: -45 },
    yaxis: { 
      title: cust_chart.cumulative ? 'Cumulative Percentage' : summary.value_column,
      tickformat: cust_chart.cumulative ? '.0%' : undefined
    },
    height: 400,
  };

  return (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                包括的ABC分析結果サマリー
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">総製品数</Typography>
                  <Typography variant="h6">{summary.total_products}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">総顧客数</Typography>
                  <Typography variant="h6">{summary.total_customers}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">総需要量</Typography>
                  <Typography variant="h6">{summary.total_value.toLocaleString()}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="body2" color="text.secondary">分析対象列</Typography>
                  <Typography variant="h6">{summary.value_column}</Typography>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>製品ABC分布:</Typography>
                {Object.entries(category_prod).map(([rank, items]) => {
                  const label = String.fromCharCode(65 + parseInt(rank)); // A, B, C, ...
                  return (
                    <Typography key={rank} variant="body2" sx={{ ml: 1 }}>
                      {label}ランク: {items.length}製品 ({items.join(', ')})
                    </Typography>
                  );
                })}
              </Box>

              <Box sx={{ mt: 1 }}>
                <Typography variant="subtitle2" gutterBottom>顧客ABC分布:</Typography>
                {Object.entries(category_cust).map(([rank, items]) => {
                  const label = String.fromCharCode(65 + parseInt(rank)); // A, B, C, ...
                  return (
                    <Typography key={rank} variant="body2" sx={{ ml: 1 }}>
                      {label}ランク: {items.length}顧客 
                    </Typography>
                  );
                })}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <DataVisualization title={prod_chart.title} data={[prodTrace]} layout={prodLayout} />
        </Grid>

        <Grid item xs={12} md={6}>
          <DataVisualization title={cust_chart.title} data={[custTrace]} layout={custLayout} />
        </Grid>
      </Grid>
    </Box>
  );
};

export interface AdvancedRankAnalysisData {
  time_series_data: Array<{
    x: string[];
    y: number[];
    name: string;
    color: string;
    mode: string;
  }>;
  x_range: string[];
  title: string;
  x_label: string;
  y_label: string;
  agg_period: string;
  top_products: string[];
  summary: {
    total_periods: number;
    total_products_analyzed: number;
    top_rank_displayed: number;
    aggregation_period: string;
    value_column: string;
  };
}

export const AdvancedRankAnalysisChart: React.FC<{ data: AdvancedRankAnalysisData }> = ({ data }) => {
  const { time_series_data, title, x_label, y_label, summary } = data;
  
  // Create traces for each product
  const traces = time_series_data.map(series => ({
    x: series.x,
    y: series.y,
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: series.name,
    line: {
      color: series.color,
      width: 2
    },
    marker: {
      color: series.color,
      size: 6
    },
    hovertemplate: '<b>%{fullData.name}</b><br>' +
                  'Period: %{x}<br>' +
                  'Rank: %{y}<br>' +
                  '<extra></extra>'
  }));

  const layout = {
    title: title,
    xaxis: { 
      title: x_label,
      tickangle: -45
    },
    yaxis: { 
      title: y_label,
      autorange: 'reversed' as const, // Lower rank (better) at top
      dtick: 1 // Show integer ranks only
    },
    height: 500,
    hovermode: 'closest' as const,
    legend: {
      orientation: 'v' as const,
      x: 1.05,
      y: 1
    }
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            高度ランク分析結果サマリー
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">総期間数</Typography>
              <Typography variant="h6">{summary.total_periods}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">分析製品数</Typography>
              <Typography variant="h6">{summary.total_products_analyzed}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">表示上位数</Typography>
              <Typography variant="h6">{summary.top_rank_displayed}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">集約期間</Typography>
              <Typography variant="h6">{summary.aggregation_period}</Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <DataVisualization title={title} data={traces} layout={layout} />
    </Box>
  );
};

export interface InventoryReductionData {
  chart_data: {
    x: string[];
    y: number[];
    colors: string[];
    type: string;
  };
  title: string;
  x_label: string;
  y_label: string;
  statistics: {
    total_reduction: number;
    average_reduction: number;
    max_reduction: number;
    min_reduction: number;
    num_products: number;
  };
  products: string[];
  summary: {
    message: string;
  };
}

export const InventoryReductionChart: React.FC<{ data: InventoryReductionData }> = ({ data }) => {
  const { chart_data, title, x_label, y_label, statistics, summary } = data;
  
  // Create bar chart trace
  const barTrace = {
    x: chart_data.x,
    y: chart_data.y,
    type: 'bar' as const,
    marker: {
      color: chart_data.colors,
    },
    name: 'Reduction Amount',
    hovertemplate: '<b>%{x}</b><br>' +
                  'Reduction: %{y:.2f}<br>' +
                  '<extra></extra>',
  };

  const layout = {
    title: title,
    xaxis: { 
      title: x_label,
      tickangle: -45
    },
    yaxis: { 
      title: y_label,
    },
    height: 500,
    showlegend: false,
  };

  return (
    <Box>
      <Card elevation={2} sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            在庫削減分析結果サマリー
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">対象製品数</Typography>
              <Typography variant="h6">{statistics.num_products}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">総削減量</Typography>
              <Typography variant="h6">{statistics.total_reduction.toFixed(2)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">平均削減量</Typography>
              <Typography variant="h6">{statistics.average_reduction.toFixed(2)}</Typography>
            </Grid>
            <Grid item xs={3}>
              <Typography variant="body2" color="text.secondary">最大削減量</Typography>
              <Typography variant="h6">{statistics.max_reduction.toFixed(2)}</Typography>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {summary.message}
            </Typography>
          </Box>
        </CardContent>
      </Card>
      
      <DataVisualization title={title} data={[barTrace]} layout={layout} />
    </Box>
  );
};

export default DataVisualization;