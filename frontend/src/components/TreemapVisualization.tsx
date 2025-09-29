import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography } from '@mui/material';

interface TreemapData {
  label: string;
  value: number;
  percentage: number;
  parent: string;
}

interface TreemapVisualizationProps {
  data: TreemapData[];
  title: string;
  height?: number;
}

const TreemapVisualization: React.FC<TreemapVisualizationProps> = ({
  data,
  title,
  height = 400
}) => {

  if (!data || data.length === 0) {
    return (
      <Box 
        sx={{ 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          border: '1px dashed #ccc',
          borderRadius: 1
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No data available for treemap visualization
        </Typography>
      </Box>
    );
  }

  // Prepare data for Plotly treemap
  const labels = data.map(item => item.label);
  const values = data.map(item => item.value);
  const parents = data.map(item => item.parent || '');
  const texts = data.map(item => `${item.label}<br>${item.percentage.toFixed(1)}%`);

  const plotData: any[] = [{
    type: 'treemap',
    labels: labels,
    parents: parents,
    values: values,
    text: texts,
    textinfo: 'label+text+value',
    textposition: 'middle center',
    hovertemplate: '<b>%{label}</b><br>Value: %{value}<br>Percentage: %{customdata}%<extra></extra>',
    customdata: data.map(item => item.percentage.toFixed(1)),
    colorscale: [
      [0, '#e3f2fd'],
      [0.5, '#2196f3'],
      [1, '#0d47a1']
    ],
    marker: {
      line: {
        width: 2,
        color: 'white'
      }
    }
  }];

  const layout: any = {
    title: {
      text: title,
      font: {
        size: 16
      }
    },
    font: {
      size: 12
    },
    margin: {
      t: 50,
      l: 25,
      r: 25,
      b: 25
    }
  };

  const config: any = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    displaylogo: false,
    responsive: true
  };

  return (
    <Box sx={{ width: '100%', height }}>
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </Box>
  );
};

export default TreemapVisualization;