import React from 'react';
import { Box, Typography, Card, CardContent, Grid } from '@mui/material';
import Plot from 'react-plotly.js';
import EchelonPolicyTable from './EchelonPolicyTable';

interface MultiEchelonResult {
  total_cost: number;
  service_level: number;
  total_inventory: number;
  network_structure: {
    Plants: number;
    DCs: number;
    Retailers: number;
  };
  recommendations: string[];
  echelon_policies: {
    Plant: {
      EOQ: number;
      Safety_Stock: number;
      Reorder_Point: number;
    };
    DC: {
      EOQ: number;
      Safety_Stock: number;
      Reorder_Point: number;
    };
    Retail: {
      EOQ: number;
      Safety_Stock: number;
      Reorder_Point: number;
    };
  };
}

interface MultiEchelonVisualizationProps {
  result: MultiEchelonResult;
}

const MultiEchelonVisualization: React.FC<MultiEchelonVisualizationProps> = ({ result }) => {
  // Network structure data for visualization
  const networkStructureData = [
    {
      x: ['Plants', 'Distribution Centers', 'Retailers'],
      y: [result.network_structure.Plants, result.network_structure.DCs, result.network_structure.Retailers],
      type: 'bar' as const,
      marker: {
        color: ['#2E8B57', '#4169E1', '#FF6347']
      },
      name: 'Network Nodes'
    }
  ];

  // Cost breakdown (estimated based on typical multi-echelon costs)
  const totalCost = result.total_cost;
  const holdingCostRatio = 0.4;
  const orderingCostRatio = 0.3;
  const shortfallCostRatio = 0.3;

  const costBreakdownData = [
    {
      labels: ['保管コスト', '発注コスト', '欠品コスト'],
      values: [
        totalCost * holdingCostRatio,
        totalCost * orderingCostRatio,
        totalCost * shortfallCostRatio
      ],
      type: 'pie' as const,
      marker: {
        colors: ['#FF6B6B', '#4ECDC4', '#45B7D1']
      }
    }
  ];

  // Echelon comparison data
  const echelonData = {
    eoq: [
      result.echelon_policies.Plant.EOQ,
      result.echelon_policies.DC.EOQ,
      result.echelon_policies.Retail.EOQ
    ],
    safetyStock: [
      result.echelon_policies.Plant.Safety_Stock,
      result.echelon_policies.DC.Safety_Stock,
      result.echelon_policies.Retail.Safety_Stock
    ],
    reorderPoint: [
      result.echelon_policies.Plant.Reorder_Point,
      result.echelon_policies.DC.Reorder_Point,
      result.echelon_policies.Retail.Reorder_Point
    ]
  };

  const echelonComparisonData = [
    {
      x: ['Plant', 'DC', 'Retail'],
      y: echelonData.eoq,
      name: 'EOQ',
      type: 'bar' as const,
      marker: { color: '#FF6B6B' }
    },
    {
      x: ['Plant', 'DC', 'Retail'],
      y: echelonData.safetyStock,
      name: '安全在庫',
      type: 'bar' as const,
      marker: { color: '#4ECDC4' }
    },
    {
      x: ['Plant', 'DC', 'Retail'],
      y: echelonData.reorderPoint,
      name: '発注点',
      type: 'bar' as const,
      marker: { color: '#45B7D1' }
    }
  ];

  // Network flow diagram data (simplified)
  const networkFlowData = [
    {
      x: [1, 2, 3],
      y: [3, 2, 1],
      mode: 'markers+text' as const,
      marker: {
        size: [60, 80, 100],
        color: ['#2E8B57', '#4169E1', '#FF6347'],
        opacity: 0.8
      },
      text: [
        `Plants (${result.network_structure.Plants})`,
        `DCs (${result.network_structure.DCs})`,
        `Retailers (${result.network_structure.Retailers})`
      ],
      textposition: 'middle center' as const,
      textfont: {
        color: 'white',
        size: 12,
        family: 'Arial Black'
      },
      type: 'scatter' as const,
      showlegend: false
    },
    // Flow arrows (simplified representation)
    {
      x: [1.5, 2.5],
      y: [2.5, 1.5],
      mode: 'markers' as const,
      marker: {
        symbol: 'arrow-right' as const,
        size: 20,
        color: '#666666'
      },
      showlegend: false,
      type: 'scatter' as const
    }
  ];

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        マルチエシェロン最適化結果の可視化
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics Cards */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                最適化指標
              </Typography>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  総コスト
                </Typography>
                <Typography variant="h5">
                  ¥{result.total_cost.toFixed(2)}
                </Typography>
              </Box>
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  サービスレベル
                </Typography>
                <Typography variant="h5" color="success.main">
                  {(result.service_level * 100).toFixed(1)}%
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  総在庫量
                </Typography>
                <Typography variant="h5">
                  {result.total_inventory.toFixed(0)} 単位
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Network Structure Bar Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ネットワーク構造
              </Typography>
              <Plot
                data={networkStructureData as any}
                layout={{
                  title: { text: 'エシェロン別ノード数' },
                  xaxis: { title: { text: 'エシェロンレベル' } },
                  yaxis: { title: { text: 'ノード数' } },
                  height: 300,
                  margin: { l: 60, r: 20, t: 40, b: 60 }
                }}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Network Flow Diagram */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ネットワークフロー図
              </Typography>
              <Plot
                data={networkFlowData as any}
                layout={{
                  title: { text: 'サプライチェーンフロー' },
                  xaxis: { 
                    range: [0.5, 3.5],
                    showgrid: false,
                    showticklabels: false,
                    zeroline: false
                  },
                  yaxis: { 
                    range: [0.5, 3.5],
                    showgrid: false,
                    showticklabels: false,
                    zeroline: false
                  },
                  height: 300,
                  margin: { l: 20, r: 20, t: 40, b: 20 },
                  annotations: [
                    {
                      x: 1.5,
                      y: 2.5,
                      text: '→',
                      showarrow: false,
                      font: { size: 20, color: '#666666' }
                    },
                    {
                      x: 2.5,
                      y: 1.5,
                      text: '→',
                      showarrow: false,
                      font: { size: 20, color: '#666666' }
                    }
                  ]
                }}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Cost Breakdown Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                コスト内訳
              </Typography>
              <Plot
                data={costBreakdownData as any}
                layout={{
                  title: { text: '総コストの内訳' },
                  height: 300,
                  margin: { l: 20, r: 20, t: 40, b: 20 }
                }}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Echelon Policy Comparison */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                エシェロン別在庫ポリシー比較
              </Typography>
              <Plot
                data={echelonComparisonData as any}
                layout={{
                  title: { text: 'EOQ・安全在庫・発注点の比較' },
                  xaxis: { title: { text: 'エシェロンレベル' } },
                  yaxis: { title: { text: '数量' } },
                  barmode: 'group',
                  height: 400,
                  margin: { l: 60, r: 20, t: 40, b: 60 }
                }}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                推奨事項
              </Typography>
              <Box>
                {result.recommendations.map((recommendation, index) => (
                  <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                    • {recommendation}
                  </Typography>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Detailed Policy Table */}
        <Grid item xs={12}>
          <EchelonPolicyTable 
            echelonPolicies={result.echelon_policies}
            networkStructure={result.network_structure}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default MultiEchelonVisualization;