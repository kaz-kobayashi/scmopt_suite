import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  Chip
} from '@mui/material';

interface EchelonPolicy {
  EOQ: number;
  Safety_Stock: number;
  Reorder_Point: number;
}

interface EchelonPolicies {
  Plant: EchelonPolicy;
  DC: EchelonPolicy;
  Retail: EchelonPolicy;
}

interface EchelonPolicyTableProps {
  echelonPolicies: EchelonPolicies;
  networkStructure: {
    Plants: number;
    DCs: number;
    Retailers: number;
  };
}

const EchelonPolicyTable: React.FC<EchelonPolicyTableProps> = ({ 
  echelonPolicies, 
  networkStructure 
}) => {
  const echelonData = [
    {
      level: 'Plant (工場)',
      nodes: networkStructure.Plants,
      eoq: echelonPolicies.Plant.EOQ,
      safetyStock: echelonPolicies.Plant.Safety_Stock,
      reorderPoint: echelonPolicies.Plant.Reorder_Point,
      color: '#2E8B57',
      description: '製造・供給拠点'
    },
    {
      level: 'DC (配送センター)',
      nodes: networkStructure.DCs,
      eoq: echelonPolicies.DC.EOQ,
      safetyStock: echelonPolicies.DC.Safety_Stock,
      reorderPoint: echelonPolicies.DC.Reorder_Point,
      color: '#4169E1',
      description: '在庫配分・配送拠点'
    },
    {
      level: 'Retail (小売店)',
      nodes: networkStructure.Retailers,
      eoq: echelonPolicies.Retail.EOQ,
      safetyStock: echelonPolicies.Retail.Safety_Stock,
      reorderPoint: echelonPolicies.Retail.Reorder_Point,
      color: '#FF6347',
      description: '最終販売拠点'
    }
  ];

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        エシェロン別在庫ポリシー詳細
      </Typography>
      
      <TableContainer component={Paper} elevation={2}>
        <Table>
          <TableHead>
            <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
              <TableCell sx={{ fontWeight: 'bold' }}>エシェロンレベル</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>ノード数</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>EOQ</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>安全在庫</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>発注点</TableCell>
              <TableCell align="center" sx={{ fontWeight: 'bold' }}>在庫回転率</TableCell>
              <TableCell sx={{ fontWeight: 'bold' }}>役割</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {echelonData.map((row) => {
              const turnoverRate = row.eoq > 0 ? (365 / (row.eoq / 50)).toFixed(1) : '0'; // Estimated turnover
              
              return (
                <TableRow 
                  key={row.level}
                  sx={{ 
                    '&:nth-of-type(odd)': { backgroundColor: '#fafafa' },
                    '&:hover': { backgroundColor: '#e3f2fd' }
                  }}
                >
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          backgroundColor: row.color,
                          borderRadius: '50%',
                          mr: 1
                        }}
                      />
                      <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                        {row.level}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="center">
                    <Chip 
                      label={row.nodes} 
                      size="small" 
                      sx={{ backgroundColor: row.color, color: 'white' }}
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {row.eoq.toFixed(0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      単位
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {row.safetyStock.toFixed(0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      単位
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {row.reorderPoint.toFixed(0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      単位
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {turnoverRate}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      回/年
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption" color="text.secondary">
                      {row.description}
                    </Typography>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      
      {/* Summary statistics */}
      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Box sx={{ p: 1, backgroundColor: '#e8f5e8', borderRadius: 1, minWidth: 120 }}>
          <Typography variant="caption" color="text.secondary">
            総ノード数
          </Typography>
          <Typography variant="h6" color="#2E8B57">
            {networkStructure.Plants + networkStructure.DCs + networkStructure.Retailers}
          </Typography>
        </Box>
        
        <Box sx={{ p: 1, backgroundColor: '#e8f0ff', borderRadius: 1, minWidth: 120 }}>
          <Typography variant="caption" color="text.secondary">
            平均EOQ
          </Typography>
          <Typography variant="h6" color="#4169E1">
            {Math.round((echelonPolicies.Plant.EOQ + echelonPolicies.DC.EOQ + echelonPolicies.Retail.EOQ) / 3)}
          </Typography>
        </Box>
        
        <Box sx={{ p: 1, backgroundColor: '#ffe8e8', borderRadius: 1, minWidth: 120 }}>
          <Typography variant="caption" color="text.secondary">
            平均安全在庫
          </Typography>
          <Typography variant="h6" color="#FF6347">
            {Math.round((echelonPolicies.Plant.Safety_Stock + echelonPolicies.DC.Safety_Stock + echelonPolicies.Retail.Safety_Stock) / 3)}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default EchelonPolicyTable;