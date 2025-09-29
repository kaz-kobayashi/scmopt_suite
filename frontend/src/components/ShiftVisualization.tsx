import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import Plot from 'react-plotly.js';
// import { Plotly } from 'react-plotly.js';

interface ShiftVisualizationProps {
  optimizationResult: any;
  sampleData: any;
}

const ShiftVisualization: React.FC<ShiftVisualizationProps> = ({
  optimizationResult,
  sampleData
}) => {
  const [selectedDay, setSelectedDay] = useState<number>(0);
  const [selectedJob, setSelectedJob] = useState<number>(1);
  const [requirementChart, setRequirementChart] = useState<any>(null);
  const [ganttChart, setGanttChart] = useState<any>(null);

  // 必要人数と実際の配置人数の比較チャート
  const generateRequirementChart = () => {
    if (!optimizationResult?.job_assign || !sampleData) return;

    const periods = sampleData.period_df.slice(0, -1); // 最後の期は除く
    const jobs = sampleData.job_df.slice(1); // 休憩を除く
    const dayData = sampleData.day_df[selectedDay];
    const dayType = dayData.day_type;

    // 必要人数データ
    const requirements = sampleData.requirement_df.filter(
      (req: any) => req.day_type === dayType && req.job === selectedJob
    );

    const requiredCounts = periods.map((period: any) => {
      const req = requirements.find((r: any) => r.period === period.id);
      return req ? req.requirement : 0;
    });

    // 実際の配置人数
    const actualCounts = periods.map((period: any) => {
      let count = 0;
      Object.entries(optimizationResult.job_assign).forEach(([key, jobId]) => {
        const [staffId, day, periodId] = key.split('_').map(Number);
        if (day === selectedDay && periodId === period.id && jobId === selectedJob) {
          count++;
        }
      });
      return count;
    });

    const trace1 = {
      x: periods.map((p: any) => p.description),
      y: requiredCounts,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: '必要人数',
      line: { color: 'red', dash: 'dot', width: 3 }
    };

    const trace2 = {
      x: periods.map((p: any) => p.description),
      y: actualCounts,
      type: 'bar' as const,
      name: '配置人数',
      marker: { color: 'rgba(55, 128, 191, 0.7)' }
    };

    const layout = {
      title: { text: `${dayData.day} - ${sampleData.job_df[selectedJob]?.description || 'ジョブ'} の人数配置` },
      xaxis: { title: { text: '時間帯' } },
      yaxis: { title: { text: '人数' } },
      showlegend: true,
      height: 400
    };

    return { data: [trace1, trace2], layout };
  };

  // ガントチャート
  const generateGanttChart = () => {
    if (!optimizationResult?.job_assign || !sampleData) return;

    const staff = sampleData.staff_df;
    const periods = sampleData.period_df;
    const jobs = sampleData.job_df;
    const dayData = sampleData.day_df[selectedDay];

    const ganttData: any[] = [];
    const colors: { [key: string]: string } = {
      '0': '#ff7f0e', // 休憩
      '1': '#2ca02c', // ジョブ1
      '2': '#d62728', // ジョブ2
      '3': '#9467bd', // ジョブ3
      '4': '#8c564b', // ジョブ4
      '5': '#e377c2'  // ジョブ5
    };

    staff.forEach((staffMember: any, staffIndex: number) => {
      const staffName = staffMember.name;
      const dayOff = Array.isArray(staffMember.day_off) ? staffMember.day_off : JSON.parse(staffMember.day_off || '[]');
      
      if (dayOff.includes(selectedDay)) return;

      // このスタッフの一日のスケジュールを構築
      const schedule: { job: number; start: number; end: number }[] = [];
      let currentJob = null;
      let startPeriod = null;

      for (let periodId = 0; periodId < periods.length - 1; periodId++) {
        const key = `${staffIndex}_${selectedDay}_${periodId}`;
        const assignedJob = optimizationResult.job_assign[key];

        if (assignedJob !== undefined) {
          if (currentJob === null) {
            currentJob = assignedJob;
            startPeriod = periodId;
          } else if (currentJob !== assignedJob) {
            // ジョブが変わった
            if (startPeriod !== null) {
              schedule.push({
                job: currentJob,
                start: startPeriod,
                end: periodId
              });
            }
            currentJob = assignedJob;
            startPeriod = periodId;
          }
        } else {
          // 割り当てなし
          if (currentJob !== null && startPeriod !== null) {
            schedule.push({
              job: currentJob,
              start: startPeriod,
              end: periodId
            });
          }
          currentJob = null;
          startPeriod = null;
        }
      }

      // 最後のジョブを追加
      if (currentJob !== null && startPeriod !== null) {
        schedule.push({
          job: currentJob,
          start: startPeriod,
          end: periods.length - 1
        });
      }

      // ガントチャート用データに変換
      schedule.forEach(task => {
        const startTime = periods[task.start]?.description || '00:00';
        const endTime = periods[task.end]?.description || '23:59';
        const jobName = jobs[task.job]?.description || `ジョブ${task.job}`;
        
        ganttData.push({
          Task: staffName,
          Start: `2024-01-01 ${startTime}`,
          Finish: `2024-01-01 ${endTime}`,
          Resource: jobName
        });
      });
    });

    if (ganttData.length === 0) {
      return {
        data: [],
        layout: {
          title: { text: 'スケジュールデータがありません' },
          height: 400
        }
      };
    }

    // Plotly用ガントチャートデータに変換
    const traces: any[] = [];
    const resourceTypes = Array.from(new Set(ganttData.map(d => d.Resource)));
    
    resourceTypes.forEach((resource, index) => {
      const resourceData = ganttData.filter(d => d.Resource === resource);
      
      traces.push({
        x: resourceData.map(d => {
          const start = new Date(d.Start);
          const end = new Date(d.Finish);
          return (end.getTime() - start.getTime()) / (1000 * 60 * 60); // 時間単位
        }),
        y: resourceData.map(d => d.Task),
        base: resourceData.map(d => {
          const start = new Date(d.Start);
          return start.getHours() + start.getMinutes() / 60;
        }),
        type: 'bar' as const,
        orientation: 'h' as const,
        name: resource,
        marker: {
          color: colors[index.toString()] || '#1f77b4'
        }
      });
    });

    const layout = {
      title: { text: `${dayData.day} のスタッフスケジュール` },
      xaxis: { title: { text: '時間' } },
      yaxis: { title: { text: 'スタッフ' } },
      barmode: 'stack' as const,
      height: Math.max(400, staff.length * 30)
    };

    return { data: traces, layout };
  };

  // コスト分析チャート
  const generateCostChart = () => {
    if (!optimizationResult?.cost_df) return null;

    const costData = optimizationResult.cost_df.filter((item: any) => item.value > 0);
    
    if (costData.length === 0) return null;

    const trace = {
      x: costData.map((item: any) => item.penalty),
      y: costData.map((item: any) => item.value),
      type: 'bar' as const,
      marker: { color: 'rgba(255, 99, 132, 0.8)' }
    };

    const layout = {
      title: { text: 'コスト・ペナルティ分析' },
      xaxis: { title: { text: 'ペナルティ項目' } },
      yaxis: { title: { text: '値' } },
      height: 300
    };

    return { data: [trace], layout };
  };

  useEffect(() => {
    setRequirementChart(generateRequirementChart());
    setGanttChart(generateGanttChart());
  }, [selectedDay, selectedJob, optimizationResult, sampleData]);

  if (!optimizationResult || !sampleData) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="h6" color="textSecondary">
          最適化結果がありません
        </Typography>
      </Box>
    );
  }

  const costChart = generateCostChart();

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        シフト最適化結果の可視化
      </Typography>

      <Grid container spacing={3}>
        {/* コントロール */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>表示する日</InputLabel>
                    <Select
                      value={selectedDay}
                      onChange={(e) => setSelectedDay(Number(e.target.value))}
                    >
                      {sampleData.day_df?.map((day: any, index: number) => (
                        <MenuItem key={index} value={index}>
                          {day.day} ({day.day_of_week})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>表示するジョブ</InputLabel>
                    <Select
                      value={selectedJob}
                      onChange={(e) => setSelectedJob(Number(e.target.value))}
                    >
                      {sampleData.job_df?.slice(1).map((job: any) => (
                        <MenuItem key={job.id} value={job.id}>
                          {job.description}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 必要人数 vs 配置人数チャート */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2 }}>
            {requirementChart ? (
              <Plot
                data={requirementChart.data}
                layout={requirementChart.layout}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            ) : (
              <Typography>チャートデータを生成中...</Typography>
            )}
          </Paper>
        </Grid>

        {/* コスト分析チャート */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2 }}>
            {costChart ? (
              <Plot
                data={costChart.data}
                layout={costChart.layout}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography color="textSecondary">
                  コストデータがありません
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* ガントチャート */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              スタッフスケジュール（ガントチャート）
            </Typography>
            {ganttChart ? (
              <Plot
                data={ganttChart.data}
                layout={ganttChart.layout}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            ) : (
              <Typography>ガントチャートを生成中...</Typography>
            )}
          </Paper>
        </Grid>

        {/* 統計情報 */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              最適化統計
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">
                  スタッフ数
                </Typography>
                <Typography variant="h6">
                  {sampleData.staff_df?.length || 0}名
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">
                  計画日数
                </Typography>
                <Typography variant="h6">
                  {sampleData.day_df?.length || 0}日
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">
                  最適化ステータス
                </Typography>
                <Typography variant="h6" color={optimizationResult.status === 0 ? 'success.main' : 'error.main'}>
                  {optimizationResult.status === 0 ? '成功' : 'エラー'}
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="body2" color="textSecondary">
                  総コスト
                </Typography>
                <Typography variant="h6">
                  {optimizationResult.cost_df?.find((item: any) => item.penalty === 'Cost')?.value || 0}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ShiftVisualization;