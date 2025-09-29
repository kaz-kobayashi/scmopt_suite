import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import Register from './components/Register';
import Analytics from './components/Analytics';
import Inventory from './components/Inventory';
import Routing from './components/Routing';
import LogisticsNetworkDesign from './components/LogisticsNetworkDesign';
import ShiftOptimization from './components/ShiftOptimization';
import JobShopScheduling from './components/JobShopScheduling';
import ScheduleTemplates from './components/ScheduleTemplates';
import SCRMAnalysis from './components/SCRMAnalysis';
import SNDAnalysis from './components/SNDAnalysis';
import RMAnalysis from './components/RMAnalysis';
import PyVRPInterface from './components/PyVRPInterface';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#ff9800',
    },
  },
  typography: {
    h3: {
      fontWeight: 600,
    },
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
});

const AppContent: React.FC = () => {
  const [currentSection, setCurrentSection] = useState('dashboard');
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const { user, isLoading } = useAuth();

  const handleNavigate = (section: string) => {
    setCurrentSection(section);
  };

  const renderCurrentSection = () => {
    switch (currentSection) {
      case 'analytics':
        return <Analytics />;
      case 'inventory':
        return <Inventory />;
      case 'routing':
        return <Routing />;
      case 'lnd':
        return <LogisticsNetworkDesign />;
      case 'shift':
        return <ShiftOptimization />;
      case 'jobshop':
        return <JobShopScheduling />;
      case 'templates':
        return <ScheduleTemplates />;
      case 'scrm':
        return <SCRMAnalysis />;
      case 'snd':
        return <SNDAnalysis />;
      case 'rm':
        return <RMAnalysis />;
      case 'pyvrp':
        return <PyVRPInterface />;
      case 'dashboard':
      default:
        return <Dashboard onNavigate={handleNavigate} />;
    }
  };

  if (isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
          読み込み中...
        </div>
      </ThemeProvider>
    );
  }

  if (!user) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {authMode === 'login' ? (
          <Login onSwitchToRegister={() => setAuthMode('register')} />
        ) : (
          <Register onSwitchToLogin={() => setAuthMode('login')} />
        )}
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout 
        onNavigate={handleNavigate}
        currentSection={currentSection}
      >
        {renderCurrentSection()}
      </Layout>
    </ThemeProvider>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
