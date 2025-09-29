import React, { useState } from 'react';
import {
  AppBar,
  Box,
  CssBaseline,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  BarChart as AnalyticsIcon,
  Storage as InventoryIcon,
  DirectionsCar as RouteIcon,
  Dashboard as DashboardIcon,
  LocationOn as LocationIcon,
  Schedule as ScheduleIcon,
  Factory as FactoryIcon,
  Description as TemplateIcon,
  Security as SecurityIcon,
  Hub as SNDIcon,
  TrendingUp as RMIcon,
  Assignment as AdvancedVRPIcon,
} from '@mui/icons-material';

const drawerWidth = 240;

interface LayoutProps {
  children: React.ReactNode;
  onNavigate: (section: string) => void;
  currentSection: string;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactElement;
}

const navItems: NavItem[] = [
  { id: 'dashboard', label: 'ダッシュボード', icon: <DashboardIcon /> },
  { id: 'analytics', label: '分析', icon: <AnalyticsIcon /> },
  { id: 'inventory', label: '在庫管理', icon: <InventoryIcon /> },
  { id: 'routing', label: '配送ルーティング', icon: <RouteIcon /> },
  { id: 'pyvrp', label: 'PyVRP - 車両配送最適化', icon: <AdvancedVRPIcon /> },
  { id: 'lnd', label: '物流ネットワーク設計', icon: <LocationIcon /> },
  { id: 'scrm', label: 'サプライチェインリスク管理', icon: <SecurityIcon /> },
  { id: 'snd', label: 'サービスネットワーク設計', icon: <SNDIcon /> },
  { id: 'rm', label: '収益管理 (MERMO)', icon: <RMIcon /> },
  { id: 'shift', label: 'シフト最適化', icon: <ScheduleIcon /> },
  { id: 'jobshop', label: 'ジョブショップスケジューリング', icon: <FactoryIcon /> },
  { id: 'templates', label: 'スケジュールテンプレート', icon: <TemplateIcon /> },
];

const Layout: React.FC<LayoutProps> = ({ children, onNavigate, currentSection }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          SCMOPT2
        </Typography>
      </Toolbar>
      <List>
        {navItems.map((item) => (
          <ListItem key={item.id} disablePadding>
            <ListItemButton
              selected={currentSection === item.id}
              onClick={() => {
                onNavigate(item.id);
                if (isMobile) {
                  setMobileOpen(false);
                }
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            サプライチェーンマネジメント最適化スイート
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="navigation menu"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
};

export default Layout;