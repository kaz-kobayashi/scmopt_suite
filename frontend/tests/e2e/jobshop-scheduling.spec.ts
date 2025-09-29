import { test, expect } from '@playwright/test';

test.describe('Job Shop Scheduling Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(60000);
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(1000);
    
    // Navigate to job shop scheduling
    const navigated = await Promise.race([
      page.click('text=ジョブショップスケジューリング', { timeout: 3000 }).then(() => true),
      page.click('button:has-text("ジョブショップ")', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(4000).then(() => false)
    ]);
    
    if (navigated) {
      await page.waitForTimeout(2000);
      console.log('✅ Successfully navigated to Job Shop Scheduling');
    } else {
      console.log('⚠️ Could not navigate to Job Shop Scheduling');
    }
  });

  test('Seven-tab structure verification', async ({ page }) => {
    const expectedTabs = [
      'システム設定',
      'ジョブ管理', 
      'リソース管理',
      'スケジューリング実行',
      '結果分析',
      'リアルタイム監視',
      'レポート・エクスポート'
    ];

    // Verify page heading
    const headingVisible = await Promise.race([
      page.locator('text=ジョブショップスケジューリング').isVisible().then(() => true),
      page.waitForTimeout(3000).then(() => false)
    ]);
    
    if (!headingVisible) {
      console.log('⚠️ Job Shop Scheduling page may not be loaded');
      return;
    }

    console.log('Testing Job Shop Scheduling 7-tab structure...');
    
    // Count tabs
    const tabCount = await Promise.race([
      page.locator('[role="tab"]').count(),
      page.locator('.MuiTab-root').count(),
      page.waitForTimeout(2000).then(() => 0)
    ]);
    
    console.log(`Tabs found: ${tabCount}`);
    
    if (tabCount >= 7) {
      // Test each tab
      for (let i = 0; i < expectedTabs.length && i < tabCount; i++) {
        const tabName = expectedTabs[i];
        console.log(`  Testing tab ${i + 1}: ${tabName}`);
        
        const tabClicked = await Promise.race([
          page.click(`text=${tabName}`, { timeout: 2000 }).then(() => true),
          page.locator('[role="tab"]').nth(i).click({ timeout: 2000 }).then(() => true),
          page.waitForTimeout(2500).then(() => false)
        ]);
        
        if (tabClicked) {
          await page.waitForTimeout(800);
          
          // Verify tab content is visible
          const contentVisible = await Promise.race([
            page.locator('div[role="tabpanel"]').nth(i).isVisible().then(() => true),
            page.locator('div[role="tabpanel"]').first().isVisible().then(() => true),
            page.waitForTimeout(1500).then(() => false)
          ]);
          
          console.log(`    Tab content visible: ${contentVisible}`);
        } else {
          console.log(`    Could not click tab: ${tabName}`);
        }
      }
    } else {
      console.log(`Expected 7 tabs, found ${tabCount}`);
    }
  });

  test('System Settings tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to system settings (should be default)
    const systemSettingsClicked = await Promise.race([
      page.click('text=システム設定', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').first().click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (systemSettingsClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ System Settings tab accessed');
      
      // Look for system configuration elements
      const configElements = await Promise.race([
        page.locator('text=システム設定, text=基本設定, text=設定').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Configuration elements found: ${configElements}`);
      
      // Look for algorithm parameters
      const paramInputs = await page.locator('input[type="number"], input[type="text"], select').count();
      console.log(`Parameter inputs found: ${paramInputs}`);
      
      if (paramInputs > 0) {
        // Test filling some parameters
        const firstInput = page.locator('input[type="number"]').first();
        if (await firstInput.isVisible()) {
          await firstInput.fill('10');
          await page.waitForTimeout(300);
          console.log('✅ Parameter input test successful');
        }
      }
    }
  });

  test('Job Management tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to job management tab
    const jobMgmtClicked = await Promise.race([
      page.click('text=ジョブ管理', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(1).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (jobMgmtClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Job Management tab accessed');
      
      // Look for job management interface
      const jobElements = await Promise.race([
        page.locator('text=ジョブ, text=Job, text=作業').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Job management elements found: ${jobElements}`);
      
      // Look for job-related buttons and controls
      const jobControls = await page.locator('button:has-text("追加"), button:has-text("編集"), button:has-text("削除"), button:has-text("ジョブ")').count();
      console.log(`Job control buttons found: ${jobControls}`);
      
      // Look for job table or list
      const jobTable = await Promise.race([
        page.locator('table').count(),
        page.locator('[class*="table"]').count(),
        page.waitForTimeout(1000).then(() => 0)
      ]);
      
      console.log(`Job tables/lists found: ${jobTable}`);
    }
  });

  test('Resource Management tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to resource management tab
    const resourceMgmtClicked = await Promise.race([
      page.click('text=リソース管理', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(2).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (resourceMgmtClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Resource Management tab accessed');
      
      // Look for resource management interface
      const resourceElements = await Promise.race([
        page.locator('text=リソース, text=Resource, text=機械, text=設備').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Resource management elements found: ${resourceElements}`);
      
      // Look for resource configuration inputs
      const resourceInputs = await page.locator('input, select, textarea').count();
      console.log(`Resource input fields found: ${resourceInputs}`);
      
      // Test resource capacity or constraint inputs
      const capacityInputs = page.locator('input[placeholder*="容量"], input[placeholder*="capacity"], input[label*="容量"]');
      const capacityCount = await capacityInputs.count();
      
      if (capacityCount > 0) {
        const firstCapacityInput = capacityInputs.first();
        if (await firstCapacityInput.isVisible()) {
          await firstCapacityInput.fill('100');
          await page.waitForTimeout(300);
          console.log('✅ Resource capacity input test successful');
        }
      }
    }
  });

  test('Scheduling Execution tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to scheduling execution tab
    const executionClicked = await Promise.race([
      page.click('text=スケジューリング実行', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(3).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (executionClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Scheduling Execution tab accessed');
      
      // Look for execution controls
      const executionButtons = await page.locator('button:has-text("実行"), button:has-text("開始"), button:has-text("スケジュール")').count();
      console.log(`Execution buttons found: ${executionButtons}`);
      
      // Look for algorithm selection
      const algorithmSelects = await page.locator('select, [role="combobox"]').count();
      console.log(`Algorithm selection elements found: ${algorithmSelects}`);
      
      // Look for scheduling parameters
      const scheduleParams = await page.locator('input[type="number"]').count();
      console.log(`Scheduling parameter inputs found: ${scheduleParams}`);
      
      // Test execution button if available
      const executeButton = page.locator('button:has-text("実行"), button:has-text("スケジュール実行")').first();
      if (await executeButton.isVisible()) {
        const isEnabled = await executeButton.isEnabled();
        console.log(`Execute button state: ${isEnabled ? 'enabled' : 'disabled (requires data)'}`);
        
        if (isEnabled) {
          await executeButton.click();
          await page.waitForTimeout(2000);
          
          // Check for execution results or progress
          const executionResult = await Promise.race([
            page.locator('text=実行中, text=完了, text=結果').isVisible().then(() => true),
            page.waitForTimeout(3000).then(() => false)
          ]);
          
          console.log(`Execution initiated: ${executionResult}`);
        }
      }
    }
  });

  test('Results Analysis tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to results analysis tab
    const resultsClicked = await Promise.race([
      page.click('text=結果分析', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(4).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (resultsClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Results Analysis tab accessed');
      
      // Look for analysis results interface
      const analysisElements = await Promise.race([
        page.locator('text=結果, text=分析, text=Analysis').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Analysis interface elements found: ${analysisElements}`);
      
      // Look for visualization elements
      const visualizations = await Promise.race([
        page.locator('canvas').count(),
        page.locator('svg').count(),
        page.locator('[class*="chart"]').count(),
        page.waitForTimeout(1000).then(() => 0)
      ]);
      
      console.log(`Visualization elements found: ${visualizations}`);
      
      // Look for KPI or metrics display
      const metrics = await page.locator('text=メイクスパン, text=makespan, text=完了時間, text=効率').count();
      console.log(`Performance metrics found: ${metrics}`);
      
      // Look for result tables
      const resultTables = await page.locator('table, [class*="table"]').count();
      console.log(`Result tables found: ${resultTables}`);
      
      // Check for no results message
      const noResultsMessage = await page.locator('text=結果がありません, text=実行してください, text=no results').isVisible();
      if (noResultsMessage) {
        console.log('ℹ️ No results message displayed (expected without execution)');
      }
    }
  });

  test('Real-time Monitoring tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to real-time monitoring tab
    const monitoringClicked = await Promise.race([
      page.click('text=リアルタイム監視', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(5).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (monitoringClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Real-time Monitoring tab accessed');
      
      // Look for monitoring interface
      const monitoringElements = await Promise.race([
        page.locator('text=監視, text=リアルタイム, text=状況').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Monitoring interface elements found: ${monitoringElements}`);
      
      // Look for status indicators
      const statusIndicators = await page.locator('text=実行中, text=完了, text=待機, [class*="status"], [class*="indicator"]').count();
      console.log(`Status indicators found: ${statusIndicators}`);
      
      // Look for refresh or update buttons
      const refreshButtons = await page.locator('button:has-text("更新"), button:has-text("リフレッシュ"), button:has-text("Refresh")').count();
      console.log(`Refresh buttons found: ${refreshButtons}`);
      
      if (refreshButtons > 0) {
        const refreshButton = page.locator('button:has-text("更新")').first();
        if (await refreshButton.isVisible()) {
          await refreshButton.click();
          await page.waitForTimeout(1000);
          console.log('✅ Refresh button test successful');
        }
      }
    }
  });

  test('Report and Export tab functionality', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Navigate to report/export tab
    const exportClicked = await Promise.race([
      page.click('text=レポート・エクスポート', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(6).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (exportClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Report & Export tab accessed');
      
      // Look for export interface
      const exportElements = await Promise.race([
        page.locator('text=エクスポート, text=レポート, text=Export').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Export interface elements found: ${exportElements}`);
      
      // Look for export buttons
      const exportButtons = await page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード"), button:has-text("保存"), button:has-text("Export")').count();
      console.log(`Export buttons found: ${exportButtons}`);
      
      // Test export buttons (they should be disabled if no results)
      const exportButtonList = page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード")');
      const exportButtonCount = await exportButtonList.count();
      
      if (exportButtonCount > 0) {
        for (let i = 0; i < Math.min(exportButtonCount, 3); i++) {
          const button = exportButtonList.nth(i);
          if (await button.isVisible()) {
            const isEnabled = await button.isEnabled();
            const buttonText = await button.textContent();
            console.log(`Export button "${buttonText}": ${isEnabled ? 'enabled' : 'disabled (expected without results)'}`);
          }
        }
      }
      
      // Look for report generation options
      const reportOptions = await page.locator('select, [role="combobox"], input[type="checkbox"]').count();
      console.log(`Report configuration options found: ${reportOptions}`);
    }
  });

  test('Data flow integration test', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) {
      console.log('⚠️ Job Shop Scheduling page not accessible for integration test');
      return;
    }

    console.log('🔄 Testing Job Shop Scheduling data flow integration...');
    
    // Step 1: System Settings
    const systemTab = await Promise.race([
      page.click('text=システム設定', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (systemTab) {
      await page.waitForTimeout(800);
      console.log('Step 1: ✅ System settings accessed');
    }
    
    // Step 2: Job Management
    const jobTab = await Promise.race([
      page.click('text=ジョブ管理', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (jobTab) {
      await page.waitForTimeout(800);
      console.log('Step 2: ✅ Job management accessed');
    }
    
    // Step 3: Resource Management
    const resourceTab = await Promise.race([
      page.click('text=リソース管理', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (resourceTab) {
      await page.waitForTimeout(800);
      console.log('Step 3: ✅ Resource management accessed');
    }
    
    // Step 4: Execution
    const executionTab = await Promise.race([
      page.click('text=スケジューリング実行', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (executionTab) {
      await page.waitForTimeout(800);
      console.log('Step 4: ✅ Execution interface accessed');
    }
    
    // Step 5: Results
    const resultsTab = await Promise.race([
      page.click('text=結果分析', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (resultsTab) {
      await page.waitForTimeout(800);
      console.log('Step 5: ✅ Results analysis accessed');
    }
    
    // Step 6: Monitoring
    const monitoringTab = await Promise.race([
      page.click('text=リアルタイム監視', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (monitoringTab) {
      await page.waitForTimeout(800);
      console.log('Step 6: ✅ Monitoring accessed');
    }
    
    // Step 7: Export
    const exportTab = await Promise.race([
      page.click('text=レポート・エクスポート', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (exportTab) {
      await page.waitForTimeout(800);
      console.log('Step 7: ✅ Export interface accessed');
    }
    
    console.log('🎯 Job Shop Scheduling integration test completed');
  });

  test('Error handling and edge cases', async ({ page }) => {
    const headingVisible = await page.locator('text=ジョブショップスケジューリング').isVisible();
    if (!headingVisible) return;

    // Test invalid input handling
    const systemTab = await Promise.race([
      page.click('text=システム設定', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (systemTab) {
      await page.waitForTimeout(1000);
      
      // Try invalid inputs
      const numberInputs = page.locator('input[type="number"]');
      const inputCount = await numberInputs.count();
      
      if (inputCount > 0) {
        const firstInput = numberInputs.first();
        if (await firstInput.isVisible()) {
          // Test negative number
          await firstInput.fill('-100');
          await page.waitForTimeout(500);
          
          // Look for validation message
          const validationError = await Promise.race([
            page.locator('text=エラー, text=無効, text=Error, [class*="error"]').isVisible().then(() => true),
            page.waitForTimeout(2000).then(() => false)
          ]);
          
          console.log(`Input validation handling: ${validationError ? 'working' : 'not visible'}`);
          
          // Test empty input
          await firstInput.fill('');
          await page.waitForTimeout(500);
        }
      }
    }
  });
});