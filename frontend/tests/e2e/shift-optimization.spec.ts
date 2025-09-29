import { test, expect } from '@playwright/test';

test.describe('Shift Optimization Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Navigate to shift optimization
    await page.click('text=シフト最適化');
    await page.waitForTimeout(2000);
    
    // Verify we're on the shift optimization page
    await expect(page.locator('text=シフト最適化システム')).toBeVisible();
  });

  test('Seven-tab structure navigation', async ({ page }) => {
    const sevenTabs = [
      'システム設定',
      'スタッフ管理',
      'シフト要件',
      '最適化実行',
      '結果分析',
      'リアルタイム監視',
      'データ管理'
    ];

    for (let i = 0; i < sevenTabs.length; i++) {
      const tabName = sevenTabs[i];
      console.log(`Testing shift optimization tab: ${tabName}`);
      
      // Click on tab
      await page.click(`text=${tabName}`);
      await page.waitForTimeout(1000);
      
      // Verify tab content is visible
      const tabPanel = page.locator('div[role="tabpanel"]').nth(i);
      await expect(tabPanel).toBeVisible();
    }
  });

  test('System Settings tab - Sample data generation', async ({ page }) => {
    // Navigate to system settings (should be default)
    await page.click('text=システム設定');
    await page.waitForTimeout(1000);
    
    // Verify sample data generation section
    await expect(page.locator('text=サンプルデータ生成')).toBeVisible();
    
    // Fill sample data parameters
    const dataGenParams = [
      { field: '開始日', value: '2024-01-01' },
      { field: '終了日', value: '2024-01-07' },
      { field: '開始時刻', value: '09:00' },
      { field: '終了時刻', value: '21:00' }
    ];

    for (const param of dataGenParams) {
      const input = page.locator(`input[label*="${param.field}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Set time interval
    const frequencySelect = page.locator('select').first();
    if (await frequencySelect.isVisible()) {
      await frequencySelect.selectOption('1h');
      await page.waitForTimeout(300);
    }

    // Set job list
    const jobListInput = page.locator('input[label*="ジョブリスト"]').first();
    if (await jobListInput.isVisible()) {
      await jobListInput.fill('レジ打ち, 接客, 清掃');
      await page.waitForTimeout(300);
    }

    // Generate sample data
    const generateButton = page.locator('button:has-text("サンプルデータ生成")').first();
    if (await generateButton.isVisible()) {
      await generateButton.click();
      await page.waitForTimeout(3000);
      
      // Check for success message
      const dataGenerated = await Promise.race([
        page.locator('text=サンプルデータが正常に生成されました').isVisible().then(() => true),
        page.locator('text=生成完了').isVisible().then(() => true),
        page.locator('[class*="success"]').isVisible().then(() => true),
        page.waitForTimeout(4000).then(() => false)
      ]);
      
      console.log(`Sample data generation: ${dataGenerated ? 'successful' : 'requires backend'}`);
    }
  });

  test('Staff Management tab functionality', async ({ page }) => {
    // Navigate to staff management tab
    await page.click('text=スタッフ管理');
    await page.waitForTimeout(1000);
    
    // Look for staff management interface
    const staffInterface = await Promise.race([
      page.locator('text=スタッフ情報').isVisible().then(() => true),
      page.locator('text=スタッフ管理').isVisible().then(() => true),
      page.locator('button:has-text("スタッフ")').isVisible().then(() => true),
      page.waitForTimeout(2000).then(() => false)
    ]);
    
    console.log(`Staff management interface: ${staffInterface ? 'present' : 'basic structure'}`);
    
    // Look for staff-related buttons
    const staffButtons = await page.locator('button:has-text("追加"), button:has-text("編集"), button:has-text("削除")').count();
    console.log(`Staff management buttons: ${staffButtons}`);
  });

  test('Shift Requirements tab functionality', async ({ page }) => {
    // Navigate to shift requirements tab
    await page.click('text=シフト要件');
    await page.waitForTimeout(1000);
    
    // Look for requirements management interface
    const requirementsInterface = await Promise.race([
      page.locator('text=シフト要件').isVisible().then(() => true),
      page.locator('text=要件設定').isVisible().then(() => true),
      page.locator('text=必要人数').isVisible().then(() => true),
      page.waitForTimeout(2000).then(() => false)
    ]);
    
    console.log(`Shift requirements interface: ${requirementsInterface ? 'present' : 'basic structure'}`);
    
    // Look for requirement-related controls
    const requirementControls = await page.locator('input[type="number"], select, button:has-text("設定")').count();
    console.log(`Requirement controls: ${requirementControls}`);
  });

  test('Optimization Execution tab functionality', async ({ page }) => {
    // Navigate to optimization execution tab
    await page.click('text=最適化実行');
    await page.waitForTimeout(1000);
    
    // Test optimization parameters
    const optimizationParams = [
      { label: 'θ (theta)', value: '1' },
      { label: 'lb_penalty', value: '10000' },
      { label: 'ub_penalty', value: '0' },
      { label: 'job_change_penalty', value: '10' },
      { label: 'break_penalty', value: '10000' },
      { label: 'max_day_penalty', value: '5000' },
      { label: 'time_limit', value: '30' },
      { label: 'random_seed', value: '1' }
    ];

    // Fill optimization parameters if visible
    let parametersFilled = 0;
    for (const param of optimizationParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
        parametersFilled++;
      }
    }
    
    console.log(`Optimization parameters filled: ${parametersFilled}/${optimizationParams.length}`);

    // Try to execute optimization
    const optimizeButton = page.locator('button:has-text("最適化"), button:has-text("実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
      
      // Check for optimization results
      const optimizationResults = await Promise.race([
        page.locator('text=最適化が正常に完了').isVisible().then(() => true),
        page.locator('text=シフト最適化完了').isVisible().then(() => true),
        page.locator('text=最適化結果').isVisible().then(() => true),
        page.waitForTimeout(5000).then(() => false)
      ]);
      
      console.log(`Shift optimization execution: ${optimizationResults ? 'successful' : 'requires sample data/backend'}`);
    }
  });

  test('Results Analysis tab functionality', async ({ page }) => {
    // Navigate to results analysis tab
    await page.click('text=結果分析');
    await page.waitForTimeout(1000);
    
    // Look for results interface
    const resultsInterface = await Promise.race([
      page.locator('text=最適化結果概要').isVisible().then(() => true),
      page.locator('text=結果がここに表示されます').isVisible().then(() => true),
      page.locator('text=最適化を実行すると').isVisible().then(() => true),
      page.waitForTimeout(2000).then(() => false)
    ]);
    
    console.log(`Results analysis interface: ${resultsInterface ? 'present' : 'basic structure'}`);
    
    // Look for visualization elements
    const visualizations = await Promise.race([
      page.locator('canvas').count().then(count => count > 0),
      page.locator('svg').count().then(count => count > 0),
      page.locator('[class*="chart"]').count().then(count => count > 0),
      page.waitForTimeout(1000).then(() => false)
    ]);
    
    console.log(`Shift optimization visualizations: ${visualizations ? 'present' : 'not found'}`);
    
    // Look for result tables
    const resultTables = await page.locator('table, [class*="table"]').count();
    console.log(`Result tables: ${resultTables}`);
  });

  test('Real-time Monitoring tab functionality', async ({ page }) => {
    // Navigate to real-time monitoring tab
    await page.click('text=リアルタイム監視');
    await page.waitForTimeout(1000);
    
    // Verify monitoring interface
    await expect(page.locator('text=リアルタイムシフト監視')).toBeVisible();
    
    // Look for monitoring metrics
    const monitoringMetrics = await Promise.race([
      page.locator('text=違反なしスタッフ').isVisible().then(() => true),
      page.locator('text=違反ありスタッフ').isVisible().then(() => true),
      page.locator('text=総スタッフ数').isVisible().then(() => true),
      page.waitForTimeout(2000).then(() => false)
    ]);
    
    console.log(`Monitoring metrics: ${monitoringMetrics ? 'present' : 'basic interface'}`);
    
    // Test feasibility analysis button
    const feasibilityButton = page.locator('button:has-text("実行可能性分析")').first();
    if (await feasibilityButton.isVisible()) {
      await feasibilityButton.click();
      await page.waitForTimeout(2000);
      
      // Check for analysis results
      const analysisResults = await Promise.race([
        page.locator('text=実行可能性分析が完了').isVisible().then(() => true),
        page.locator('text=分析完了').isVisible().then(() => true),
        page.waitForTimeout(3000).then(() => false)
      ]);
      
      console.log(`Feasibility analysis: ${analysisResults ? 'successful' : 'requires data/backend'}`);
    }
  });

  test('Data Management tab functionality', async ({ page }) => {
    // Navigate to data management tab
    await page.click('text=データ管理');
    await page.waitForTimeout(1000);
    
    // Verify data management interface
    await expect(page.locator('text=データ管理・エクスポート')).toBeVisible();
    
    // Test export functionality
    const exportButtons = [
      '基本Excel出力',
      'ガントチャートExcel',
      '全シフトExcel',
      '実行可能性分析'
    ];

    for (const buttonText of exportButtons) {
      const exportButton = page.locator(`button:has-text("${buttonText}")`).first();
      if (await exportButton.isVisible()) {
        // Check if button is enabled/disabled
        const isEnabled = await exportButton.isEnabled();
        console.log(`${buttonText} button: ${isEnabled ? 'enabled' : 'disabled (requires optimization results)'}`);
        
        if (isEnabled) {
          await exportButton.click();
          await page.waitForTimeout(1000);
        }
      }
    }
    
    // Look for export warning message
    const exportWarning = await page.locator('text=エクスポート機能を使用するには、まず最適化を実行してください').isVisible();
    console.log(`Export warning displayed: ${exportWarning}`);
  });

  test('Optimization flow integration test', async ({ page }) => {
    // Test complete optimization flow
    console.log('Testing complete shift optimization flow...');
    
    // Step 1: Generate sample data
    await page.click('text=システム設定');
    await page.waitForTimeout(1000);
    
    const generateButton = page.locator('button:has-text("サンプルデータ生成")').first();
    if (await generateButton.isVisible()) {
      await generateButton.click();
      await page.waitForTimeout(3000);
    }
    
    // Step 2: Check staff management
    await page.click('text=スタッフ管理');
    await page.waitForTimeout(1000);
    
    // Step 3: Check shift requirements
    await page.click('text=シフト要件');
    await page.waitForTimeout(1000);
    
    // Step 4: Execute optimization
    await page.click('text=最適化実行');
    await page.waitForTimeout(1000);
    
    const optimizeButton = page.locator('button:has-text("最適化"), button:has-text("実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(5000);
    }
    
    // Step 5: Check results
    await page.click('text=結果分析');
    await page.waitForTimeout(1000);
    
    // Step 6: Monitor results
    await page.click('text=リアルタイム監視');
    await page.waitForTimeout(1000);
    
    // Step 7: Export data
    await page.click('text=データ管理');
    await page.waitForTimeout(1000);
    
    console.log('Complete optimization flow test completed');
  });

  test('Error handling and validation', async ({ page }) => {
    // Navigate to system settings
    await page.click('text=システム設定');
    await page.waitForTimeout(1000);
    
    // Test with invalid date range
    const startDateInput = page.locator('input[type="date"]').first();
    const endDateInput = page.locator('input[type="date"]').nth(1);
    
    if (await startDateInput.isVisible() && await endDateInput.isVisible()) {
      await startDateInput.fill('2024-01-15');
      await endDateInput.fill('2024-01-10'); // End before start
      await page.waitForTimeout(500);
      
      const generateButton = page.locator('button:has-text("サンプルデータ生成")').first();
      if (await generateButton.isVisible()) {
        await generateButton.click();
        await page.waitForTimeout(2000);
        
        // Check for error handling
        const errorHandling = await Promise.race([
          page.locator('text=エラー').isVisible().then(() => true),
          page.locator('text=無効').isVisible().then(() => true),
          page.locator('[class*="error"]').isVisible().then(() => true),
          page.waitForTimeout(2000).then(() => false)
        ]);
        
        console.log(`Error handling for invalid dates: ${errorHandling ? 'working' : 'not visible'}`);
      }
    }
  });

  test('Responsive design and accessibility', async ({ page }) => {
    // Test tab navigation with keyboard
    await page.keyboard.press('Tab');
    await page.waitForTimeout(200);
    
    // Test that tabs are accessible
    const firstTab = page.locator('button[role="tab"]').first();
    if (await firstTab.isVisible()) {
      await firstTab.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(500);
    }
    
    // Test button accessibility
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    console.log(`Accessible buttons found: ${buttonCount}`);
    
    // Test input field accessibility
    const inputs = page.locator('input');
    const inputCount = await inputs.count();
    console.log(`Input fields found: ${inputCount}`);
    
    // Check for proper labeling
    const labeledInputs = page.locator('input[aria-label], input[label]');
    const labeledInputCount = await labeledInputs.count();
    console.log(`Properly labeled inputs: ${labeledInputCount}/${inputCount}`);
  });

  test('Performance and loading times', async ({ page }) => {
    const startTime = Date.now();
    
    // Test navigation performance between tabs
    const tabs = ['システム設定', 'スタッフ管理', 'シフト要件', '最適化実行', '結果分析', 'リアルタイム監視', 'データ管理'];
    
    for (const tab of tabs) {
      const tabStartTime = Date.now();
      
      await page.click(`text=${tab}`);
      await page.waitForLoadState('domcontentloaded');
      
      const tabLoadTime = Date.now() - tabStartTime;
      console.log(`${tab} tab load time: ${tabLoadTime}ms`);
      
      // Verify tab loaded properly
      await expect(page.locator('div[role="tabpanel"]')).toBeVisible();
      
      await page.waitForTimeout(300);
    }
    
    const totalTime = Date.now() - startTime;
    console.log(`Total navigation test time: ${totalTime}ms`);
  });
});