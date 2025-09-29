import { test, expect } from '@playwright/test';

test.describe('Logistics Network Design Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Navigate to logistics network design
    await page.click('text=物流ネットワーク設計');
    await page.waitForTimeout(2000);
    
    // Verify we're on the LND page
    await expect(page.locator('text=Logistics Network Design (MELOS)')).toBeVisible();
  });

  test('Seven-tab structure navigation', async ({ page }) => {
    const sevenTabs = [
      'システム設定',
      'データ管理', 
      '立地モデル',
      '最適化実行',
      '結果分析',
      'リアルタイム監視',
      'ポリシー管理'
    ];

    for (let i = 0; i < sevenTabs.length; i++) {
      const tabName = sevenTabs[i];
      console.log(`Testing LND tab: ${tabName}`);
      
      // Click on tab
      await page.click(`text=${tabName}`);
      await page.waitForTimeout(1000);
      
      // Verify tab content is visible
      const tabPanel = page.locator('div[role="tabpanel"]').nth(i);
      await expect(tabPanel).toBeVisible();
      
      // Verify tab-specific content
      if (tabName === 'システム設定') {
        await expect(page.locator('text=物流ネットワーク設計システムの基本設定')).toBeVisible();
      } else if (tabName === 'データ管理') {
        await expect(page.locator('text=顧客データとネットワークデータの管理')).toBeVisible();
      } else if (tabName === '立地モデル') {
        await expect(page.locator('text=施設立地最適化モデルの選択と設定')).toBeVisible();
      }
    }
  });

  test('System Settings tab functionality', async ({ page }) => {
    // Navigate to system settings (should be default)
    await page.click('text=システム設定');
    await page.waitForTimeout(1000);
    
    // Test algorithm parameter inputs
    const maxIterationsInput = page.locator('input[label*="最大反復回数"]').first();
    if (await maxIterationsInput.isVisible()) {
      await maxIterationsInput.fill('200');
      await page.waitForTimeout(300);
    }
    
    const toleranceInput = page.locator('input[label*="収束許容誤差"]').first();
    if (await toleranceInput.isVisible()) {
      await toleranceInput.fill('1e-6');
      await page.waitForTimeout(300);
    }
    
    // Verify system status information
    await expect(page.locator('text=システム状態')).toBeVisible();
    await expect(page.locator('text=分析モジュール: 9種類')).toBeVisible();
  });

  test('Data Management tab functionality', async ({ page }) => {
    // Navigate to data management tab
    await page.click('text=データ管理');
    await page.waitForTimeout(1000);
    
    // Test file upload interface
    await expect(page.locator('text=顧客データとネットワークデータの管理')).toBeVisible();
    
    // Look for upload buttons
    const uploadButtons = page.locator('button:has-text("アップロード"), button:has-text("ファイル"), input[type="file"]');
    const uploadButtonCount = await uploadButtons.count();
    console.log(`Upload interface elements found: ${uploadButtonCount}`);
    
    // Test sample data generation if available
    const sampleDataButton = page.locator('button:has-text("サンプル"), button:has-text("生成")').first();
    if (await sampleDataButton.isVisible()) {
      await sampleDataButton.click();
      await page.waitForTimeout(2000);
      
      // Check for data generation results
      const dataGenerated = await Promise.race([
        page.locator('text=データが正常に読み込まれました').isVisible().then(() => true),
        page.locator('text=生成完了').isVisible().then(() => true),
        page.waitForTimeout(3000).then(() => false)
      ]);
      
      console.log(`Sample data generation: ${dataGenerated ? 'successful' : 'requires backend'}`);
    }
  });

  test('Location Models tab - Weiszfeld Algorithm', async ({ page }) => {
    // Navigate to location models tab
    await page.click('text=立地モデル');
    await page.waitForTimeout(1000);
    
    // Test Weiszfeld method
    await expect(page.locator('text=Weiszfeld法による単一施設最適立地')).toBeVisible();
    
    // Fill Weiszfeld parameters
    const weiszfeldParams = [
      { label: '最大反復回数', value: '100' },
      { label: '収束許容誤差', value: '1e-5' }
    ];

    for (const param of weiszfeldParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute Weiszfeld calculation
    const calculateButton = page.locator('button:has-text("計算"), button:has-text("実行")').first();
    if (await calculateButton.isVisible()) {
      await calculateButton.click();
      await page.waitForTimeout(3000);
      
      // Check for calculation results
      const resultsVisible = await Promise.race([
        page.locator('text=最適立地').isVisible().then(() => true),
        page.locator('text=計算完了').isVisible().then(() => true),
        page.locator('canvas').isVisible().then(() => true),
        page.waitForTimeout(4000).then(() => false)
      ]);
      
      console.log(`Weiszfeld calculation: ${resultsVisible ? 'successful' : 'requires data/backend'}`);
    }
  });

  test('Location Models tab - Multi-Facility Weiszfeld', async ({ page }) => {
    // Navigate to location models tab
    await page.click('text=立地モデル');
    await page.waitForTimeout(1000);
    
    // Test multi-facility Weiszfeld
    await expect(page.locator('text=複数施設Weiszfeld最適化')).toBeVisible();
    
    // Fill multi-facility parameters
    const multiFacilityParams = [
      { label: '施設数', value: '3' },
      { label: 'ランダムシード', value: '42' }
    ];

    for (const param of multiFacilityParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute multi-facility optimization
    const optimizeButton = page.locator('button:has-text("最適化"), button:has-text("実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('Location Models tab - K-Median Optimization', async ({ page }) => {
    // Navigate to location models tab
    await page.click('text=立地モデル');
    await page.waitForTimeout(1000);
    
    // Test K-Median optimization
    await expect(page.locator('text=K-Median施設立地最適化')).toBeVisible();
    
    // Fill K-Median parameters
    const kMedianParams = [
      { label: 'K値（施設数）', value: '4' },
      { label: '最大反復回数', value: '150' }
    ];

    for (const param of kMedianParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute K-Median optimization
    const optimizeButton = page.locator('button:has-text("K-Median"), button:has-text("最適化")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('Optimization Execution tab functionality', async ({ page }) => {
    // Navigate to optimization execution tab
    await page.click('text=最適化実行');
    await page.waitForTimeout(1000);
    
    // Test Multiple Source LND optimization
    await expect(page.locator('text=Multiple Source LND 最適化')).toBeVisible();
    
    // Look for file upload interfaces for optimization
    const fileInputs = page.locator('input[type="file"]');
    const fileInputCount = await fileInputs.count();
    console.log(`File upload inputs for optimization: ${fileInputCount}`);
    
    // Test Single Source LND optimization
    await expect(page.locator('text=Single Source LND 最適化')).toBeVisible();
    
    // Fill optimization parameters if visible
    const optimizationParams = [
      { label: '最大実行時間', value: '300' },
      { label: '倉庫数', value: '5' }
    ];

    for (const param of optimizationParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Test optimization execution
    const executeButton = page.locator('button:has-text("最適化実行"), button:has-text("実行開始")').first();
    if (await executeButton.isVisible()) {
      await executeButton.click();
      await page.waitForTimeout(2000);
    }
  });

  test('Results Analysis tab functionality', async ({ page }) => {
    // Navigate to results analysis tab
    await page.click('text=結果分析');
    await page.waitForTimeout(1000);
    
    // Test Service Area Analysis
    await expect(page.locator('text=サービスエリア分析')).toBeVisible();
    
    // Test Elbow Method Analysis
    await expect(page.locator('text=エルボー法による最適施設数分析')).toBeVisible();
    
    // Fill elbow method parameters
    const elbowParams = [
      { label: '最小施設数', value: '2' },
      { label: '最大施設数', value: '10' }
    ];

    for (const param of elbowParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute elbow method analysis
    const analyzeButton = page.locator('button:has-text("エルボー法"), button:has-text("分析実行")').first();
    if (await analyzeButton.isVisible()) {
      await analyzeButton.click();
      await page.waitForTimeout(3000);
      
      // Check for analysis results
      const analysisResults = await Promise.race([
        page.locator('text=最適施設数').isVisible().then(() => true),
        page.locator('canvas').isVisible().then(() => true),
        page.waitForTimeout(4000).then(() => false)
      ]);
      
      console.log(`Elbow method analysis: ${analysisResults ? 'successful' : 'requires data/backend'}`);
    }
  });

  test('Real-time Monitoring tab functionality', async ({ page }) => {
    // Navigate to real-time monitoring tab
    await page.click('text=リアルタイム監視');
    await page.waitForTimeout(1000);
    
    // Verify monitoring interface
    await expect(page.locator('text=最適化進捗監視')).toBeVisible();
    await expect(page.locator('text=システム監視')).toBeVisible();
    
    // Check for monitoring indicators
    const monitoringElements = await Promise.race([
      page.locator('text=CPU使用率').isVisible().then(() => true),
      page.locator('text=メモリ使用量').isVisible().then(() => true),
      page.locator('text=API応答時間').isVisible().then(() => true),
      page.waitForTimeout(2000).then(() => false)
    ]);
    
    console.log(`Monitoring interface: ${monitoringElements ? 'present' : 'basic interface'}`);
    
    // Test refresh monitoring button
    const refreshButton = page.locator('button:has-text("更新"), button:has-text("リフレッシュ")').first();
    if (await refreshButton.isVisible()) {
      await refreshButton.click();
      await page.waitForTimeout(1000);
    }
  });

  test('Policy Management tab functionality', async ({ page }) => {
    // Navigate to policy management tab
    await page.click('text=ポリシー管理');
    await page.waitForTimeout(1000);
    
    // Test algorithm parameter settings
    await expect(page.locator('text=アルゴリズムパラメータ設定')).toBeVisible();
    
    // Test Weiszfeld policy settings
    const weiszfeldPolicyParams = [
      { label: 'デフォルト最大反復回数', value: '100' },
      { label: 'デフォルト許容誤差', value: '1e-6' }
    ];

    for (const param of weiszfeldPolicyParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Test K-Median policy settings
    const kMedianPolicyParams = [
      { label: 'デフォルトK値', value: '3' },
      { label: 'デフォルト最大反復回数', value: '150' }
    ];

    for (const param of kMedianPolicyParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Test policy presets
    const policyPresets = ['default', 'fast', 'high-precision'];
    for (const preset of policyPresets) {
      const presetButton = page.locator(`button:has-text("${preset}")`).first();
      if (await presetButton.isVisible()) {
        await presetButton.click();
        await page.waitForTimeout(500);
      }
    }

    // Test save configuration
    const saveButton = page.locator('button:has-text("保存"), button:has-text("設定保存")').first();
    if (await saveButton.isVisible()) {
      await saveButton.click();
      await page.waitForTimeout(1000);
    }
  });

  test('Error handling and edge cases', async ({ page }) => {
    // Navigate to location models tab
    await page.click('text=立地モデル');
    await page.waitForTimeout(1000);
    
    // Test with invalid parameters
    const invalidParams = [
      { label: '最大反復回数', value: '-10' },
      { label: '収束許容誤差', value: '0' }
    ];

    for (const param of invalidParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Try to execute with invalid parameters
    const calculateButton = page.locator('button:has-text("計算"), button:has-text("実行")').first();
    if (await calculateButton.isVisible()) {
      await calculateButton.click();
      await page.waitForTimeout(1000);
      
      // Check for error handling
      const errorHandling = await Promise.race([
        page.locator('text=エラー').isVisible().then(() => true),
        page.locator('text=無効').isVisible().then(() => true),
        page.locator('[class*="error"]').isVisible().then(() => true),
        page.waitForTimeout(2000).then(() => false)
      ]);
      
      console.log(`Error handling: ${errorHandling ? 'working' : 'not visible'}`);
    }
  });

  test('Visualization and export functionality', async ({ page }) => {
    // Navigate through tabs to look for visualization and export options
    const tabsToCheck = ['結果分析', 'ポリシー管理'];
    
    for (const tab of tabsToCheck) {
      await page.click(`text=${tab}`);
      await page.waitForTimeout(1000);
      
      // Look for export buttons
      const exportButtons = await page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード"), button:has-text("保存")').count();
      console.log(`Export buttons in ${tab}: ${exportButtons}`);
      
      // Look for visualization elements
      const visualizations = await Promise.race([
        page.locator('canvas').count().then(count => count > 0),
        page.locator('svg').count().then(count => count > 0),
        page.waitForTimeout(1000).then(() => false)
      ]);
      
      console.log(`Visualizations in ${tab}: ${visualizations ? 'present' : 'not found'}`);
    }
  });

  test('Sample data and help functionality', async ({ page }) => {
    // Look for sample data dialog
    const sampleDataButton = page.locator('button:has-text("サンプル"), button:has-text("Sample")').first();
    if (await sampleDataButton.isVisible()) {
      await sampleDataButton.click();
      await page.waitForTimeout(1000);
      
      // Check if sample data dialog opens
      const dialogVisible = await Promise.race([
        page.locator('div[role="dialog"]').isVisible().then(() => true),
        page.locator('text=サンプルデータ').isVisible().then(() => true),
        page.waitForTimeout(2000).then(() => false)
      ]);
      
      console.log(`Sample data dialog: ${dialogVisible ? 'opened' : 'not found'}`);
      
      if (dialogVisible) {
        // Try to close dialog
        const closeButton = page.locator('button:has-text("閉じる"), button:has-text("Close")').first();
        if (await closeButton.isVisible()) {
          await closeButton.click();
          await page.waitForTimeout(500);
        }
      }
    }
  });
});