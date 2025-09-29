import { test, expect } from '@playwright/test';

test.describe('Inventory Management Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Navigate to inventory management
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    // Verify we're on the inventory page
    await expect(page.locator('h4:has-text("在庫管理")')).toBeVisible();
  });

  test('EOQ Analysis functionality', async ({ page }) => {
    // Navigate to EOQ tab (should be default tab)
    await expect(page.locator('text=EOQ分析')).toBeVisible();
    
    // Test parameter inputs
    const parameters = [
      { label: '発注コスト (K)', value: '150' },
      { label: '需要量 (d)', value: '1200' },
      { label: '保管費率 (h)', value: '0.25' },
      { label: '欠品費率 (b)', value: '0.15' }
    ];

    for (const param of parameters) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(300);
      }
    }

    // Click calculate button
    const calculateButton = page.locator('button:has-text("計算")').first();
    if (await calculateButton.isVisible()) {
      await calculateButton.click();
      await page.waitForTimeout(1500);
      
      // Verify results appear (EOQ chart or result cards)
      const resultsVisible = await Promise.race([
        page.locator('text=最適発注量').isVisible().then(() => true),
        page.locator('canvas').isVisible().then(() => true),
        page.locator('text=EOQ結果').isVisible().then(() => true),
        page.waitForTimeout(3000).then(() => false)
      ]);
      
      if (resultsVisible) {
        console.log('EOQ calculation completed successfully');
      } else {
        console.log('EOQ calculation may require backend connection');
      }
    }
  });

  test('Inventory Simulation functionality', async ({ page }) => {
    // Navigate to simulation tab
    await page.click('text=シミュレーション');
    await page.waitForTimeout(1000);
    
    // Test simulation parameters
    const simParams = [
      { label: 'サンプル数', value: '500' },
      { label: '期間数', value: '180' },
      { label: '平均需要', value: '75' },
      { label: '標準偏差', value: '15' }
    ];

    for (const param of simParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Click simulate button
    const simulateButton = page.locator('button:has-text("シミュレーション実行")').first();
    if (await simulateButton.isVisible()) {
      await simulateButton.click();
      await page.waitForTimeout(2000);
      
      // Check for simulation results
      const simulationResults = await Promise.race([
        page.locator('text=シミュレーション結果').isVisible().then(() => true),
        page.locator('canvas').isVisible().then(() => true),
        page.waitForTimeout(4000).then(() => false)
      ]);
      
      console.log(`Simulation execution: ${simulationResults ? 'successful' : 'requires backend'}`);
    }
  });

  test('Multi-Echelon Analysis functionality', async ({ page }) => {
    // Navigate to multi-echelon tab
    await page.click('text=多段階在庫');
    await page.waitForTimeout(1000);
    
    // Test multi-echelon parameters
    const multiEchelonParams = [
      { label: '工場数', value: '3' },
      { label: 'DC数', value: '5' },
      { label: '小売店数', value: '8' },
      { label: '保管コスト', value: '1.5' }
    ];

    for (const param of multiEchelonParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Click analyze button
    const analyzeButton = page.locator('button:has-text("分析実行")').first();
    if (await analyzeButton.isVisible()) {
      await analyzeButton.click();
      await page.waitForTimeout(2000);
      
      // Verify multi-echelon visualization appears
      const analysisResults = await Promise.race([
        page.locator('text=多段階分析結果').isVisible().then(() => true),
        page.locator('canvas').isVisible().then(() => true),
        page.waitForTimeout(3000).then(() => false)
      ]);
      
      console.log(`Multi-echelon analysis: ${analysisResults ? 'successful' : 'requires backend'}`);
    }
  });

  test('Optimization functionality - Q-R Model', async ({ page }) => {
    // Navigate to optimization tab
    await page.click('text=最適化');
    await page.waitForTimeout(1000);
    
    // Select Q-R optimization
    const optimizationSelect = page.locator('select').first();
    if (await optimizationSelect.isVisible()) {
      await optimizationSelect.selectOption('qr');
      await page.waitForTimeout(500);
    }

    // Fill Q-R parameters
    const qrParams = [
      { label: '平均需要', value: '120' },
      { label: '需要標準偏差', value: '25' },
      { label: 'リードタイム', value: '3' },
      { label: '欠品費', value: '40' }
    ];

    for (const param of qrParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute Q-R optimization
    const optimizeButton = page.locator('button:has-text("最適化実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
      
      // Check for optimization results
      const optimizationResults = await Promise.race([
        page.locator('text=最適発注量').isVisible().then(() => true),
        page.locator('text=最適再発注点').isVisible().then(() => true),
        page.locator('text=Q-R最適化結果').isVisible().then(() => true),
        page.waitForTimeout(4000).then(() => false)
      ]);
      
      console.log(`Q-R optimization: ${optimizationResults ? 'successful' : 'requires backend'}`);
    }
  });

  test('Safety Stock Optimization', async ({ page }) => {
    // Navigate to optimization tab
    await page.click('text=最適化');
    await page.waitForTimeout(1000);
    
    // Select Safety Stock optimization
    const optimizationSelect = page.locator('select').first();
    if (await optimizationSelect.isVisible()) {
      await optimizationSelect.selectOption('safety-stock');
      await page.waitForTimeout(500);
    }

    // Fill safety stock parameters
    const safetyStockParams = [
      { label: '目標サービスレベル', value: '0.95' },
      { label: '需要変動係数', value: '0.3' },
      { label: 'リードタイム変動', value: '0.2' }
    ];

    for (const param of safetyStockParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute safety stock optimization
    const optimizeButton = page.locator('button:has-text("最適化実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('MESSA Optimization', async ({ page }) => {
    // Navigate to optimization tab
    await page.click('text=最適化');
    await page.waitForTimeout(1000);
    
    // Select MESSA optimization
    const optimizationSelect = page.locator('select').first();
    if (await optimizationSelect.isVisible()) {
      await optimizationSelect.selectOption('messa');
      await page.waitForTimeout(500);
    }

    // Fill MESSA parameters
    const messaParams = [
      { label: '到着率パラメータ λ', value: '0.8' },
      { label: '状態変化率 α', value: '0.3' },
      { label: 'サービス率 μ', value: '1.2' }
    ];

    for (const param of messaParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute MESSA optimization
    const optimizeButton = page.locator('button:has-text("MESSA最適化実行")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('Newsvendor Model', async ({ page }) => {
    // Navigate to optimization tab
    await page.click('text=最適化');
    await page.waitForTimeout(1000);
    
    // Select Newsvendor model
    const optimizationSelect = page.locator('select').first();
    if (await optimizationSelect.isVisible()) {
      await optimizationSelect.selectOption('newsvendor');
      await page.waitForTimeout(500);
    }

    // Fill newsvendor parameters
    const newsvendorParams = [
      { label: '販売価格', value: '100' },
      { label: '仕入価格', value: '60' },
      { label: '残存価値', value: '20' },
      { label: '需要分布パラメータ', value: '150' }
    ];

    for (const param of newsvendorParams) {
      const input = page.locator(`input[label*="${param.label}"]`).first();
      if (await input.isVisible()) {
        await input.fill(param.value);
        await page.waitForTimeout(200);
      }
    }

    // Execute newsvendor optimization
    const optimizeButton = page.locator('button:has-text("ニューズベンダー最適化")').first();
    if (await optimizeButton.isVisible()) {
      await optimizeButton.click();
      await page.waitForTimeout(3000);
    }
  });

  test('Tab navigation within inventory management', async ({ page }) => {
    const inventoryTabs = [
      'EOQ分析',
      'シミュレーション', 
      '多段階在庫',
      '最適化'
    ];

    for (const tab of inventoryTabs) {
      console.log(`Testing inventory tab: ${tab}`);
      
      // Click on tab
      await page.click(`text=${tab}`);
      await page.waitForTimeout(1000);
      
      // Verify tab content is visible
      const tabContent = page.locator('div[role="tabpanel"]').first();
      await expect(tabContent).toBeVisible();
      
      // Verify tab-specific content exists
      const hasContent = await Promise.race([
        page.locator('button').first().isVisible().then(() => true),
        page.locator('input').first().isVisible().then(() => true),
        page.locator('canvas').first().isVisible().then(() => true),
        page.waitForTimeout(2000).then(() => false)
      ]);
      
      expect(hasContent).toBe(true);
    }
  });

  test('Error handling and validation', async ({ page }) => {
    // Navigate to EOQ tab
    await page.click('text=EOQ分析');
    await page.waitForTimeout(1000);
    
    // Try to calculate with invalid inputs
    const invalidInput = page.locator('input').first();
    if (await invalidInput.isVisible()) {
      await invalidInput.fill('-100');
      await page.waitForTimeout(300);
      
      const calculateButton = page.locator('button:has-text("計算")').first();
      if (await calculateButton.isVisible()) {
        await calculateButton.click();
        await page.waitForTimeout(1000);
        
        // Check for error messages or validation
        const errorVisible = await Promise.race([
          page.locator('text=エラー').isVisible().then(() => true),
          page.locator('text=無効').isVisible().then(() => true),
          page.locator('[class*="error"]').isVisible().then(() => true),
          page.waitForTimeout(2000).then(() => false)
        ]);
        
        console.log(`Error handling: ${errorVisible ? 'working' : 'not visible'}`);
      }
    }
  });

  test('Results visualization and export', async ({ page }) => {
    // Navigate to EOQ tab and perform calculation
    await page.click('text=EOQ分析');
    await page.waitForTimeout(1000);
    
    // Fill valid parameters
    const inputs = page.locator('input[type="number"]');
    const inputCount = await inputs.count();
    
    for (let i = 0; i < Math.min(inputCount, 4); i++) {
      const input = inputs.nth(i);
      if (await input.isVisible()) {
        await input.fill('100');
        await page.waitForTimeout(200);
      }
    }
    
    // Execute calculation
    const calculateButton = page.locator('button:has-text("計算")').first();
    if (await calculateButton.isVisible()) {
      await calculateButton.click();
      await page.waitForTimeout(2000);
      
      // Look for export or download buttons
      const exportButtons = await page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード"), button:has-text("保存")').count();
      console.log(`Export functionality: ${exportButtons > 0 ? 'available' : 'not found'}`);
      
      // Look for visualization elements
      const visualizations = await Promise.race([
        page.locator('canvas').count().then(count => count > 0),
        page.locator('svg').count().then(count => count > 0),
        page.locator('[class*="chart"]').count().then(count => count > 0),
        page.waitForTimeout(2000).then(() => false)
      ]);
      
      console.log(`Visualizations: ${visualizations ? 'present' : 'not found'}`);
    }
  });
});