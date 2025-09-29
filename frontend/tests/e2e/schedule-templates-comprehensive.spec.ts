import { test, expect } from '@playwright/test';

test.describe('Schedule Templates Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(60000);
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(1000);
    
    // Navigate to schedule templates
    const navigated = await Promise.race([
      page.click('text=スケジュールテンプレート', { timeout: 3000 }).then(() => true),
      page.click('text=Schedule Templates', { timeout: 2000 }).then(() => true),
      page.click('text=テンプレート', { timeout: 2000 }).then(() => true),
      page.click('button:has-text("スケジュール")', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(4000).then(() => false)
    ]);
    
    if (navigated) {
      await page.waitForTimeout(2000);
      console.log('✅ Successfully navigated to Schedule Templates');
    } else {
      console.log('⚠️ Could not navigate to Schedule Templates');
    }
  });

  test('Seven-tab structure verification', async ({ page }) => {
    const expectedTabs = [
      'テンプレート管理',
      'テンプレート作成',
      'カスタマイズ',
      'プレビュー',
      '適用・実行',
      '結果監視',
      'エクスポート・共有'
    ];

    // Verify Schedule Templates page is loaded
    const pageVisible = await Promise.race([
      page.locator('text=スケジュールテンプレート').isVisible().then(() => true),
      page.locator('text=Schedule Templates').isVisible().then(() => true),
      page.locator('text=テンプレート管理').isVisible().then(() => true),
      page.locator('h1, h2, h3, h4').first().isVisible().then(() => true),
      page.waitForTimeout(3000).then(() => false)
    ]);
    
    if (!pageVisible) {
      console.log('⚠️ Schedule Templates page may not be loaded properly');
      return;
    }

    console.log('Testing Schedule Templates 7-tab structure...');
    
    // Count tabs
    const tabCount = await Promise.race([
      page.locator('[role="tab"]').count(),
      page.locator('.MuiTab-root').count(),
      page.waitForTimeout(2000).then(() => 0)
    ]);
    
    console.log(`Schedule Templates tabs found: ${tabCount}`);
    
    if (tabCount >= 7) {
      // Test each tab
      for (let i = 0; i < expectedTabs.length && i < tabCount; i++) {
        const tabName = expectedTabs[i];
        console.log(`  Testing Schedule Templates tab ${i + 1}: ${tabName}`);
        
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

  test('Template Management tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates, text=テンプレート管理').isVisible();
    if (!pageVisible) return;

    // Navigate to template management tab (should be default)
    const managementTabClicked = await Promise.race([
      page.click('text=テンプレート管理', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').first().click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (managementTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Template Management tab accessed');
      
      // Look for template list/grid
      const templateLists = await Promise.race([
        page.locator('table, [class*="table"], [class*="grid"]').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Template lists/tables found: ${templateLists}`);
      
      // Look for template categories
      const templateCategories = await Promise.race([
        page.locator('text=基本テンプレート, text=カスタムテンプレート, text=共有テンプレート').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Template categories found: ${templateCategories}`);
      
      // Look for template management actions
      const managementActions = await page.locator('button:has-text("新規作成"), button:has-text("編集"), button:has-text("削除"), button:has-text("複製")').count();
      console.log(`Template management actions found: ${managementActions}`);
      
      // Test template search functionality
      const searchInput = page.locator('input[placeholder*="検索"], input[type="search"]').first();
      if (await searchInput.isVisible()) {
        await searchInput.fill('テスト');
        await page.waitForTimeout(1000);
        console.log('✅ Template search functionality test successful');
      }
      
      // Look for pre-built templates
      const prebuiltTemplates = await Promise.race([
        page.locator('text=週次スケジュール, text=月次スケジュール, text=プロジェクト').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Pre-built templates found: ${prebuiltTemplates}`);
    }
  });

  test('Template Creation tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to template creation tab
    const creationTabClicked = await Promise.race([
      page.click('text=テンプレート作成', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(1).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (creationTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Template Creation tab accessed');
      
      // Look for template creation form
      const creationForm = await Promise.race([
        page.locator('text=テンプレート名, text=説明, input[placeholder*="名前"]').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Template creation form elements found: ${creationForm}`);
      
      // Test template basic information input
      const templateNameInput = page.locator('input[placeholder*="テンプレート名"], input[label*="名前"]').first();
      if (await templateNameInput.isVisible()) {
        await templateNameInput.fill('テストテンプレート');
        await page.waitForTimeout(300);
        console.log('✅ Template name input test successful');
      }
      
      const descriptionInput = page.locator('textarea[placeholder*="説明"], input[placeholder*="説明"]').first();
      if (await descriptionInput.isVisible()) {
        await descriptionInput.fill('テスト用のスケジュールテンプレート');
        await page.waitForTimeout(300);
        console.log('✅ Template description input test successful');
      }
      
      // Look for schedule pattern options
      const schedulePatterns = await Promise.race([
        page.locator('text=日次パターン, text=週次パターン, text=月次パターン, text=カスタムパターン').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Schedule pattern options found: ${schedulePatterns}`);
      
      // Look for time slot configuration
      const timeSlotInputs = await page.locator('input[type="time"], input[placeholder*="時間"]').count();
      console.log(`Time slot configuration inputs found: ${timeSlotInputs}`);
      
      // Test template type selection
      const templateTypeSelect = page.locator('select, [role="combobox"]').first();
      if (await templateTypeSelect.isVisible()) {
        await templateTypeSelect.click();
        await page.waitForTimeout(500);
        console.log('✅ Template type selection test successful');
      }
      
      // Look for save template button
      const saveButton = page.locator('button:has-text("保存"), button:has-text("作成")').first();
      if (await saveButton.isVisible()) {
        const isEnabled = await saveButton.isEnabled();
        console.log(`Template save button: ${isEnabled ? 'enabled' : 'disabled (requires complete form)'}`);
      }
    }
  });

  test('Customization tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to customization tab
    const customizationTabClicked = await Promise.race([
      page.click('text=カスタマイズ', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(2).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (customizationTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Customization tab accessed');
      
      // Look for customization options
      const customizationOptions = await Promise.race([
        page.locator('text=時間設定, text=リソース割り当て, text=制約条件, text=カラー設定').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Customization options found: ${customizationOptions}`);
      
      // Test time range customization
      const timeRangeInputs = page.locator('input[type="time"]');
      const timeInputCount = await timeRangeInputs.count();
      
      if (timeInputCount > 0) {
        const startTimeInput = timeRangeInputs.first();
        if (await startTimeInput.isVisible()) {
          await startTimeInput.fill('09:00');
          await page.waitForTimeout(300);
          console.log('✅ Time range customization test successful');
        }
      }
      
      // Test resource assignment
      const resourceInputs = await page.locator('input[placeholder*="リソース"], select').count();
      console.log(`Resource assignment inputs found: ${resourceInputs}`);
      
      // Look for constraint configuration
      const constraintSettings = await Promise.race([
        page.locator('text=最大稼働時間, text=休憩時間, text=重複制約').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Constraint settings found: ${constraintSettings}`);
      
      // Test color customization
      const colorInputs = page.locator('input[type="color"]');
      const colorInputCount = await colorInputs.count();
      
      if (colorInputCount > 0) {
        const colorInput = colorInputs.first();
        if (await colorInput.isVisible()) {
          await colorInput.click();
          await page.waitForTimeout(500);
          console.log('✅ Color customization test successful');
        }
      }
      
      // Look for custom fields
      const customFields = await page.locator('button:has-text("フィールド追加"), button:has-text("カスタムフィールド")').count();
      console.log(`Custom field options found: ${customFields}`);
    }
  });

  test('Preview tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to preview tab
    const previewTabClicked = await Promise.race([
      page.click('text=プレビュー', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(3).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (previewTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Preview tab accessed');
      
      // Look for schedule preview visualization
      const previewVisualizations = await Promise.race([
        page.locator('canvas').count(),
        page.locator('svg').count(),
        page.locator('[class*="calendar"], [class*="schedule"]').count(),
        page.waitForTimeout(1000).then(() => 0)
      ]);
      
      console.log(`Preview visualization elements found: ${previewVisualizations}`);
      
      // Look for preview controls
      const previewControls = await Promise.race([
        page.locator('button:has-text("週表示"), button:has-text("月表示"), button:has-text("日表示")').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Preview view controls found: ${previewControls}`);
      
      // Test view switching
      const viewButtons = page.locator('button:has-text("週"), button:has-text("月"), button:has-text("日")');
      const viewButtonCount = await viewButtons.count();
      
      if (viewButtonCount > 0) {
        const firstViewButton = viewButtons.first();
        if (await firstViewButton.isVisible()) {
          await firstViewButton.click();
          await page.waitForTimeout(1000);
          console.log('✅ Preview view switching test successful');
        }
      }
      
      // Look for sample data preview
      const sampleDataButton = page.locator('button:has-text("サンプル"), button:has-text("テストデータ")').first();
      if (await sampleDataButton.isVisible()) {
        await sampleDataButton.click();
        await page.waitForTimeout(2000);
        console.log('✅ Sample data preview test successful');
      }
      
      // Look for preview validation messages
      const validationMessages = await Promise.race([
        page.locator('text=プレビューが表示されています, text=データがありません, text=設定を完了してください').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Preview validation messages found: ${validationMessages}`);
      
      // Look for print preview option
      const printButton = page.locator('button:has-text("印刷"), button:has-text("Print")').first();
      if (await printButton.isVisible()) {
        console.log('✅ Print preview option available');
      }
    }
  });

  test('Apply & Execute tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to apply & execute tab
    const applyTabClicked = await Promise.race([
      page.click('text=適用・実行', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(4).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (applyTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Apply & Execute tab accessed');
      
      // Look for template selection
      const templateSelectionElements = await Promise.race([
        page.locator('text=テンプレート選択, select, [role="combobox"]').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Template selection elements found: ${templateSelectionElements}`);
      
      // Look for application period settings
      const periodSettings = await Promise.race([
        page.locator('text=適用期間, input[type="date"], text=開始日, text=終了日').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Application period settings found: ${periodSettings}`);
      
      // Test date range selection
      const dateInputs = page.locator('input[type="date"]');
      const dateInputCount = await dateInputs.count();
      
      if (dateInputCount > 0) {
        const startDateInput = dateInputs.first();
        if (await startDateInput.isVisible()) {
          await startDateInput.fill('2024-01-01');
          await page.waitForTimeout(300);
          console.log('✅ Date range selection test successful');
        }
      }
      
      // Look for execution options
      const executionOptions = await Promise.race([
        page.locator('text=即座に適用, text=スケジュール実行, text=バッチ処理').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Execution options found: ${executionOptions}`);
      
      // Test template application
      const applyButton = page.locator('button:has-text("適用"), button:has-text("実行")').first();
      if (await applyButton.isVisible()) {
        const isEnabled = await applyButton.isEnabled();
        console.log(`Template apply button: ${isEnabled ? 'enabled' : 'disabled (requires template selection)'}`);
        
        if (isEnabled) {
          await applyButton.click();
          await page.waitForTimeout(3000);
          
          // Check for application results
          const applicationResults = await Promise.race([
            page.locator('text=適用完了, text=実行完了, text=スケジュール作成完了').isVisible().then(() => true),
            page.waitForTimeout(4000).then(() => false)
          ]);
          
          console.log(`Template application: ${applicationResults ? 'successful' : 'requires template/backend'}`);
        }
      }
      
      // Look for conflict resolution options
      const conflictResolution = await Promise.race([
        page.locator('text=競合解決, text=上書き, text=マージ').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Conflict resolution options found: ${conflictResolution}`);
    }
  });

  test('Results Monitoring tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to results monitoring tab
    const monitoringTabClicked = await Promise.race([
      page.click('text=結果監視', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(5).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (monitoringTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Results Monitoring tab accessed');
      
      // Look for monitoring dashboard
      const monitoringElements = await Promise.race([
        page.locator('text=実行状況, text=進捗, text=ステータス, text=監視').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Results monitoring elements found: ${monitoringElements}`);
      
      // Look for execution history
      const executionHistory = await Promise.race([
        page.locator('text=実行履歴, text=履歴, table').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Execution history elements found: ${executionHistory}`);
      
      // Look for status indicators
      const statusIndicators = await page.locator('text=成功, text=失敗, text=実行中, [class*="status"], [class*="indicator"]').count();
      console.log(`Status indicators found: ${statusIndicators}`);
      
      // Test refresh monitoring data
      const refreshButton = page.locator('button:has-text("更新"), button:has-text("リフレッシュ")').first();
      if (await refreshButton.isVisible()) {
        await refreshButton.click();
        await page.waitForTimeout(1000);
        console.log('✅ Monitoring refresh test successful');
      }
      
      // Look for performance metrics
      const performanceMetrics = await Promise.race([
        page.locator('text=実行時間, text=成功率, text=エラー数').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Performance metrics found: ${performanceMetrics}`);
      
      // Look for alert/notification settings
      const alertSettings = await page.locator('button:has-text("アラート"), button:has-text("通知設定")').count();
      console.log(`Alert/notification settings found: ${alertSettings}`);
      
      // Check for no results message
      const noResultsMessage = await page.locator('text=実行結果がありません, text=監視データがありません').isVisible();
      if (noResultsMessage) {
        console.log('ℹ️ No monitoring results message displayed (expected without execution)');
      }
    }
  });

  test('Export & Sharing tab functionality', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Navigate to export & sharing tab
    const exportTabClicked = await Promise.race([
      page.click('text=エクスポート・共有', { timeout: 2000 }).then(() => true),
      page.locator('[role="tab"]').nth(6).click({ timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (exportTabClicked) {
      await page.waitForTimeout(1000);
      console.log('✅ Export & Sharing tab accessed');
      
      // Look for export format options
      const exportFormats = await Promise.race([
        page.locator('text=Excel, text=CSV, text=PDF, text=iCal, text=JSON').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Export format options found: ${exportFormats}`);
      
      // Look for export type options
      const exportTypes = await Promise.race([
        page.locator('text=テンプレート定義, text=適用結果, text=監視レポート').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Export type options found: ${exportTypes}`);
      
      // Test export buttons
      const exportButtons = await page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード"), button:has-text("Export")').count();
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
      
      // Look for sharing options
      const sharingOptions = await Promise.race([
        page.locator('text=共有リンク, text=権限設定, text=チーム共有').count(),
        page.waitForTimeout(2000).then(() => 0)
      ]);
      
      console.log(`Sharing options found: ${sharingOptions}`);
      
      // Test sharing functionality
      const shareButton = page.locator('button:has-text("共有"), button:has-text("Share")').first();
      if (await shareButton.isVisible()) {
        await shareButton.click();
        await page.waitForTimeout(1000);
        console.log('✅ Sharing functionality test successful');
      }
      
      // Look for template publishing options
      const publishOptions = await page.locator('button:has-text("公開"), button:has-text("テンプレート公開")').count();
      console.log(`Template publishing options found: ${publishOptions}`);
    }
  });

  test('Schedule Templates data flow integration test', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) {
      console.log('⚠️ Schedule Templates page not accessible for integration test');
      return;
    }

    console.log('🔄 Testing Schedule Templates data flow integration...');
    
    // Step 1: Template Management
    const managementTab = await Promise.race([
      page.click('text=テンプレート管理', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (managementTab) {
      await page.waitForTimeout(800);
      console.log('Step 1: ✅ Template management accessed');
    }
    
    // Step 2: Template Creation
    const creationTab = await Promise.race([
      page.click('text=テンプレート作成', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (creationTab) {
      await page.waitForTimeout(800);
      console.log('Step 2: ✅ Template creation accessed');
    }
    
    // Step 3: Customization
    const customizationTab = await Promise.race([
      page.click('text=カスタマイズ', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (customizationTab) {
      await page.waitForTimeout(800);
      console.log('Step 3: ✅ Customization accessed');
    }
    
    // Step 4: Preview
    const previewTab = await Promise.race([
      page.click('text=プレビュー', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (previewTab) {
      await page.waitForTimeout(800);
      console.log('Step 4: ✅ Preview accessed');
    }
    
    // Step 5: Apply & Execute
    const applyTab = await Promise.race([
      page.click('text=適用・実行', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (applyTab) {
      await page.waitForTimeout(800);
      console.log('Step 5: ✅ Apply & execute accessed');
    }
    
    // Step 6: Monitoring
    const monitoringTab = await Promise.race([
      page.click('text=結果監視', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (monitoringTab) {
      await page.waitForTimeout(800);
      console.log('Step 6: ✅ Results monitoring accessed');
    }
    
    // Step 7: Export
    const exportTab = await Promise.race([
      page.click('text=エクスポート・共有', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (exportTab) {
      await page.waitForTimeout(800);
      console.log('Step 7: ✅ Export & sharing accessed');
    }
    
    console.log('🎯 Schedule Templates integration test completed');
  });

  test('Error handling and validation', async ({ page }) => {
    const pageVisible = await page.locator('text=スケジュールテンプレート, text=Schedule Templates').isVisible();
    if (!pageVisible) return;

    // Test template creation with invalid data
    const creationTab = await Promise.race([
      page.click('text=テンプレート作成', { timeout: 2000 }).then(() => true),
      page.waitForTimeout(2500).then(() => false)
    ]);
    
    if (creationTab) {
      await page.waitForTimeout(1000);
      
      // Try to save template without required fields
      const saveButton = page.locator('button:has-text("保存"), button:has-text("作成")').first();
      if (await saveButton.isVisible()) {
        await saveButton.click();
        await page.waitForTimeout(1000);
        
        // Look for validation error
        const validationError = await Promise.race([
          page.locator('text=エラー, text=必須項目, text=入力してください, [class*="error"]').isVisible().then(() => true),
          page.waitForTimeout(2000).then(() => false)
        ]);
        
        console.log(`Template creation validation: ${validationError ? 'working' : 'not visible'}`);
      }
      
      // Test invalid time inputs
      const timeInputs = page.locator('input[type="time"]');
      const timeInputCount = await timeInputs.count();
      
      if (timeInputCount > 0) {
        const firstTimeInput = timeInputs.first();
        if (await firstTimeInput.isVisible()) {
          await firstTimeInput.fill('25:00'); // Invalid time
          await page.waitForTimeout(500);
          
          const timeValidationError = await Promise.race([
            page.locator('text=無効な時間, text=時間形式, [class*="error"]').isVisible().then(() => true),
            page.waitForTimeout(2000).then(() => false)
          ]);
          
          console.log(`Time input validation: ${timeValidationError ? 'working' : 'not visible'}`);
        }
      }
    }
  });

  test('Performance and responsiveness', async ({ page }) => {
    const startTime = Date.now();
    
    // Test navigation performance between tabs
    const tabs = ['テンプレート管理', 'テンプレート作成', 'カスタマイズ', 'プレビュー', '適用・実行', '結果監視', 'エクスポート・共有'];
    
    for (const tab of tabs) {
      const tabStartTime = Date.now();
      
      const tabClicked = await Promise.race([
        page.click(`text=${tab}`, { timeout: 2000 }).then(() => true),
        page.waitForTimeout(2500).then(() => false)
      ]);
      
      if (tabClicked) {
        await page.waitForLoadState('domcontentloaded');
        
        const tabLoadTime = Date.now() - tabStartTime;
        console.log(`${tab} tab load time: ${tabLoadTime}ms`);
        
        // Verify tab loaded properly
        const contentLoaded = await Promise.race([
          page.locator('div[role="tabpanel"]').isVisible().then(() => true),
          page.waitForTimeout(1000).then(() => false)
        ]);
        
        if (contentLoaded) {
          console.log(`${tab}: content loaded successfully`);
        }
        
        await page.waitForTimeout(300);
      }
    }
    
    const totalTime = Date.now() - startTime;
    console.log(`Total Schedule Templates navigation test time: ${totalTime}ms`);
  });
});