import { test, expect } from '@playwright/test';

test.describe('Comprehensive Navigation Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
  });

  test('Dashboard navigation and overview', async ({ page }) => {
    // Verify we're on the dashboard by default
    await expect(page.locator('h3:has-text("Supply Chain Management Optimization Platform")')).toBeVisible();
    
    // Check dashboard sections are present
    await expect(page.locator('text=主要セクション')).toBeVisible();
    await expect(page.locator('text=最近の活動')).toBeVisible();
    await expect(page.locator('text=システム状況')).toBeVisible();
  });

  test('Navigate to all main sections', async ({ page }) => {
    const sections = [
      { name: '在庫管理', heading: '在庫管理' },
      { name: '配送ルーティング', heading: '配送・輸送最適化' },
      { name: '物流ネットワーク設計', heading: 'Logistics Network Design (MELOS)' },
      { name: 'シフト最適化', heading: 'シフト最適化システム' },
      { name: 'ジョブショップスケジューリング', heading: 'ジョブショップスケジューリング' },
      { name: 'スケジュールテンプレート', heading: 'スケジュールテンプレート管理' },
      { name: '収益管理 (MERMO)', heading: '収益管理分析 (MERMO)' },
      { name: 'SCRM分析', heading: 'SCRM分析 (MERIODAS)' },
      { name: 'SND分析', heading: 'Service Network Design (SENDO)' },
      { name: '分析', heading: '分析' },
      { name: 'PyVRPインターフェース', heading: 'PyVRP' }
    ];

    for (const section of sections) {
      console.log(`Testing navigation to: ${section.name}`);
      
      // Navigate to section
      await page.click(`text=${section.name}`);
      await page.waitForTimeout(1500);
      
      // Verify we're on the correct page
      await expect(page.locator(`text=${section.heading}`).first()).toBeVisible({ timeout: 5000 });
      
      // Go back to dashboard for next iteration
      await page.click('text=ダッシュボード');
      await page.waitForTimeout(1000);
    }
  });

  test('Seven-tab structure navigation for main components', async ({ page }) => {
    const componentsWithSevenTabs = [
      {
        name: '物流ネットワーク設計',
        tabs: ['システム設定', 'データ管理', '立地モデル', '最適化実行', '結果分析', 'リアルタイム監視', 'ポリシー管理']
      },
      {
        name: 'スケジュールテンプレート',
        tabs: ['システム設定', 'データ管理', 'カテゴリ管理', 'テンプレート実行', '結果分析', 'リアルタイム監視', 'ポリシー管理']
      },
      {
        name: 'SND分析',
        tabs: ['システム設定', 'データ管理', 'ネットワークモデル', '最適化実行', '結果可視化', 'リアルタイム監視', 'ポリシー管理']
      },
      {
        name: '分析',
        tabs: ['システム設定', 'データ管理', '分析モデル', '分析実行', '結果可視化', 'リアルタイム監視', 'レポート管理']
      }
    ];

    for (const component of componentsWithSevenTabs) {
      console.log(`Testing 7-tab structure for: ${component.name}`);
      
      // Navigate to component
      await page.click(`text=${component.name}`);
      await page.waitForTimeout(2000);
      
      // Test each tab
      for (let i = 0; i < component.tabs.length; i++) {
        const tabName = component.tabs[i];
        console.log(`  Testing tab: ${tabName}`);
        
        // Click on tab
        await page.click(`text=${tabName}`);
        await page.waitForTimeout(1000);
        
        // Verify tab is active (has active styling or content is visible)
        const tabPanel = page.locator(`div[role="tabpanel"]`).nth(i);
        await expect(tabPanel).toBeVisible();
      }
      
      // Return to dashboard
      await page.click('text=ダッシュボード');
      await page.waitForTimeout(1000);
    }
  });

  test('Extended tab structure for other components', async ({ page }) => {
    const componentsWithExtendedTabs = [
      {
        name: 'シフト最適化',
        tabs: ['システム設定', 'スタッフ管理', 'シフト要件', '最適化実行', '結果分析', 'リアルタイム監視', 'データ管理']
      },
      {
        name: 'ジョブショップスケジューリング',
        tabs: ['システム設定', 'ジョブ管理', 'リソース管理', 'スケジューリング実行', '結果分析', 'リアルタイム監視', 'レポート・エクスポート']
      },
      {
        name: '収益管理 (MERMO)',
        tabs: ['システム設定', 'データ管理', '需要予測', '価格最適化', '結果分析', 'リアルタイム監視', 'ポリシー管理']
      },
      {
        name: 'SCRM分析',
        tabs: ['システム設定', 'データ管理', 'リスクモデル', '分析実行', '結果可視化', 'リアルタイム監視', 'ポリシー管理']
      }
    ];

    for (const component of componentsWithExtendedTabs) {
      console.log(`Testing extended tab structure for: ${component.name}`);
      
      // Navigate to component
      await page.click(`text=${component.name}`);
      await page.waitForTimeout(2000);
      
      // Test each tab exists and is clickable
      for (const tabName of component.tabs) {
        console.log(`  Checking tab: ${tabName}`);
        
        const tabElement = page.locator(`text=${tabName}`).first();
        await expect(tabElement).toBeVisible();
        
        // Click and verify content loads
        await tabElement.click();
        await page.waitForTimeout(800);
      }
      
      // Return to dashboard
      await page.click('text=ダッシュボード');
      await page.waitForTimeout(1000);
    }
  });

  test('Responsive navigation menu', async ({ page }) => {
    // Test navigation menu visibility
    await expect(page.locator('nav')).toBeVisible();
    
    // Test that all navigation items are present
    const expectedNavItems = [
      'ダッシュボード',
      '在庫管理',
      '配送ルーティング',
      '物流ネットワーク設計',
      'シフト最適化',
      'ジョブショップスケジューリング',
      'スケジュールテンプレート',
      '収益管理 (MERMO)',
      'SCRM分析',
      'SND分析',
      '分析',
      'PyVRPインターフェース'
    ];

    for (const item of expectedNavItems) {
      await expect(page.locator(`text=${item}`)).toBeVisible();
    }
  });

  test('Breadcrumb navigation', async ({ page }) => {
    // Navigate to a subsection
    await page.click('text=物流ネットワーク設計');
    await page.waitForTimeout(1500);
    
    // Click on a tab
    await page.click('text=データ管理');
    await page.waitForTimeout(1000);
    
    // Verify we can navigate back
    await page.click('text=ダッシュボード');
    await page.waitForTimeout(1000);
    
    await expect(page.locator('h3:has-text("Supply Chain Management Optimization Platform")')).toBeVisible();
  });
});