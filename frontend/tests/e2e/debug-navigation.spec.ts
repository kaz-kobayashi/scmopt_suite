import { test, expect } from '@playwright/test';

test.describe('Debug Navigation Tests', () => {
  test('Debug inventory navigation with screenshots', async ({ page }) => {
    // Navigate to home page
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Take screenshot of home page
    await page.screenshot({ path: 'debug-home.png', fullPage: true });
    
    // Click on inventory navigation
    await page.click('text=在庫管理');
    await page.waitForTimeout(3000);
    
    // Take screenshot after navigation
    await page.screenshot({ path: 'debug-inventory.png', fullPage: true });
    
    // Log all visible text on the page for debugging
    const pageText = await page.textContent('body');
    console.log('Page text after inventory navigation:', pageText?.substring(0, 500));
    
    // Try multiple possible selectors
    const selectors = [
      'text=在庫管理',
      'text=EOQ計算機',
      'text=在庫シミュレーション',
      'text=最適化',
      'text=マルチエシェロン',
      'h1',
      'h2',
      'h3'
    ];
    
    for (const selector of selectors) {
      try {
        const element = page.locator(selector);
        const isVisible = await element.isVisible();
        const text = isVisible ? await element.textContent() : 'Not visible';
        console.log(`Selector "${selector}": ${isVisible ? 'VISIBLE' : 'NOT VISIBLE'} - Text: "${text}"`);
      } catch (error) {
        console.log(`Selector "${selector}": ERROR - ${error}`);
      }
    }
    
    // Verify inventory page loaded successfully
    await expect(page.locator('text=在庫管理')).toBeVisible({ timeout: 10000 });
  });

  test('Debug routing navigation with screenshots', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Take screenshot of home page
    await page.screenshot({ path: 'debug-home-routing.png', fullPage: true });
    
    // Click on routing navigation
    await page.click('text=配送ルーティング');
    await page.waitForTimeout(3000);
    
    // Take screenshot after navigation
    await page.screenshot({ path: 'debug-routing.png', fullPage: true });
    
    // Log page content
    const pageText = await page.textContent('body');
    console.log('Page text after routing navigation:', pageText?.substring(0, 500));
    
    // Check multiple possible routing selectors
    const routingSelectors = [
      'text=配送・輸送最適化',
      'text=CO2計算',
      'text=距離行列',
      'text=配送ルート最適化',
      'text=高度なVRP'
    ];
    
    for (const selector of routingSelectors) {
      try {
        const element = page.locator(selector);
        const isVisible = await element.isVisible();
        const text = isVisible ? await element.textContent() : 'Not visible';
        console.log(`Routing selector "${selector}": ${isVisible ? 'VISIBLE' : 'NOT VISIBLE'} - Text: "${text}"`);
      } catch (error) {
        console.log(`Routing selector "${selector}": ERROR - ${error}`);
      }
    }
    
    // Verify routing page
    await expect(page.locator('text=配送・輸送最適化')).toBeVisible({ timeout: 10000 });
  });

  test('Debug LND with actual file upload', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Navigate to LND
    await page.click('text=物流ネットワーク設計');
    await page.waitForTimeout(2000);
    
    // Navigate to Weiszfeld tab
    await page.click('text=単一施設最適立地');
    await page.waitForTimeout(1000);
    
    // Take screenshot before file upload
    await page.screenshot({ path: 'debug-lnd-before-upload.png', fullPage: true });
    
    // Upload file
    const fileInput = page.locator('input[type="file"]');
    const filePath = '/Users/kazuhiro/Documents/2509/scmopt_suite/frontend/public/sample_data/customers_standard.csv';
    await fileInput.setInputFiles(filePath);
    
    // Wait for file processing
    await page.waitForTimeout(3000);
    
    // Take screenshot after file upload
    await page.screenshot({ path: 'debug-lnd-after-upload.png', fullPage: true });
    
    // Try to click calculate button
    const calculateButton = page.locator('button:has-text("CALCULATE OPTIMAL LOCATION")');
    await expect(calculateButton).toBeVisible();
    
    // Take screenshot before clicking
    await page.screenshot({ path: 'debug-lnd-ready-to-calculate.png', fullPage: true });
    
    console.log('LND test completed successfully');
  });
});