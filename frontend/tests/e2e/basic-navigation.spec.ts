import { test, expect } from '@playwright/test';

test.describe('Basic Navigation Tests', () => {
  test('Can navigate to LND section and perform basic interaction', async ({ page }) => {
    // Navigate to home page
    await page.goto('/');
    
    // Wait for page to load
    await page.waitForLoadState('domcontentloaded');
    
    // Click on LND navigation
    await page.click('text=物流ネットワーク設計');
    
    // Wait for LND page to load
    await page.waitForTimeout(2000);
    
    // Verify we're on the LND page
    await expect(page.locator('text=Logistics Network Design (MELOS)')).toBeVisible();
    
    // Click on the first tab (single facility)
    await page.click('text=単一施設最適立地');
    
    // Wait for tab content to load
    await page.waitForTimeout(1000);
    
    // Verify the tab content loaded
    await expect(page.locator('text=Optimal Facility Location (Weiszfeld Algorithm)')).toBeVisible();
    
    // Verify the calculate button is present
    await expect(page.locator('button:has-text("CALCULATE OPTIMAL LOCATION")')).toBeVisible();
  });

  test('Can navigate to Inventory section', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Click on Inventory navigation
    await page.click('text=在庫管理');
    
    await page.waitForTimeout(2000);
    
    // Verify we're on the inventory page - using specific heading selector
    await expect(page.locator('h4:has-text("在庫管理")')).toBeVisible();
  });

  test('Can navigate to Routing section', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    // Click on Routing navigation
    await page.click('text=配送ルーティング');
    
    await page.waitForTimeout(2000);
    
    // Verify we're on the routing page - using Japanese text
    await expect(page.locator('text=配送・輸送最適化')).toBeVisible();
  });

  test('Backend API is accessible', async ({ page }) => {
    // Test if backend endpoints are reachable
    const response = await page.request.get('http://localhost:8000/docs');
    expect(response.status()).toBe(200);
    
    // Test a specific API endpoint
    const healthResponse = await page.request.get('http://localhost:8000/');
    expect(healthResponse.status()).toBe(200);
  });
});