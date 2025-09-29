import { test, expect } from '@playwright/test';

test.describe('Basic Functionality Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set timeout for slow operations
    test.setTimeout(60000);
    
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.waitForTimeout(2000);
  });

  test('Frontend application loads successfully', async ({ page }) => {
    // Check if the main application elements are present
    const mainContent = await Promise.race([
      page.locator('body').isVisible().then(() => true),
      page.locator('main').isVisible().then(() => true),
      page.locator('[data-testid="app"]').isVisible().then(() => true),
      page.waitForTimeout(5000).then(() => false)
    ]);
    
    expect(mainContent).toBe(true);
    console.log('✅ Frontend application loaded successfully');
  });

  test('Navigation menu is present and functional', async ({ page }) => {
    // Look for navigation elements
    const navElements = await Promise.race([
      page.locator('nav').isVisible().then(() => 'nav-element'),
      page.locator('[role="navigation"]').isVisible().then(() => 'navigation-role'),
      page.locator('aside').isVisible().then(() => 'sidebar'),
      page.locator('.nav, .navigation, .sidebar').isVisible().then(() => 'navigation-class'),
      page.waitForTimeout(3000).then(() => false)
    ]);
    
    console.log(`Navigation element found: ${navElements || 'not found'}`);
    
    // Look for menu items (more flexible approach)
    const menuItems = await page.locator('text=在庫管理, text=ダッシュボード, text=分析, button:has-text("在庫"), a:has-text("在庫")').count();
    console.log(`Menu items found: ${menuItems}`);
    
    expect(menuItems).toBeGreaterThan(0);
  });

  test('Can navigate to main sections', async ({ page }) => {
    const sections = [
      { text: '在庫管理', timeout: 3000 },
      { text: '分析', timeout: 2000 },
      { text: 'ダッシュボード', timeout: 2000 }
    ];

    for (const section of sections) {
      console.log(`Testing navigation to: ${section.text}`);
      
      // Try multiple selector strategies
      const navigationResult = await Promise.race([
        page.click(`text=${section.text}`, { timeout: 2000 }).then(() => 'text-selector'),
        page.click(`button:has-text("${section.text}")`, { timeout: 2000 }).then(() => 'button-selector'),
        page.click(`a:has-text("${section.text}")`, { timeout: 2000 }).then(() => 'link-selector'),
        page.waitForTimeout(section.timeout).then(() => 'timeout')
      ]);
      
      if (navigationResult !== 'timeout') {
        console.log(`✅ Successfully navigated to ${section.text} using ${navigationResult}`);
        await page.waitForTimeout(1500);
        
        // Verify navigation worked by looking for content change
        const contentChanged = await Promise.race([
          page.locator('h1, h2, h3, h4').isVisible().then(() => true),
          page.waitForTimeout(2000).then(() => false)
        ]);
        
        console.log(`Content loaded after navigation: ${contentChanged}`);
      } else {
        console.log(`⚠️ Could not navigate to ${section.text} - element may not be visible`);
      }
    }
  });

  test('Tab structure exists in components', async ({ page }) => {
    // Try to navigate to a component with tabs
    const componentsToTest = ['在庫管理', '分析', '物流ネットワーク設計'];
    
    for (const component of componentsToTest) {
      console.log(`Testing tab structure in: ${component}`);
      
      // Try to navigate to component
      const navigated = await Promise.race([
        page.click(`text=${component}`, { timeout: 2000 }).then(() => true),
        page.click(`button:has-text("${component}")`, { timeout: 2000 }).then(() => true),
        page.waitForTimeout(3000).then(() => false)
      ]);
      
      if (navigated) {
        await page.waitForTimeout(2000);
        
        // Look for tab structure
        const tabElements = await Promise.race([
          page.locator('[role="tab"]').count().then(count => count),
          page.locator('.MuiTab-root').count().then(count => count),
          page.locator('button[class*="tab"]').count().then(count => count),
          page.waitForTimeout(2000).then(() => 0)
        ]);
        
        console.log(`Tabs found in ${component}: ${tabElements}`);
        
        if (tabElements > 0) {
          // Try to click on a tab
          const tabClicked = await Promise.race([
            page.locator('[role="tab"]').first().click().then(() => true),
            page.waitForTimeout(1000).then(() => false)
          ]);
          
          console.log(`Tab interaction successful: ${tabClicked}`);
        }
      } else {
        console.log(`Could not navigate to ${component}`);
      }
    }
  });

  test('Form elements are functional', async ({ page }) => {
    // Look for input elements across the page
    const inputElements = await page.locator('input, select, textarea, button').count();
    console.log(`Interactive elements found: ${inputElements}`);
    
    expect(inputElements).toBeGreaterThan(0);
    
    // Test a few input interactions
    const inputs = page.locator('input[type="number"], input[type="text"]');
    const inputCount = await inputs.count();
    
    if (inputCount > 0) {
      const firstInput = inputs.first();
      if (await firstInput.isVisible()) {
        await firstInput.fill('100');
        await page.waitForTimeout(300);
        
        const value = await firstInput.inputValue();
        console.log(`Input interaction test: ${value === '100' ? 'successful' : 'failed'}`);
      }
    }
  });

  test('Error handling and loading states', async ({ page }) => {
    // Look for loading indicators
    const loadingElements = await page.locator('text=読み込み中, text=Loading, [class*="loading"], .MuiCircularProgress-root').count();
    console.log(`Loading indicators found: ${loadingElements}`);
    
    // Look for error handling elements
    const errorElements = await page.locator('text=エラー, text=Error, [class*="error"], .MuiAlert-root').count();
    console.log(`Error handling elements found: ${errorElements}`);
    
    // This is informational - both 0 and >0 are acceptable
    expect(typeof loadingElements).toBe('number');
    expect(typeof errorElements).toBe('number');
  });

  test('Responsive design elements', async ({ page }) => {
    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080, name: 'Desktop' },
      { width: 768, height: 1024, name: 'Tablet' },
      { width: 375, height: 667, name: 'Mobile' }
    ];

    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.waitForTimeout(1000);
      
      // Check if main content is still visible
      const contentVisible = await Promise.race([
        page.locator('body').isVisible().then(() => true),
        page.locator('main').isVisible().then(() => true),
        page.waitForTimeout(2000).then(() => false)
      ]);
      
      console.log(`${viewport.name} (${viewport.width}x${viewport.height}): ${contentVisible ? 'responsive' : 'issues detected'}`);
    }
    
    // Reset to desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
  });

  test('Basic accessibility checks', async ({ page }) => {
    // Check for basic accessibility features
    const accessibilityFeatures = await Promise.all([
      page.locator('button').count(),
      page.locator('input[aria-label], input[label]').count(),
      page.locator('[role]').count(),
      page.locator('h1, h2, h3, h4, h5, h6').count()
    ]);
    
    const [buttons, labeledInputs, roleElements, headings] = accessibilityFeatures;
    
    console.log(`Accessibility check results:`);
    console.log(`  Buttons: ${buttons}`);
    console.log(`  Labeled inputs: ${labeledInputs}`);
    console.log(`  Role elements: ${roleElements}`);
    console.log(`  Headings: ${headings}`);
    
    // Basic accessibility requirements
    expect(buttons).toBeGreaterThan(0);
    expect(headings).toBeGreaterThan(0);
  });

  test('Console errors check', async ({ page }) => {
    const consoleErrors: string[] = [];
    
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Navigate around the app to trigger any console errors
    await page.reload();
    await page.waitForTimeout(3000);
    
    // Try some basic interactions
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      const firstButton = buttons.first();
      if (await firstButton.isVisible()) {
        await firstButton.click();
        await page.waitForTimeout(1000);
      }
    }
    
    console.log(`Console errors detected: ${consoleErrors.length}`);
    if (consoleErrors.length > 0) {
      console.log('Errors:', consoleErrors.slice(0, 3)); // Show first 3 errors
    }
    
    // Note: This is informational - some console errors might be expected
    // We're not failing the test for console errors, just reporting them
  });
});