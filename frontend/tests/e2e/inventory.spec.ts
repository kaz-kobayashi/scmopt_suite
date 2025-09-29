import { test, expect } from '@playwright/test';
import { uploadFile } from './helpers/fileUpload';
import { waitForLoadingToDisappear, waitForResult, waitForChart, waitForApiResponse } from './helpers/waitHelpers';

test.describe('Inventory Management Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Navigate to Inventory section - using Japanese text
    await page.click('text=在庫管理');
  });

  test('EOQ calculation with basic parameters', async ({ page }) => {
    // Navigate to EOQ tab
    await page.click('text=EOQ Analysis');
    
    // Fill in EOQ parameters
    await page.fill('input[name="K"]', '100'); // Fixed cost
    await page.fill('input[name="d"]', '1000'); // Demand rate
    await page.fill('input[name="h"]', '0.2'); // Holding cost
    
    // Click calculate button
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/eoq');
    await page.click('button:has-text("Calculate EOQ")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="eoq-result"]');
    
    // Check EOQ value
    const eoq = await page.textContent('[data-testid="optimal-order-quantity"]');
    expect(eoq).toMatch(/Optimal Order Quantity: \d+(\.\d+)?/);
    
    // Check total cost
    const totalCost = await page.textContent('[data-testid="total-annual-cost"]');
    expect(totalCost).toMatch(/Total Annual Cost: \$?\d+(\.\d+)?/);
    
    // Verify cost breakdown chart
    await waitForChart(page, '[data-testid="cost-breakdown-chart"]');
  });

  test('EOQ calculation with backorder cost', async ({ page }) => {
    // Navigate to EOQ tab
    await page.click('text=EOQ Analysis');
    
    // Enable backorder option
    await page.check('input[name="includeBackorder"]');
    
    // Fill parameters
    await page.fill('input[name="K"]', '150');
    await page.fill('input[name="d"]', '2000');
    await page.fill('input[name="h"]', '0.3');
    await page.fill('input[name="b"]', '5'); // Backorder cost
    
    // Calculate
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/eoq');
    await page.click('button:has-text("Calculate EOQ")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="eoq-result"]');
    
    // Check if backorder quantity is displayed
    const backorderQty = await page.textContent('[data-testid="backorder-quantity"]');
    expect(backorderQty).toContain('Backorder Quantity');
  });

  test('Inventory simulation - (Q,R) policy', async ({ page }) => {
    // Navigate to Simulation tab
    await page.click('text=Inventory Simulation');
    
    // Fill simulation parameters
    await page.fill('input[name="n_samples"]', '100');
    await page.fill('input[name="n_periods"]', '52'); // 52 weeks
    await page.fill('input[name="mu"]', '100'); // Mean demand
    await page.fill('input[name="sigma"]', '20'); // Std deviation
    await page.fill('input[name="LT"]', '2'); // Lead time
    await page.fill('input[name="Q"]', '500'); // Order quantity
    await page.fill('input[name="R"]', '200'); // Reorder point
    await page.fill('input[name="h"]', '0.5'); // Holding cost
    await page.fill('input[name="b"]', '10'); // Backorder cost
    
    // Run simulation
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/simulate');
    await page.click('button:has-text("Run Simulation")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="simulation-result"]');
    
    // Check simulation statistics
    const avgCost = await page.textContent('[data-testid="average-cost"]');
    expect(avgCost).toMatch(/Average Cost per Period: \$?\d+(\.\d+)?/);
    
    const stdDev = await page.textContent('[data-testid="cost-std-dev"]');
    expect(stdDev).toContain('Standard Deviation');
    
    // Verify confidence interval
    const ci = await page.textContent('[data-testid="confidence-interval"]');
    expect(ci).toContain('95% Confidence Interval');
    
    // Check simulation chart
    await waitForChart(page, '[data-testid="simulation-chart"]');
  });

  test('Inventory simulation - (s,S) policy', async ({ page }) => {
    // Navigate to Simulation tab
    await page.click('text=Inventory Simulation');
    
    // Select (s,S) policy
    await page.selectOption('select[name="policyType"]', 'sS');
    
    // Fill parameters
    await page.fill('input[name="n_samples"]', '50');
    await page.fill('input[name="n_periods"]', '26'); // 26 periods
    await page.fill('input[name="mu"]', '150');
    await page.fill('input[name="sigma"]', '30');
    await page.fill('input[name="LT"]', '1');
    await page.fill('input[name="s"]', '100'); // Reorder point
    await page.fill('input[name="S"]', '600'); // Order-up-to level
    
    // Run simulation
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/simulate');
    await page.click('button:has-text("Run Simulation")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="simulation-result"]');
    
    // Check inventory level chart
    await waitForChart(page, '[data-testid="inventory-level-chart"]');
  });

  test('Multi-echelon inventory optimization', async ({ page }) => {
    // Navigate to Multi-Echelon tab
    await page.click('text=Multi-Echelon');
    
    // Define network structure
    await page.fill('textarea[name="networkStructure"]', JSON.stringify({
      nodes: ['Plant', 'DC1', 'DC2', 'Store1', 'Store2', 'Store3'],
      edges: [
        ['Plant', 'DC1'],
        ['Plant', 'DC2'],
        ['DC1', 'Store1'],
        ['DC1', 'Store2'],
        ['DC2', 'Store3']
      ]
    }));
    
    // Define demand data
    await page.fill('textarea[name="demandData"]', JSON.stringify({
      'Store1': { mean: 100, std: 20 },
      'Store2': { mean: 150, std: 30 },
      'Store3': { mean: 80, std: 15 }
    }));
    
    // Set cost parameters
    await page.fill('textarea[name="costParams"]', JSON.stringify({
      holding_cost: {
        'Plant': 0.1,
        'DC1': 0.15,
        'DC2': 0.15,
        'Store1': 0.2,
        'Store2': 0.2,
        'Store3': 0.2
      },
      ordering_cost: {
        'DC1': 500,
        'DC2': 500,
        'Store1': 100,
        'Store2': 100,
        'Store3': 100
      }
    }));
    
    // Optimize
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/multi-echelon-json');
    await page.click('button:has-text("Optimize Network")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="multi-echelon-result"]');
    
    // Check echelon policy table
    const policyTable = await page.waitForSelector('[data-testid="echelon-policy-table"]');
    expect(policyTable).toBeTruthy();
    
    // Verify base-stock levels
    const baseStockLevels = await page.$$('[data-testid="base-stock-level"]');
    expect(baseStockLevels.length).toBeGreaterThan(0);
    
    // Check network visualization
    await waitForChart(page, '[data-testid="network-diagram"]');
  });

  test('Inventory ABC analysis', async ({ page }) => {
    // Navigate to ABC Analysis tab
    await page.click('text=ABC Analysis');
    
    // Upload inventory data file
    await uploadFile(page, 'input[type="file"]', 'product_sample.csv');
    
    // Set thresholds
    await page.fill('input[name="threshold"]', '0.8,0.15,0.05');
    await page.fill('input[name="aggCol"]', 'product');
    await page.fill('input[name="value"]', 'annual_value');
    
    // Perform analysis
    const apiResponsePromise = waitForApiResponse(page, '/api/inventory/inventory-abc');
    await page.click('button:has-text("Analyze")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="abc-result"]');
    
    // Check ABC distribution
    const categoryA = await page.textContent('[data-testid="category-A"]');
    expect(categoryA).toContain('A Items');
    expect(categoryA).toMatch(/\d+%/); // Should show percentage
    
    // Verify Pareto chart
    await waitForChart(page, '[data-testid="pareto-chart"]');
    
    // Check item classification table
    const classificationTable = await page.waitForSelector('[data-testid="classification-table"]');
    expect(classificationTable).toBeTruthy();
  });

  test('Safety stock calculation', async ({ page }) => {
    // Navigate to Safety Stock tab
    await page.click('text=Safety Stock');
    
    // Fill parameters
    await page.fill('input[name="demandMean"]', '1000');
    await page.fill('input[name="demandStd"]', '100');
    await page.fill('input[name="leadTime"]', '7');
    await page.fill('input[name="leadTimeStd"]', '1');
    await page.fill('input[name="serviceLevel"]', '0.95');
    
    // Calculate safety stock
    await page.click('button:has-text("Calculate Safety Stock")');
    
    await waitForResult(page, '[data-testid="safety-stock-result"]');
    
    // Check calculated values
    const safetyStock = await page.textContent('[data-testid="safety-stock-value"]');
    expect(safetyStock).toMatch(/Safety Stock: \d+(\.\d+)?/);
    
    const reorderPoint = await page.textContent('[data-testid="reorder-point"]');
    expect(reorderPoint).toMatch(/Reorder Point: \d+(\.\d+)?/);
    
    // Verify service level chart
    await waitForChart(page, '[data-testid="service-level-chart"]');
  });

  test('Inventory turnover analysis', async ({ page }) => {
    // Navigate to Turnover Analysis tab
    await page.click('text=Turnover Analysis');
    
    // Upload sales and inventory data
    await uploadFile(page, 'input[name="salesFile"]', 'demand_sample.csv');
    
    // Set analysis period
    await page.selectOption('select[name="period"]', 'annual');
    
    // Analyze
    const apiResponsePromise = waitForApiResponse(page, '/api/analytics/inventory-turnover');
    await page.click('button:has-text("Analyze Turnover")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="turnover-result"]');
    
    // Check turnover metrics
    const turnoverRatio = await page.textContent('[data-testid="turnover-ratio"]');
    expect(turnoverRatio).toMatch(/Turnover Ratio: \d+(\.\d+)?/);
    
    const daysInventory = await page.textContent('[data-testid="days-inventory"]');
    expect(daysInventory).toMatch(/Days of Inventory: \d+(\.\d+)?/);
  });
});