import { test, expect } from '@playwright/test';
import { uploadFile, uploadFiles } from './helpers/fileUpload';
import { waitForLoadingToDisappear, waitForResult, waitForChart, waitForApiResponse } from './helpers/waitHelpers';

test.describe('Logistics Network Design (LND) Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Navigate to LND section - using Japanese text
    await page.click('text=物流ネットワーク設計');
  });

  test('Weiszfeld single facility location', async ({ page }) => {
    // Navigate to Weiszfeld tab - using Japanese text
    await page.click('text=単一施設最適立地');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_standard.csv');
    
    // Wait for file to load
    await page.waitForTimeout(2000);
    
    // Click calculate button - using actual button text from UI
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/weiszfeld-location');
    await page.click('button:has-text("CALCULATE OPTIMAL LOCATION")');
    
    // Wait for API response
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    // Wait for loading to complete
    await waitForLoadingToDisappear(page);
    
    // Check if results are displayed
    await waitForResult(page, '[data-testid="weiszfeld-result"]');
    
    // Verify map is rendered
    await waitForChart(page, '[data-testid="weiszfeld-map"]');
    
    // Check if optimal location coordinates are displayed
    const optimalLocation = await page.textContent('[data-testid="optimal-location"]');
    expect(optimalLocation).toContain('Optimal Location:');
    expect(optimalLocation).toMatch(/Latitude: -?\d+\.\d+/);
    expect(optimalLocation).toMatch(/Longitude: -?\d+\.\d+/);
    
    // Check if total cost is displayed
    const totalCost = await page.textContent('[data-testid="total-cost"]');
    expect(totalCost).toMatch(/Total Cost: \d+(\.\d+)?/);
  });

  test('Multi-facility Weiszfeld location', async ({ page }) => {
    // Navigate to Multi-facility tab
    await page.click('text=Multi-Facility Weiszfeld');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_standard.csv');
    
    // Set parameters
    await page.fill('input[name="numFacilities"]', '3');
    await page.fill('input[name="maxIterations"]', '500');
    await page.fill('input[name="tolerance"]', '1e-4');
    
    // Click calculate button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/multi-facility-weiszfeld');
    await page.click('button:has-text("Calculate Multiple Facilities")');
    
    // Wait for response
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="multi-facility-result"]');
    
    // Check if all 3 facilities are displayed
    const facilities = await page.$$('[data-testid="facility-location"]');
    expect(facilities).toHaveLength(3);
    
    // Verify assignments are shown
    const assignments = await page.textContent('[data-testid="customer-assignments"]');
    expect(assignments).toContain('Customer Assignments');
  });

  test('Customer clustering - K-means', async ({ page }) => {
    // Navigate to Customer Clustering tab
    await page.click('text=Customer Clustering');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_standard.csv');
    
    // Select K-means method
    await page.selectOption('select[name="clusteringMethod"]', 'kmeans');
    
    // Set number of clusters
    await page.fill('input[name="numClusters"]', '5');
    
    // Click cluster button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/customer-clustering');
    await page.click('button:has-text("Perform Clustering")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="clustering-result"]');
    
    // Verify cluster visualization
    await waitForChart(page, '[data-testid="cluster-map"]');
    
    // Check cluster statistics
    const clusterStats = await page.$$('[data-testid="cluster-stat"]');
    expect(clusterStats).toHaveLength(5);
  });

  test('Customer clustering - Hierarchical', async ({ page }) => {
    // Navigate to Customer Clustering tab
    await page.click('text=Customer Clustering');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_standard.csv');
    
    // Select hierarchical method
    await page.selectOption('select[name="clusteringMethod"]', 'hierarchical');
    
    // Set parameters
    await page.fill('input[name="numClusters"]', '4');
    await page.selectOption('select[name="linkageMethod"]', 'ward');
    
    // Click cluster button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/customer-clustering');
    await page.click('button:has-text("Perform Clustering")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="clustering-result"]');
    
    // Verify dendrogram is displayed
    await waitForChart(page, '[data-testid="dendrogram"]');
  });

  test('K-median optimization', async ({ page }) => {
    // Navigate to K-Median tab
    await page.click('text=K-Median Optimization');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_small.csv');
    
    // Set parameters
    await page.fill('input[name="k"]', '2');
    await page.fill('input[name="maxIterations"]', '50');
    
    // Click solve button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/k-median-optimization');
    await page.click('button:has-text("Solve K-Median")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="k-median-result"]');
    
    // Check optimization progress chart
    await waitForChart(page, '[data-testid="optimization-progress"]');
    
    // Verify selected facilities
    const selectedFacilities = await page.$$('[data-testid="selected-facility"]');
    expect(selectedFacilities).toHaveLength(2);
  });

  test('Multiple source LND solving', async ({ page }) => {
    // Navigate to Multi-Source LND tab
    await page.click('text=Multi-Source LND');
    
    // Upload all required files
    const fileInputs = await page.$$('input[type="file"]');
    expect(fileInputs).toHaveLength(6); // 6 files required
    
    await uploadFile(page, 'input[name="customerFile"]', 'ms_lnd_customers.csv');
    await uploadFile(page, 'input[name="warehouseFile"]', 'ms_lnd_warehouses.csv');
    await uploadFile(page, 'input[name="factoryFile"]', 'ms_lnd_factories.csv');
    await uploadFile(page, 'input[name="productFile"]', 'ms_lnd_products.csv');
    await uploadFile(page, 'input[name="demandFile"]', 'ms_lnd_demand.csv');
    await uploadFile(page, 'input[name="capacityFile"]', 'ms_lnd_factory_capacity.csv');
    
    // Set cost parameters
    await page.fill('input[name="transportationCost"]', '1.5');
    await page.fill('input[name="deliveryCost"]', '2.0');
    await page.fill('input[name="warehouseFixedCost"]', '15000');
    
    // Click solve button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/multiple-source-lnd');
    await page.click('button:has-text("Solve LND Problem")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    // This might take longer
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="lnd-solution"]', 60000); // 60 second timeout
    
    // Check if solution summary is displayed
    const solutionSummary = await page.textContent('[data-testid="solution-summary"]');
    expect(solutionSummary).toContain('Selected Warehouses');
    expect(solutionSummary).toContain('Total Cost');
    
    // Verify network visualization
    await waitForChart(page, '[data-testid="network-map"]');
  });

  test('Elbow method analysis', async ({ page }) => {
    // Navigate to Elbow Method tab
    await page.click('text=Elbow Method');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'elbow_customers_3clusters.csv');
    
    // Set parameters
    await page.fill('input[name="minFacilities"]', '1');
    await page.fill('input[name="maxFacilities"]', '6');
    await page.selectOption('select[name="algorithm"]', 'weiszfeld');
    
    // Click analyze button
    const apiResponsePromise = waitForApiResponse(page, '/api/lnd/elbow-method-analysis');
    await page.click('button:has-text("Run Elbow Analysis")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    // This analysis takes time
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="elbow-result"]', 90000); // 90 second timeout
    
    // Check elbow chart
    await waitForChart(page, '[data-testid="elbow-chart"]');
    
    // Verify optimal number is suggested
    const optimalSuggestion = await page.textContent('[data-testid="optimal-facilities"]');
    expect(optimalSuggestion).toContain('Suggested number of facilities');
  });

  test('Sample data download', async ({ page }) => {
    // Click on sample data info button
    await page.click('button[aria-label="Sample Data Info"]');
    
    // Wait for dialog
    await page.waitForSelector('[role="dialog"]');
    
    // Check if sample datasets are listed
    const datasetList = await page.$$('[data-testid="dataset-item"]');
    expect(datasetList.length).toBeGreaterThan(0);
    
    // Download a sample dataset
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="download-standard"]');
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('.csv');
  });

  // New tests for recently added features

  test('Excel template generation and workflow', async ({ page }) => {
    // Navigate to Excel Integration tab
    await page.click('text=Excel Integration');
    
    // Download template
    const downloadPromise = page.waitForEvent('download');
    await page.click('button:has-text("Download Excel Template")');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('template');
    expect(download.suggestedFilename()).toContain('.xlsx');
  });

  test('Customer aggregation with K-means', async ({ page }) => {
    // Navigate to Aggregation tab
    await page.click('text=Customer Aggregation');
    
    // Upload customer file
    await uploadFile(page, 'input[type="file"]', 'customers_regional.csv');
    
    // Select K-means aggregation
    await page.click('input[value="kmeans"]');
    await page.fill('input[name="numClusters"]', '10');
    
    // Click aggregate button
    const apiResponsePromise = waitForApiResponse(page, '/api/aggregation/kmeans');
    await page.click('button:has-text("Aggregate Customers")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="aggregation-result"]');
    
    // Verify aggregated customers are displayed
    const aggregatedCustomers = await page.$$('[data-testid="aggregated-customer"]');
    expect(aggregatedCustomers).toHaveLength(10);
  });

  test('CO2 emission calculation', async ({ page }) => {
    // Navigate to CO2 Calculation tab
    await page.click('text=CO2 Emissions');
    
    // Fill in parameters
    await page.fill('input[name="capacity"]', '5000');
    await page.fill('input[name="loadRate"]', '0.7');
    await page.fill('input[name="distance"]', '100');
    await page.check('input[name="isDiesel"]');
    
    // Click calculate button
    const apiResponsePromise = waitForApiResponse(page, '/api/co2/calculate');
    await page.click('button:has-text("Calculate CO2")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForResult(page, '[data-testid="co2-result"]');
    
    // Check emission values
    const fuelConsumption = await page.textContent('[data-testid="fuel-consumption"]');
    expect(fuelConsumption).toMatch(/Fuel Consumption: \d+\.\d+ L\/ton-km/);
    
    const co2Emission = await page.textContent('[data-testid="co2-emission"]');
    expect(co2Emission).toMatch(/CO2 Emission: \d+\.\d+ g\/ton-km/);
  });

  test('Network visualization with great circle distances', async ({ page }) => {
    // Navigate to Network Generation tab
    await page.click('text=Network Generation');
    
    // Upload files
    await uploadFile(page, 'input[name="customerFile"]', 'customers_small.csv');
    await uploadFile(page, 'input[name="dcFile"]', 'facilities.csv');
    await uploadFile(page, 'input[name="plantFile"]', 'ms_lnd_factories.csv');
    
    // Set thresholds
    await page.fill('input[name="plantDcThreshold"]', '500');
    await page.fill('input[name="dcCustThreshold"]', '200');
    
    // Generate network
    const apiResponsePromise = waitForApiResponse(page, '/api/network/generate-great-circle');
    await page.click('button:has-text("Generate Network")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    
    // Check network visualization
    await waitForChart(page, '[data-testid="network-visualization"]');
    
    // Verify network statistics
    const stats = await page.textContent('[data-testid="network-stats"]');
    expect(stats).toContain('Total Connections');
    expect(stats).toContain('Total Distance');
  });
});