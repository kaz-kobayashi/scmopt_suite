import { test, expect } from '@playwright/test';
import { uploadFile } from './helpers/fileUpload';
import { waitForLoadingToDisappear, waitForResult, waitForChart, waitForMap, waitForApiResponse } from './helpers/waitHelpers';

test.describe('Routing and VRP Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Navigate to Routing section - using Japanese text
    await page.click('text=配送ルーティング');
  });

  test('Basic route optimization', async ({ page }) => {
    // Navigate to Route Optimization tab
    await page.click('text=Route Optimization');
    
    // Upload locations file
    await uploadFile(page, 'input[type="file"]', 'locations_sample.csv');
    
    // Set parameters
    await page.fill('input[name="vehicleCapacity"]', '1000');
    await page.fill('input[name="maxRoutes"]', '3');
    await page.fill('input[name="depotName"]', 'Depot');
    
    // Click optimize button
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/route-optimization');
    await page.click('button:has-text("Optimize Routes")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="route-result"]');
    
    // Check if routes are displayed
    const routes = await page.$$('[data-testid="route-item"]');
    expect(routes.length).toBeGreaterThan(0);
    expect(routes.length).toBeLessThanOrEqual(3);
    
    // Verify map visualization
    await waitForMap(page, '[data-testid="route-map"]');
    
    // Check route statistics
    const totalDistance = await page.textContent('[data-testid="total-distance"]');
    expect(totalDistance).toMatch(/Total Distance: \d+(\.\d+)? km/);
    
    const vehiclesUsed = await page.textContent('[data-testid="vehicles-used"]');
    expect(vehiclesUsed).toMatch(/Vehicles Used: \d+/);
  });

  test('Advanced VRP with time windows', async ({ page }) => {
    // Navigate to Advanced VRP tab
    await page.click('text=Advanced VRP');
    
    // Upload locations with time windows
    await uploadFile(page, 'input[type="file"]', 'orders_sample.csv');
    
    // Set vehicle parameters
    await page.fill('input[name="vehicleCapacity"]', '800');
    await page.fill('input[name="maxRoutes"]', '4');
    await page.fill('input[name="workingStart"]', '8');
    await page.fill('input[name="workingEnd"]', '18');
    await page.fill('input[name="serviceTime"]', '15'); // 15 minutes per stop
    
    // Enable time windows
    await page.check('input[name="enableTimeWindows"]');
    
    // Optimize
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/advanced-vrp');
    await page.click('button:has-text("Solve VRP")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="vrp-solution"]');
    
    // Check Gantt chart for time windows
    await waitForChart(page, '[data-testid="vrp-gantt-chart"]');
    
    // Verify time window violations
    const violations = await page.textContent('[data-testid="time-violations"]');
    expect(violations).toContain('Time Window Violations: 0');
  });

  test('PyVRP - Basic CVRP', async ({ page }) => {
    // Navigate to PyVRP Interface
    await page.click('text=PyVRP Interface');
    
    // Select CVRP variant
    await page.selectOption('select[name="vrpVariant"]', 'cvrp');
    
    // Add locations
    await page.click('button:has-text("Add Location")');
    await page.fill('input[name="location[0].name"]', 'Depot');
    await page.fill('input[name="location[0].lat"]', '35.6762');
    await page.fill('input[name="location[0].lon"]', '139.6503');
    await page.fill('input[name="location[0].demand"]', '0');
    
    // Add customers
    for (let i = 1; i <= 5; i++) {
      await page.click('button:has-text("Add Location")');
      await page.fill(`input[name="location[${i}].name"]`, `Customer ${i}`);
      await page.fill(`input[name="location[${i}].lat"]`, String(35.6762 + Math.random() * 0.1));
      await page.fill(`input[name="location[${i}].lon"]`, String(139.6503 + Math.random() * 0.1));
      await page.fill(`input[name="location[${i}].demand"]`, String(Math.floor(Math.random() * 100 + 50)));
    }
    
    // Set vehicle capacity
    await page.fill('input[name="vehicleCapacity"]', '300');
    
    // Solve
    const apiResponsePromise = waitForApiResponse(page, '/api/pyvrp/solve/cvrp');
    await page.click('button:has-text("Solve CVRP")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="pyvrp-solution"]');
    
    // Check solution status
    const status = await page.textContent('[data-testid="solution-status"]');
    expect(status).toContain('Status: optimal');
    
    // Verify route visualization
    await waitForMap(page, '[data-testid="pyvrp-route-map"]');
  });

  test('PyVRP - Multi-depot VRP', async ({ page }) => {
    // Navigate to PyVRP Interface
    await page.click('text=PyVRP Interface');
    
    // Select MDVRP variant
    await page.selectOption('select[name="vrpVariant"]', 'mdvrp');
    
    // Upload pre-configured MDVRP data
    await uploadFile(page, 'input[name="dataFile"]', 'locations_sample.csv');
    
    // Add depot configuration
    await page.click('button:has-text("Add Depot")');
    await page.fill('input[name="depot[0].name"]', 'Depot 1');
    await page.fill('input[name="depot[0].capacity"]', '1000');
    await page.fill('input[name="depot[0].numVehicles"]', '3');
    
    await page.click('button:has-text("Add Depot")');
    await page.fill('input[name="depot[1].name"]', 'Depot 2');
    await page.fill('input[name="depot[1].capacity"]', '800');
    await page.fill('input[name="depot[1].numVehicles"]', '2');
    
    // Solve MDVRP
    const apiResponsePromise = waitForApiResponse(page, '/api/pyvrp/solve/mdvrp');
    await page.click('button:has-text("Solve MDVRP")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="mdvrp-solution"]');
    
    // Check routes by depot
    const depot1Routes = await page.$$('[data-testid="depot-1-routes"] [data-testid="route"]');
    const depot2Routes = await page.$$('[data-testid="depot-2-routes"] [data-testid="route"]');
    
    expect(depot1Routes.length).toBeLessThanOrEqual(3);
    expect(depot2Routes.length).toBeLessThanOrEqual(2);
  });

  test('CO2 emission calculation for routes', async ({ page }) => {
    // Navigate to Emissions tab
    await page.click('text=Route Emissions');
    
    // Fill vehicle and route parameters
    await page.fill('input[name="capacity"]', '3500'); // 3.5 ton truck
    await page.fill('input[name="loadingRate"]', '0.75');
    await page.selectOption('select[name="fuelType"]', 'diesel');
    
    // Click calculate
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/co2-calculation');
    await page.click('button:has-text("Calculate Emissions")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForResult(page, '[data-testid="emission-result"]');
    
    // Check emission calculations
    const fuelConsumption = await page.textContent('[data-testid="fuel-consumption"]');
    expect(fuelConsumption).toMatch(/Fuel Consumption: \d+\.\d+ L\/ton-km/);
    
    const co2Emissions = await page.textContent('[data-testid="co2-emissions"]');
    expect(co2Emissions).toMatch(/CO2 Emissions: \d+\.\d+ g\/ton-km/);
    
    // Check annual estimates
    const annualEstimates = await page.textContent('[data-testid="annual-estimates"]');
    expect(annualEstimates).toContain('Annual CO2');
  });

  test('Distance matrix generation', async ({ page }) => {
    // Navigate to Distance Matrix tab
    await page.click('text=Distance Matrix');
    
    // Upload locations file
    await uploadFile(page, 'input[type="file"]', 'facilities.csv');
    
    // Select output format
    await page.selectOption('select[name="outputFormat"]', 'json');
    
    // Generate matrix
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/distance-matrix');
    await page.click('button:has-text("Generate Matrix")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="distance-matrix-result"]');
    
    // Check matrix display
    await waitForChart(page, '[data-testid="distance-heatmap"]');
    
    // Verify matrix size
    const matrixSize = await page.textContent('[data-testid="matrix-size"]');
    expect(matrixSize).toMatch(/Matrix Size: \d+ x \d+/);
  });

  test('Delivery schedule creation', async ({ page }) => {
    // Navigate to Delivery Schedule tab
    await page.click('text=Delivery Schedule');
    
    // Upload orders file
    await uploadFile(page, 'input[type="file"]', 'orders_sample.csv');
    
    // Set working hours
    await page.fill('input[name="workingStart"]', '7');
    await page.fill('input[name="workingEnd"]', '19');
    await page.fill('input[name="serviceTime"]', '20');
    
    // Create schedule
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/delivery-schedule');
    await page.click('button:has-text("Create Schedule")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="schedule-result"]');
    
    // Check schedule view
    await waitForChart(page, '[data-testid="schedule-timeline"]');
    
    // Download schedule
    const downloadPromise = page.waitForEvent('download');
    await page.click('button:has-text("Export Schedule")');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toContain('schedule');
    expect(download.suggestedFilename()).toMatch(/\.(csv|xlsx)$/);
  });

  test('VRPLIB instance solving', async ({ page }) => {
    // Navigate to VRPLIB Solver tab
    await page.click('text=VRPLIB Solver');
    
    // Upload VRPLIB format file
    await uploadFile(page, 'input[type="file"]', 'X-n439-k37.vrp');
    
    // Set solver parameters
    await page.fill('input[name="maxRuntime"]', '30');
    await page.fill('input[name="maxIterations"]', '5000');
    
    // Solve
    const apiResponsePromise = waitForApiResponse(page, '/api/routing/vrplib-solve');
    await page.click('button:has-text("Solve Instance")');
    
    const response = await apiResponsePromise;
    expect(response.status()).toBe(200);
    
    // This may take longer for large instances
    await waitForLoadingToDisappear(page);
    await waitForResult(page, '[data-testid="vrplib-solution"]', 60000); // 60 second timeout
    
    // Check solution quality
    const objectiveValue = await page.textContent('[data-testid="objective-value"]');
    expect(objectiveValue).toMatch(/Objective Value: \d+(\.\d+)?/);
    
    const gap = await page.textContent('[data-testid="optimality-gap"]');
    expect(gap).toMatch(/Gap: \d+(\.\d+)?%/);
    
    // Verify solution visualization
    await waitForMap(page, '[data-testid="vrplib-solution-map"]');
  });

  test('Route comparison', async ({ page }) => {
    // Navigate to Route Comparison tab
    await page.click('text=Route Comparison');
    
    // Upload same file for different scenarios
    await uploadFile(page, 'input[name="baselineFile"]', 'locations_sample.csv');
    
    // Set baseline parameters
    await page.fill('input[name="baselineCapacity"]', '1000');
    await page.fill('input[name="baselineRoutes"]', '3');
    
    // Calculate baseline
    await page.click('button:has-text("Calculate Baseline")');
    await waitForLoadingToDisappear(page);
    
    // Set optimized parameters
    await page.fill('input[name="optimizedCapacity"]', '1200');
    await page.fill('input[name="optimizedRoutes"]', '2');
    
    // Calculate optimized
    await page.click('button:has-text("Calculate Optimized")');
    await waitForLoadingToDisappear(page);
    
    // View comparison
    await page.click('button:has-text("Compare Results")');
    await waitForResult(page, '[data-testid="comparison-result"]');
    
    // Check comparison metrics
    const distanceImprovement = await page.textContent('[data-testid="distance-improvement"]');
    expect(distanceImprovement).toMatch(/Distance Improvement: -?\d+(\.\d+)?%/);
    
    const vehicleReduction = await page.textContent('[data-testid="vehicle-reduction"]');
    expect(vehicleReduction).toContain('Vehicle Reduction');
    
    // Verify comparison chart
    await waitForChart(page, '[data-testid="comparison-chart"]');
  });
});