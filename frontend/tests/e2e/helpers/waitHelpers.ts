import { Page, Locator } from '@playwright/test';

/**
 * Wait for loading indicator to disappear
 */
export async function waitForLoadingToDisappear(page: Page) {
  await page.waitForSelector('[role="progressbar"]', { state: 'detached' });
}

/**
 * Wait for result container to appear
 */
export async function waitForResult(page: Page, selector: string, timeout: number = 30000) {
  await page.waitForSelector(selector, { state: 'visible', timeout });
}

/**
 * Wait for chart to render (Plotly)
 */
export async function waitForChart(page: Page, chartId: string) {
  await page.waitForFunction(
    (id) => {
      const element = document.querySelector(id);
      return element && element.querySelector('.plot-container');
    },
    chartId,
    { timeout: 10000 }
  );
}

/**
 * Wait for map to render (Leaflet)
 */
export async function waitForMap(page: Page, mapSelector: string) {
  await page.waitForFunction(
    (selector) => {
      const element = document.querySelector(selector);
      return element && element.querySelector('.leaflet-container');
    },
    mapSelector,
    { timeout: 10000 }
  );
}

/**
 * Wait for API response
 */
export async function waitForApiResponse(page: Page, urlPattern: string | RegExp) {
  return page.waitForResponse(response => 
    typeof urlPattern === 'string' 
      ? response.url().includes(urlPattern)
      : urlPattern.test(response.url())
  );
}