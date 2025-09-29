import { Page } from '@playwright/test';
import * as path from 'path';

/**
 * Helper function to upload a file
 */
export async function uploadFile(page: Page, selector: string, fileName: string) {
  const filePath = path.join(__dirname, '..', '..', '..', 'public', 'sample_data', fileName);
  await page.setInputFiles(selector, filePath);
}

/**
 * Helper function to upload multiple files
 */
export async function uploadFiles(page: Page, selector: string, fileNames: string[]) {
  const filePaths = fileNames.map(fileName => 
    path.join(__dirname, '..', '..', '..', 'public', 'sample_data', fileName)
  );
  await page.setInputFiles(selector, filePaths);
}

/**
 * Helper function to download file and get its content
 */
export async function downloadAndReadFile(page: Page): Promise<Buffer> {
  const downloadPromise = page.waitForEvent('download');
  // Trigger download here
  const download = await downloadPromise;
  const path = await download.path();
  if (!path) throw new Error('Download failed');
  
  // Read the downloaded file
  const fs = require('fs');
  return fs.readFileSync(path);
}