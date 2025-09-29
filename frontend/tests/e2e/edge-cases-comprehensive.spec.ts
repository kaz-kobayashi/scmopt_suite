import { test, expect } from '@playwright/test';

test.describe('Edge Cases and Error Scenarios - Complete Coverage', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(90000);
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000);
  });

  test.describe('境界値テストケース', () => {
    test('在庫管理 - 極端な値での境界値テスト', async ({ page }) => {
      console.log('🔍 在庫管理境界値テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // EOQ分析で極端な値をテスト
      await page.click('text=EOQ分析');
      await page.waitForTimeout(1000);
      
      const extremeValues = [
        { label: '発注コスト', values: ['0', '0.001', '999999999', '-100'] },
        { label: '需要量', values: ['0', '1', '1e10', '-1000'] },
        { label: '保管費率', values: ['0', '0.00001', '100', '-0.5'] },
        { label: '欠品費率', values: ['0', '0.000001', '1000', '-10'] }
      ];
      
      for (const param of extremeValues) {
        const input = page.locator(`input[label*="${param.label}"]`).first();
        if (await input.isVisible()) {
          for (const value of param.values) {
            await input.clear();
            await input.fill(value);
            await page.waitForTimeout(200);
            
            // 計算実行
            await page.click('button:has-text("計算")');
            await page.waitForTimeout(1000);
            
            // エラーまたは結果の確認
            const hasError = await page.locator('[class*="error"], text=エラー, text=無効').isVisible();
            const hasResult = await page.locator('text=最適発注量, canvas').isVisible();
            
            console.log(`値 ${value} for ${param.label}: ${hasError ? 'エラー' : hasResult ? '計算成功' : '不明'}`);
            
            // エラーダイアログを閉じる
            if (hasError) {
              const closeButton = page.locator('button:has-text("閉じる"), button:has-text("OK")').first();
              if (await closeButton.isVisible()) {
                await closeButton.click();
                await page.waitForTimeout(500);
              }
            }
          }
        }
      }
    });

    test('物流ネットワーク設計 - 座標系の境界値テスト', async ({ page }) => {
      console.log('🗺️ 座標系境界値テスト');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      
      // 極端な座標値でのテスト
      const extremeCoordinates = [
        { x: '-180', y: '-90' },  // 最小値
        { x: '180', y: '90' },    // 最大値
        { x: '0', y: '0' },       // 原点
        { x: '999999', y: '999999' }, // 範囲外
        { x: 'NaN', y: 'Infinity' }   // 無効値
      ];
      
      for (const coord of extremeCoordinates) {
        const xInput = page.locator('input[placeholder*="X座標"], input[placeholder*="経度"]').first();
        const yInput = page.locator('input[placeholder*="Y座標"], input[placeholder*="緯度"]').first();
        
        if (await xInput.isVisible() && await yInput.isVisible()) {
          await xInput.clear();
          await xInput.fill(coord.x);
          await yInput.clear();
          await yInput.fill(coord.y);
          await page.waitForTimeout(300);
          
          console.log(`座標テスト: (${coord.x}, ${coord.y})`);
        }
      }
    });

    test('シフト最適化 - 時間制約の境界値テスト', async ({ page }) => {
      console.log('⏰ シフト時間制約境界値テスト');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      await page.click('text=システム設定');
      await page.waitForTimeout(1000);
      
      // 異常な時間設定
      const timeTests = [
        { start: '00:00', end: '00:00' },     // 同じ時刻
        { start: '23:59', end: '00:01' },     // 日跨ぎ
        { start: '12:00', end: '11:00' },     // 逆順
        { start: '25:00', end: '26:00' },     // 無効時刻
        { start: '', end: '' }                // 空値
      ];
      
      for (const timeTest of timeTests) {
        const startInput = page.locator('input[label*="開始時刻"]').first();
        const endInput = page.locator('input[label*="終了時刻"]').first();
        
        if (await startInput.isVisible() && await endInput.isVisible()) {
          await startInput.clear();
          if (timeTest.start) await startInput.fill(timeTest.start);
          await endInput.clear();
          if (timeTest.end) await endInput.fill(timeTest.end);
          await page.waitForTimeout(300);
          
          // バリデーションチェック
          const hasError = await page.locator('[class*="error"]').isVisible();
          console.log(`時間設定 ${timeTest.start}-${timeTest.end}: ${hasError ? 'エラー検出' : 'エラーなし'}`);
        }
      }
    });
  });

  test.describe('同時実行と競合状態のテスト', () => {
    test('複数タブでの同時操作', async ({ page, context }) => {
      console.log('🔀 同時操作テスト');
      
      // 複数のタブを開く
      const page2 = await context.newPage();
      await page2.goto('/');
      await page2.waitForTimeout(2000);
      
      // 両方のタブで在庫管理を開く
      await page.click('text=在庫管理');
      await page2.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // 同時に異なる操作を実行
      await Promise.all([
        page.click('text=EOQ分析'),
        page2.click('text=シミュレーション')
      ]);
      await page.waitForTimeout(2000);
      
      // 両方でデータ入力
      await Promise.all([
        page.fill('input[label*="発注コスト"]', '1000'),
        page2.fill('input[label*="サンプル数"]', '500')
      ]);
      await page.waitForTimeout(1000);
      
      // 同時実行
      await Promise.all([
        page.click('button:has-text("計算")'),
        page2.click('button:has-text("シミュレーション実行")')
      ]);
      await page.waitForTimeout(3000);
      
      // 結果の確認
      const page1HasResult = await page.locator('text=最適発注量, canvas').isVisible();
      const page2HasResult = await page2.locator('text=シミュレーション結果, canvas').isVisible();
      
      console.log(`タブ1結果: ${page1HasResult}, タブ2結果: ${page2HasResult}`);
      
      await page2.close();
    });

    test('高速連続クリックでの安定性テスト', async ({ page }) => {
      console.log('⚡ 高速連続クリックテスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // タブの高速切り替え
      const tabs = ['EOQ分析', 'シミュレーション', '多段階在庫', '最適化'];
      
      for (let i = 0; i < 3; i++) {
        for (const tab of tabs) {
          await page.click(`text=${tab}`);
          await page.waitForTimeout(100); // 短い待機時間
        }
      }
      
      // ボタンの連続クリック防止テスト
      const calcButton = page.locator('button:has-text("計算")').first();
      if (await calcButton.isVisible()) {
        // 5回連続クリック
        for (let i = 0; i < 5; i++) {
          await calcButton.click();
          await page.waitForTimeout(50);
        }
        
        // 複数の処理が走っていないか確認
        await page.waitForTimeout(2000);
        const loadingCount = await page.locator('[class*="loading"], [class*="spinner"]').count();
        console.log(`同時実行中の処理数: ${loadingCount}`);
      }
    });
  });

  test.describe('ブラウザ互換性とレンダリングエッジケース', () => {
    test('異なるズームレベルでの表示確認', async ({ page }) => {
      console.log('🔍 ズームレベル互換性テスト');
      
      const zoomLevels = [0.5, 0.75, 1, 1.25, 1.5, 2];
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      for (const zoom of zoomLevels) {
        await page.evaluate((z) => {
          document.body.style.zoom = z.toString();
        }, zoom);
        await page.waitForTimeout(1000);
        
        // 主要要素の表示確認
        const tabsVisible = await page.locator('[role="tab"]').first().isVisible();
        const contentVisible = await page.locator('div[role="tabpanel"]').first().isVisible();
        
        console.log(`ズーム ${zoom * 100}%: タブ=${tabsVisible}, コンテンツ=${contentVisible}`);
        
        // スクリーンショット保存
        await page.screenshot({ 
          path: `test-results/zoom-${zoom * 100}.png`,
          fullPage: false 
        });
      }
      
      // ズームをリセット
      await page.evaluate(() => {
        document.body.style.zoom = '1';
      });
    });

    test('長時間実行後のメモリリークテスト', async ({ page }) => {
      console.log('💾 メモリリークテスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // 初期メモリ使用量を記録
      const initialMemory = await page.evaluate(() => {
        if (performance.memory) {
          return performance.memory.usedJSHeapSize;
        }
        return 0;
      });
      
      // 繰り返し操作を実行
      for (let i = 0; i < 10; i++) {
        // タブ切り替え
        await page.click('text=シミュレーション');
        await page.waitForTimeout(500);
        
        // データ入力
        await page.fill('input[label*="サンプル数"]', '1000');
        await page.waitForTimeout(200);
        
        // 実行
        const simButton = page.locator('button:has-text("シミュレーション実行")').first();
        if (await simButton.isVisible()) {
          await simButton.click();
          await page.waitForTimeout(2000);
        }
        
        // 別のタブに切り替え（メモリ解放のトリガー）
        await page.click('text=EOQ分析');
        await page.waitForTimeout(500);
      }
      
      // 最終メモリ使用量を記録
      const finalMemory = await page.evaluate(() => {
        if (performance.memory) {
          return performance.memory.usedJSHeapSize;
        }
        return 0;
      });
      
      const memoryIncrease = (finalMemory - initialMemory) / 1048576; // MB換算
      console.log(`メモリ増加量: ${memoryIncrease.toFixed(2)} MB`);
      
      if (memoryIncrease > 100) {
        console.log('⚠️ 潜在的なメモリリークの可能性');
      } else {
        console.log('✅ メモリ使用量は正常範囲内');
      }
    });
  });

  test.describe('データ整合性とバリデーションエッジケース', () => {
    test('不正なCSVファイルのアップロード処理', async ({ page }) => {
      console.log('📄 不正ファイルアップロードテスト');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      await page.click('text=データ管理');
      await page.waitForTimeout(1000);
      
      const fileInput = page.locator('input[type="file"]').first();
      if (await fileInput.isVisible()) {
        // 様々な不正なファイルをテスト
        const invalidFiles = [
          {
            name: 'empty.csv',
            content: '',
            description: '空ファイル'
          },
          {
            name: 'invalid-encoding.csv',
            content: '\xFF\xFE不正なエンコーディング',
            description: '不正なエンコーディング'
          },
          {
            name: 'huge-file.csv',
            content: 'x'.repeat(10 * 1024 * 1024), // 10MB
            description: '巨大ファイル'
          },
          {
            name: 'malformed.csv',
            content: 'header1,header2,header3\nvalue1,value2\nvalue3,value4,value5,value6',
            description: '不整合なCSV'
          },
          {
            name: 'script-injection.csv',
            content: '=cmd|"/c calc"!A1,<script>alert("XSS")</script>,${alert("XSS")}',
            description: 'スクリプトインジェクション'
          }
        ];
        
        for (const file of invalidFiles) {
          console.log(`テスト: ${file.description}`);
          
          await fileInput.setInputFiles({
            name: file.name,
            mimeType: 'text/csv',
            buffer: Buffer.from(file.content)
          });
          await page.waitForTimeout(2000);
          
          // エラー処理の確認
          const hasError = await page.locator('text=エラー, text=無効, text=失敗, [class*="error"]').isVisible();
          const hasWarning = await page.locator('text=警告, text=注意, [class*="warning"]').isVisible();
          
          console.log(`${file.description}: エラー=${hasError}, 警告=${hasWarning}`);
          
          // エラーダイアログをクリア
          const dismissButton = page.locator('button:has-text("閉じる"), button:has-text("OK"), button:has-text("キャンセル")').first();
          if (await dismissButton.isVisible()) {
            await dismissButton.click();
            await page.waitForTimeout(500);
          }
        }
      }
    });

    test('循環参照と無限ループの防止テスト', async ({ page }) => {
      console.log('♾️ 循環参照防止テスト');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      
      // 同一座標での複数施設設定
      const coordInputs = page.locator('input[type="number"]');
      const inputCount = await coordInputs.count();
      
      if (inputCount >= 4) {
        // 全て同じ座標に設定
        for (let i = 0; i < inputCount; i++) {
          await coordInputs.nth(i).fill('100');
          await page.waitForTimeout(100);
        }
        
        // 最適化実行
        await page.click('button:has-text("計算"), button:has-text("最適化")');
        await page.waitForTimeout(3000);
        
        // 無限ループに陥っていないか確認
        const isStillProcessing = await page.locator('[class*="loading"], text=処理中').isVisible();
        if (!isStillProcessing) {
          console.log('✅ 循環参照が適切に処理されました');
        } else {
          console.log('⚠️ 処理が終了しません');
        }
      }
    });
  });

  test.describe('国際化とローカライゼーションエッジケース', () => {
    test('多言語文字と特殊文字の処理', async ({ page }) => {
      console.log('🌍 多言語文字処理テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      const specialStrings = [
        '製品名テスト123',           // 日本語英数字混在
        '🚀📦💰',                  // 絵文字
        'Tëst Ñämé',              // アクセント文字
        '测试产品',                 // 中国語
        'מוצר בדיקה',             // ヘブライ語（RTL）
        'مُنتَج اختبار',           // アラビア語（RTL）
        '<script>alert("XSS")</script>', // HTMLタグ
        'Product & Co.',          // 特殊文字
        '\\n\\r\\t',             // エスケープシーケンス
        ''                       // 空文字列
      ];
      
      for (const str of specialStrings) {
        const input = page.locator('input[type="text"]').first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(str);
          await page.waitForTimeout(300);
          
          // 入力値が正しく処理されているか確認
          const value = await input.inputValue();
          const isCorrect = value === str || value === ''; // XSS対策で除去される場合もOK
          
          console.log(`"${str}": ${isCorrect ? '正常' : '異常'}`);
        }
      }
    });

    test('日付・時刻・数値フォーマットの国際化', async ({ page }) => {
      console.log('📅 日付時刻フォーマットテスト');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      // 様々な日付フォーマットをテスト
      const dateFormats = [
        '2024-01-15',      // ISO形式
        '2024/01/15',      // スラッシュ区切り
        '15/01/2024',      // 日/月/年
        '01-15-2024',      // 月-日-年
        '令和6年1月15日',   // 和暦
        'invalid-date'     // 無効な日付
      ];
      
      const dateInput = page.locator('input[type="date"]').first();
      if (await dateInput.isVisible()) {
        for (const dateStr of dateFormats) {
          await dateInput.clear();
          await dateInput.fill(dateStr);
          await page.waitForTimeout(300);
          
          const value = await dateInput.inputValue();
          console.log(`日付入力 "${dateStr}": 結果="${value}"`);
        }
      }
      
      // 数値フォーマットテスト
      const numberFormats = [
        '1,234.56',        // カンマ区切り
        '1.234,56',        // ヨーロッパ形式
        '１２３４',         // 全角数字
        '1.23e5',          // 指数表記
        '￥1,234',         // 通貨記号付き
        'NaN'              // 非数値
      ];
      
      const numberInput = page.locator('input[type="number"]').first();
      if (await numberInput.isVisible()) {
        for (const numStr of numberFormats) {
          await numberInput.clear();
          await numberInput.fill(numStr);
          await page.waitForTimeout(300);
          
          const value = await numberInput.inputValue();
          console.log(`数値入力 "${numStr}": 結果="${value}"`);
        }
      }
    });
  });

  test.describe('ネットワークエラーと接続障害のシミュレーション', () => {
    test('オフライン状態での動作確認', async ({ page, context }) => {
      console.log('📡 オフライン動作テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // ネットワークをオフラインに設定
      await context.setOffline(true);
      
      // オフライン状態での操作
      await page.click('text=EOQ分析');
      await page.waitForTimeout(1000);
      
      await page.fill('input[label*="発注コスト"]', '1000');
      await page.fill('input[label*="需要量"]', '500');
      await page.waitForTimeout(500);
      
      await page.click('button:has-text("計算")');
      await page.waitForTimeout(2000);
      
      // オフライン通知の確認
      const offlineNotice = await page.locator('text=オフライン, text=接続なし, text=offline, [class*="offline"]').isVisible();
      console.log(`オフライン通知: ${offlineNotice ? '表示' : '非表示'}`);
      
      // オンラインに復帰
      await context.setOffline(false);
      await page.waitForTimeout(2000);
      
      // 再接続後の動作確認
      await page.click('button:has-text("計算")');
      await page.waitForTimeout(2000);
      
      const hasResult = await page.locator('text=最適発注量, canvas').isVisible();
      console.log(`オンライン復帰後の計算: ${hasResult ? '成功' : '失敗'}`);
    });

    test('遅延ネットワークでのタイムアウト処理', async ({ page, context }) => {
      console.log('🐌 低速ネットワークテスト');
      
      // ネットワーク速度を制限（3G相当）
      await context.route('**/*', async route => {
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1秒遅延
        await route.continue();
      });
      
      await page.click('text=PyVRP');
      await page.waitForTimeout(3000);
      
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      const startTime = Date.now();
      await page.click('button:has-text("最適化実行")');
      
      // タイムアウトまたは完了を待つ
      const result = await Promise.race([
        page.waitForSelector('text=最適化完了', { timeout: 10000 }).then(() => 'completed'),
        page.waitForSelector('text=タイムアウト, text=timeout', { timeout: 10000 }).then(() => 'timeout'),
        page.waitForTimeout(10000).then(() => 'no-response')
      ]);
      
      const elapsedTime = Date.now() - startTime;
      console.log(`低速ネットワークでの結果: ${result} (${elapsedTime}ms)`);
    });
  });

  test.describe('ストレージとキャッシュのエッジケース', () => {
    test('ローカルストレージ容量制限テスト', async ({ page }) => {
      console.log('💾 ストレージ容量制限テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // 大量のデータをローカルストレージに保存
      const result = await page.evaluate(() => {
        try {
          const largeData = 'x'.repeat(1024 * 1024); // 1MB
          for (let i = 0; i < 10; i++) {
            localStorage.setItem(`test-data-${i}`, largeData);
          }
          return 'success';
        } catch (e) {
          return e.name;
        }
      });
      
      console.log(`ストレージ保存結果: ${result}`);
      
      if (result === 'QuotaExceededError') {
        console.log('✅ ストレージ容量制限が正しく機能しています');
        
        // エラー後の動作確認
        await page.reload();
        await page.waitForTimeout(2000);
        
        const appLoaded = await page.locator('text=在庫管理').isVisible();
        console.log(`容量超過後のアプリ起動: ${appLoaded ? '成功' : '失敗'}`);
      }
      
      // クリーンアップ
      await page.evaluate(() => {
        localStorage.clear();
      });
    });

    test('ブラウザキャッシュ無効化時の動作', async ({ page, context }) => {
      console.log('🚫 キャッシュ無効化テスト');
      
      // キャッシュを無効化
      await context.route('**/*', route => {
        route.continue({
          headers: {
            ...route.request().headers(),
            'cache-control': 'no-cache, no-store, must-revalidate',
            'pragma': 'no-cache',
            'expires': '0'
          }
        });
      });
      
      const startTime = Date.now();
      await page.reload();
      await page.waitForLoadState('domcontentloaded');
      const loadTime1 = Date.now() - startTime;
      
      // 2回目のリロード（キャッシュなし）
      const startTime2 = Date.now();
      await page.reload();
      await page.waitForLoadState('domcontentloaded');
      const loadTime2 = Date.now() - startTime2;
      
      console.log(`初回ロード時間: ${loadTime1}ms`);
      console.log(`2回目ロード時間: ${loadTime2}ms`);
      console.log(`キャッシュ効果: ${loadTime1 - loadTime2}ms`);
    });
  });

  test('ブラウザ拡張機能との互換性テスト', async ({ page }) => {
    console.log('🔧 ブラウザ拡張機能互換性テスト');
    
    // 広告ブロッカーのシミュレーション
    await page.route('**/*analytics*', route => route.abort());
    await page.route('**/*tracking*', route => route.abort());
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // アプリケーションが正常に動作するか確認
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    const isAppFunctional = await page.locator('[role="tab"]').count() > 0;
    console.log(`広告ブロッカー環境での動作: ${isAppFunctional ? '正常' : '異常'}`);
    
    // パスワードマネージャーのシミュレーション
    await page.evaluate(() => {
      // パスワードマネージャーが入力フィールドに自動入力
      const inputs = document.querySelectorAll('input');
      inputs.forEach(input => {
        if (input.type === 'text' || input.type === 'number') {
          input.value = 'AUTO_FILLED_VALUE';
          input.dispatchEvent(new Event('input', { bubbles: true }));
        }
      });
    });
    
    await page.waitForTimeout(1000);
    
    // 自動入力後も正常に動作するか確認
    await page.click('button:has-text("計算")');
    await page.waitForTimeout(2000);
    
    const hasError = await page.locator('[class*="error"]').isVisible();
    console.log(`自動入力後の動作: ${hasError ? 'エラーあり' : '正常'}`);
  });
});