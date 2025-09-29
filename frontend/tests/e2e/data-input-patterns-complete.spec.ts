import { test, expect } from '@playwright/test';

test.describe('Complete Data Input Patterns and Validation Tests', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(90000);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
  });

  test.describe('在庫管理 - 全データ入力パターン', () => {
    test('EOQ分析の全パラメータ組み合わせテスト', async ({ page }) => {
      console.log('📊 EOQ分析 - 全パラメータ組み合わせテスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      await page.click('text=EOQ分析');
      await page.waitForTimeout(1000);
      
      // パラメータの組み合わせパターン
      const parameterCombinations = [
        // 通常ケース
        { K: '100', d: '1000', h: '0.2', b: '0.1', expected: 'valid' },
        // 極小値
        { K: '0.01', d: '1', h: '0.001', b: '0.001', expected: 'valid' },
        // 極大値
        { K: '999999', d: '999999', h: '0.99', b: '0.99', expected: 'valid' },
        // ゼロ値
        { K: '0', d: '1000', h: '0.2', b: '0.1', expected: 'error' },
        { K: '100', d: '0', h: '0.2', b: '0.1', expected: 'error' },
        // 負の値
        { K: '-100', d: '1000', h: '0.2', b: '0.1', expected: 'error' },
        // 小数点
        { K: '100.5', d: '1000.75', h: '0.2', b: '0.1', expected: 'valid' },
        // 指数表記
        { K: '1e2', d: '1e3', h: '2e-1', b: '1e-1', expected: 'valid' },
        // 特殊ケース
        { K: 'NaN', d: 'Infinity', h: '-Infinity', b: 'undefined', expected: 'error' }
      ];
      
      for (const combo of parameterCombinations) {
        console.log(`テスト: K=${combo.K}, d=${combo.d}, h=${combo.h}, b=${combo.b}`);
        
        // パラメータ入力
        await page.fill('input[label*="発注コスト"]', combo.K);
        await page.fill('input[label*="需要量"]', combo.d);
        await page.fill('input[label*="保管費率"]', combo.h);
        await page.fill('input[label*="欠品費率"]', combo.b);
        await page.waitForTimeout(500);
        
        // 計算実行
        await page.click('button:has-text("計算")');
        await page.waitForTimeout(2000);
        
        // 結果確認
        const hasError = await page.locator('[class*="error"], text=エラー').isVisible();
        const hasResult = await page.locator('text=最適発注量, canvas').isVisible();
        
        const actual = hasError ? 'error' : hasResult ? 'valid' : 'unknown';
        console.log(`期待値: ${combo.expected}, 実際: ${actual}, 一致: ${actual === combo.expected ? '✅' : '❌'}`);
        
        // エラーダイアログをクリア
        if (hasError) {
          const closeBtn = page.locator('button:has-text("閉じる"), button:has-text("OK")').first();
          if (await closeBtn.isVisible()) {
            await closeBtn.click();
            await page.waitForTimeout(500);
          }
        }
      }
    });

    test('シミュレーション - 確率分布とパラメータの全組み合わせ', async ({ page }) => {
      console.log('🎲 シミュレーション - 確率分布テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      await page.click('text=シミュレーション');
      await page.waitForTimeout(1000);
      
      // 確率分布の種類
      const distributions = [
        { type: 'normal', params: { mean: '100', std: '20' } },
        { type: 'poisson', params: { lambda: '50' } },
        { type: 'exponential', params: { rate: '0.1' } },
        { type: 'uniform', params: { min: '50', max: '150' } },
        { type: 'gamma', params: { shape: '2', scale: '50' } }
      ];
      
      for (const dist of distributions) {
        const distSelect = page.locator('select[label*="分布"]').first();
        if (await distSelect.isVisible()) {
          await distSelect.selectOption(dist.type);
          await page.waitForTimeout(500);
          
          // 分布固有のパラメータ入力
          for (const [param, value] of Object.entries(dist.params)) {
            const input = page.locator(`input[label*="${param}"]`).first();
            if (await input.isVisible()) {
              await input.fill(value);
              await page.waitForTimeout(300);
            }
          }
          
          // ポリシー設定
          const policies = ['(Q,R)', '(s,S)', '基準在庫', '定期発注'];
          for (const policy of policies) {
            const policyRadio = page.locator(`input[type="radio"][value*="${policy}"]`).first();
            if (await policyRadio.isVisible()) {
              await policyRadio.check();
              await page.waitForTimeout(300);
            }
          }
          
          // シミュレーション実行
          await page.click('button:has-text("シミュレーション実行")');
          await page.waitForTimeout(3000);
          
          console.log(`${dist.type}分布でのシミュレーション完了`);
        }
      }
    });

    test('多段階在庫 - ネットワーク構成の全パターン', async ({ page }) => {
      console.log('🏭 多段階在庫 - ネットワーク構成テスト');
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      await page.click('text=多段階在庫');
      await page.waitForTimeout(1000);
      
      // ネットワーク構成パターン
      const networkConfigs = [
        { factories: 1, dcs: 1, retailers: 1 }, // 最小構成
        { factories: 1, dcs: 3, retailers: 10 }, // 標準構成
        { factories: 3, dcs: 5, retailers: 50 }, // 大規模構成
        { factories: 1, dcs: 0, retailers: 20 }, // DC無し（直送）
        { factories: 5, dcs: 10, retailers: 100 } // 超大規模
      ];
      
      for (const config of networkConfigs) {
        console.log(`構成: 工場${config.factories}, DC${config.dcs}, 小売${config.retailers}`);
        
        await page.fill('input[label*="工場数"]', config.factories.toString());
        await page.fill('input[label*="DC数"]', config.dcs.toString());
        await page.fill('input[label*="小売店数"]', config.retailers.toString());
        await page.waitForTimeout(500);
        
        // リードタイム設定
        const leadTimes = ['固定', '確率的', '距離依存'];
        for (const ltType of leadTimes) {
          const ltRadio = page.locator(`input[type="radio"][value*="${ltType}"]`).first();
          if (await ltRadio.isVisible()) {
            await ltRadio.check();
            await page.waitForTimeout(300);
          }
        }
        
        // 分析実行
        await page.click('button:has-text("分析実行")');
        await page.waitForTimeout(3000);
        
        // ネットワーク可視化の確認
        const hasVisualization = await page.locator('canvas, svg[class*="network"]').isVisible();
        console.log(`ネットワーク可視化: ${hasVisualization ? '表示' : '非表示'}`);
      }
    });
  });

  test.describe('物流ネットワーク設計 - 全データ入力パターン', () => {
    test('顧客データの全フォーマットテスト', async ({ page }) => {
      console.log('🗺️ 顧客データフォーマットテスト');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      await page.click('text=データ管理');
      await page.waitForTimeout(1000);
      
      // 様々なデータフォーマット
      const dataFormats = [
        {
          name: '標準CSV',
          content: 'customer_id,latitude,longitude,demand\nC001,35.6762,139.6503,100\nC002,34.6937,135.5023,150',
          type: 'text/csv'
        },
        {
          name: 'タブ区切り',
          content: 'customer_id\tlatitude\tlongitude\tdemand\nC001\t35.6762\t139.6503\t100',
          type: 'text/tab-separated-values'
        },
        {
          name: '日本語ヘッダー',
          content: '顧客ID,緯度,経度,需要量\nC001,35.6762,139.6503,100',
          type: 'text/csv'
        },
        {
          name: '追加属性付き',
          content: 'id,lat,lon,demand,priority,time_window_start,time_window_end\nC001,35.6762,139.6503,100,high,09:00,17:00',
          type: 'text/csv'
        }
      ];
      
      for (const format of dataFormats) {
        console.log(`テスト: ${format.name}`);
        
        const fileInput = page.locator('input[type="file"]').first();
        if (await fileInput.isVisible()) {
          await fileInput.setInputFiles({
            name: `test-${format.name}.csv`,
            mimeType: format.type,
            buffer: Buffer.from(format.content)
          });
          await page.waitForTimeout(2000);
          
          // インポート結果の確認
          const hasSuccess = await page.locator('text=成功, text=読み込み完了').isVisible();
          const hasError = await page.locator('text=エラー, text=失敗').isVisible();
          
          console.log(`${format.name}: ${hasSuccess ? '成功' : hasError ? 'エラー' : '不明'}`);
        }
      }
    });

    test('立地モデルパラメータの網羅的テスト', async ({ page }) => {
      console.log('📍 立地モデルパラメータテスト');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      
      // Weiszfeld法の収束条件テスト
      const convergenceTests = [
        { maxIter: '1', tolerance: '0.1', expected: 'poor' },
        { maxIter: '100', tolerance: '1e-6', expected: 'good' },
        { maxIter: '1000', tolerance: '1e-10', expected: 'good' },
        { maxIter: '10000', tolerance: '0', expected: 'timeout' }
      ];
      
      for (const test of convergenceTests) {
        await page.fill('input[label*="最大反復回数"]', test.maxIter);
        await page.fill('input[label*="収束許容誤差"]', test.tolerance);
        await page.waitForTimeout(500);
        
        const startTime = Date.now();
        await page.click('button:has-text("計算")');
        await page.waitForTimeout(5000);
        
        const elapsedTime = Date.now() - startTime;
        const hasResult = await page.locator('text=最適立地, canvas').isVisible();
        
        console.log(`反復${test.maxIter}, 誤差${test.tolerance}: ${elapsedTime}ms, 結果=${hasResult}`);
      }
      
      // K-Medianのクラスタ数テスト
      const kValues = [1, 2, 3, 5, 10, 20, 50, 100];
      
      for (const k of kValues) {
        await page.fill('input[label*="K値"]', k.toString());
        await page.waitForTimeout(300);
        
        await page.click('button:has-text("K-Median")');
        await page.waitForTimeout(3000);
        
        // クラスタリング結果の確認
        const hasVisualization = await page.locator('canvas, svg').isVisible();
        console.log(`K=${k}: 可視化=${hasVisualization}`);
      }
    });
  });

  test.describe('シフト最適化 - 全制約条件テスト', () => {
    test('複雑なシフト制約の組み合わせテスト', async ({ page }) => {
      console.log('📅 複雑シフト制約テスト');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      // 制約条件の組み合わせ
      const constraints = [
        {
          name: '基本制約',
          settings: {
            '最小連続勤務': '3',
            '最大連続勤務': '5',
            '週間最大勤務': '40',
            '休憩時間': '60'
          }
        },
        {
          name: '厳格制約',
          settings: {
            '最小連続勤務': '1',
            '最大連続勤務': '2',
            '週間最大勤務': '20',
            '休憩時間': '90'
          }
        },
        {
          name: 'フレキシブル制約',
          settings: {
            '最小連続勤務': '1',
            '最大連続勤務': '7',
            '週間最大勤務': '60',
            '休憩時間': '30'
          }
        }
      ];
      
      await page.click('text=シフト要件');
      await page.waitForTimeout(1000);
      
      for (const constraint of constraints) {
        console.log(`テスト: ${constraint.name}`);
        
        for (const [label, value] of Object.entries(constraint.settings)) {
          const input = page.locator(`input[label*="${label}"]`).first();
          if (await input.isVisible()) {
            await input.fill(value);
            await page.waitForTimeout(200);
          }
        }
        
        // スキル制約の追加
        const skillCheckboxes = page.locator('input[type="checkbox"][label*="スキル"]');
        const skillCount = await skillCheckboxes.count();
        
        for (let i = 0; i < Math.min(skillCount, 3); i++) {
          await skillCheckboxes.nth(i).check();
          await page.waitForTimeout(200);
        }
        
        // 最適化実行
        await page.click('text=最適化実行');
        await page.waitForTimeout(1000);
        
        await page.click('button:has-text("最適化")');
        await page.waitForTimeout(5000);
        
        // 実行可能性の確認
        await page.click('text=リアルタイム監視');
        await page.waitForTimeout(1000);
        
        await page.click('button:has-text("実行可能性分析")');
        await page.waitForTimeout(2000);
        
        const feasibility = await page.locator('text=実行可能, text=違反').isVisible();
        console.log(`${constraint.name}: 実行可能性=${feasibility}`);
      }
    });

    test('時間帯別需要パターンの全テスト', async ({ page }) => {
      console.log('⏰ 時間帯別需要パターンテスト');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      await page.click('text=シフト要件');
      await page.waitForTimeout(1000);
      
      // 需要パターン
      const demandPatterns = [
        {
          name: '平日パターン',
          hourly: [2, 2, 1, 1, 2, 3, 5, 8, 6, 5, 5, 6, 8, 6, 5, 4, 5, 7, 6, 4, 3, 2, 2, 2]
        },
        {
          name: '週末パターン',
          hourly: [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 8, 10, 10, 9, 8, 7, 8, 9, 8, 6, 4, 3, 2, 1]
        },
        {
          name: 'イベント日',
          hourly: [3, 2, 2, 2, 3, 5, 8, 12, 15, 15, 12, 15, 15, 12, 10, 8, 10, 12, 10, 8, 6, 4, 3, 2]
        }
      ];
      
      for (const pattern of demandPatterns) {
        console.log(`パターン: ${pattern.name}`);
        
        // 時間帯別需要入力
        const hourInputs = page.locator('input[label*="時需要"]');
        const inputCount = await hourInputs.count();
        
        for (let hour = 0; hour < Math.min(inputCount, 24); hour++) {
          if (hour < pattern.hourly.length) {
            await hourInputs.nth(hour).fill(pattern.hourly[hour].toString());
            await page.waitForTimeout(100);
          }
        }
        
        // パターン保存
        const savePatternBtn = page.locator('button:has-text("パターン保存")').first();
        if (await savePatternBtn.isVisible()) {
          await savePatternBtn.click();
          await page.waitForTimeout(1000);
        }
      }
    });
  });

  test.describe('PyVRP - 全車両・制約設定テスト', () => {
    test('異種車両フリートの設定テスト', async ({ page }) => {
      console.log('🚛 異種車両フリートテスト');
      
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      await page.click('text=車両設定');
      await page.waitForTimeout(1000);
      
      // 車両タイプ
      const vehicleTypes = [
        { type: '軽トラック', capacity: '350', maxDist: '100', cost: '5000', count: '5' },
        { type: '2tトラック', capacity: '2000', maxDist: '200', cost: '8000', count: '3' },
        { type: '4tトラック', capacity: '4000', maxDist: '300', cost: '12000', count: '2' },
        { type: '10tトラック', capacity: '10000', maxDist: '500', cost: '20000', count: '1' },
        { type: '冷蔵車', capacity: '1500', maxDist: '150', cost: '15000', count: '2' }
      ];
      
      for (let i = 0; i < vehicleTypes.length; i++) {
        const vehicle = vehicleTypes[i];
        console.log(`車両タイプ: ${vehicle.type}`);
        
        // 新しい車両タイプを追加
        const addVehicleBtn = page.locator('button:has-text("車両追加")').first();
        if (await addVehicleBtn.isVisible()) {
          await addVehicleBtn.click();
          await page.waitForTimeout(500);
        }
        
        // 車両パラメータ入力
        const row = i; // 行インデックス
        await page.fill(`input[name="vehicle[${row}].capacity"]`, vehicle.capacity);
        await page.fill(`input[name="vehicle[${row}].maxDistance"]`, vehicle.maxDist);
        await page.fill(`input[name="vehicle[${row}].cost"]`, vehicle.cost);
        await page.fill(`input[name="vehicle[${row}].count"]`, vehicle.count);
        await page.waitForTimeout(300);
        
        // 特殊制約の設定
        if (vehicle.type === '冷蔵車') {
          const tempCheckbox = page.locator(`input[type="checkbox"][name="vehicle[${row}].requiresCooling"]`).first();
          if (await tempCheckbox.isVisible()) {
            await tempCheckbox.check();
          }
        }
      }
      
      // 車両割当最適化
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(5000);
      
      // 車両利用率の確認
      const utilizationInfo = await page.locator('text=車両利用率, text=使用車両').isVisible();
      console.log(`車両利用率情報: ${utilizationInfo ? '表示' : '非表示'}`);
    });

    test('複雑な時間窓制約のテスト', async ({ page }) => {
      console.log('🕐 複雑時間窓制約テスト');
      
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      await page.click('text=制約設定');
      await page.waitForTimeout(1000);
      
      // 時間窓パターン
      const timeWindowPatterns = [
        {
          name: '午前配送',
          customers: 30,
          windows: [{ start: '08:00', end: '12:00', service: '15' }]
        },
        {
          name: '複数時間窓',
          customers: 20,
          windows: [
            { start: '09:00', end: '11:00', service: '20' },
            { start: '14:00', end: '16:00', service: '20' }
          ]
        },
        {
          name: '夜間配送',
          customers: 15,
          windows: [{ start: '20:00', end: '06:00', service: '30' }]
        },
        {
          name: 'タイトな時間窓',
          customers: 10,
          windows: [{ start: '10:00', end: '10:30', service: '5' }]
        }
      ];
      
      for (const pattern of timeWindowPatterns) {
        console.log(`パターン: ${pattern.name}`);
        
        // 顧客数設定
        await page.fill('input[label*="顧客数"]', pattern.customers.toString());
        await page.waitForTimeout(300);
        
        // 時間窓設定
        for (let i = 0; i < pattern.windows.length; i++) {
          const window = pattern.windows[i];
          
          await page.fill(`input[name="timeWindow[${i}].start"]`, window.start);
          await page.fill(`input[name="timeWindow[${i}].end"]`, window.end);
          await page.fill(`input[name="timeWindow[${i}].serviceTime"]`, window.service);
          await page.waitForTimeout(300);
        }
        
        // ソフト時間窓のペナルティ設定
        await page.fill('input[label*="早着ペナルティ"]', '100');
        await page.fill('input[label*="遅着ペナルティ"]', '500');
        await page.waitForTimeout(300);
        
        // 実行可能性チェック
        const checkBtn = page.locator('button:has-text("実行可能性チェック")').first();
        if (await checkBtn.isVisible()) {
          await checkBtn.click();
          await page.waitForTimeout(2000);
          
          const isFeasible = await page.locator('text=実行可能, text=Feasible').isVisible();
          console.log(`${pattern.name}: 実行可能=${isFeasible}`);
        }
      }
    });
  });

  test.describe('分析システム - 全データ分析パターン', () => {
    test('統計分析の全手法テスト', async ({ page }) => {
      console.log('📈 統計分析全手法テスト');
      
      await page.click('text=分析');
      await page.waitForTimeout(2000);
      await page.click('text=統計分析');
      await page.waitForTimeout(1000);
      
      // 統計手法
      const statisticalMethods = [
        {
          method: '記述統計',
          params: { groupBy: 'category', metrics: ['mean', 'median', 'std', 'skewness', 'kurtosis'] }
        },
        {
          method: '相関分析',
          params: { method: 'pearson', threshold: '0.7' }
        },
        {
          method: '回帰分析',
          params: { type: 'multiple', variables: ['price', 'promotion', 'season'], regularization: 'ridge' }
        },
        {
          method: '時系列分析',
          params: { decomposition: 'seasonal', period: '12', smoothing: 'holt-winters' }
        },
        {
          method: '分散分析',
          params: { type: 'two-way', factors: ['region', 'product'], interaction: true }
        },
        {
          method: '主成分分析',
          params: { components: '3', scaling: 'standardize' }
        }
      ];
      
      for (const analysis of statisticalMethods) {
        console.log(`手法: ${analysis.method}`);
        
        const methodSelect = page.locator('select[label*="分析手法"]').first();
        if (await methodSelect.isVisible()) {
          await methodSelect.selectOption(analysis.method);
          await page.waitForTimeout(500);
          
          // パラメータ設定
          for (const [param, value] of Object.entries(analysis.params)) {
            if (Array.isArray(value)) {
              // 複数選択
              for (const v of value) {
                const checkbox = page.locator(`input[type="checkbox"][value="${v}"]`).first();
                if (await checkbox.isVisible()) {
                  await checkbox.check();
                  await page.waitForTimeout(200);
                }
              }
            } else {
              // 単一値入力
              const input = page.locator(`input[label*="${param}"], select[label*="${param}"]`).first();
              if (await input.isVisible()) {
                if (input.type === 'select') {
                  await input.selectOption(value);
                } else {
                  await input.fill(value);
                }
                await page.waitForTimeout(200);
              }
            }
          }
          
          // 分析実行
          await page.click('button:has-text("分析実行")');
          await page.waitForTimeout(3000);
          
          // 結果の確認
          const hasResults = await page.locator('text=分析結果, canvas, table[class*="results"]').isVisible();
          console.log(`${analysis.method}: 結果=${hasResults}`);
        }
      }
    });

    test('予測モデルの全アルゴリズムテスト', async ({ page }) => {
      console.log('🔮 予測モデル全アルゴリズムテスト');
      
      await page.click('text=分析');
      await page.waitForTimeout(2000);
      await page.click('text=予測分析');
      await page.waitForTimeout(1000);
      
      // 予測アルゴリズム
      const algorithms = [
        {
          name: 'ARIMA',
          params: { p: '1', d: '1', q: '1', seasonal: true, P: '1', D: '1', Q: '1', s: '12' }
        },
        {
          name: 'Prophet',
          params: { changepoint_prior_scale: '0.05', seasonality_mode: 'multiplicative', holidays: true }
        },
        {
          name: 'LSTM',
          params: { layers: '2', units: '50', dropout: '0.2', epochs: '100', batch_size: '32' }
        },
        {
          name: 'XGBoost',
          params: { max_depth: '6', learning_rate: '0.1', n_estimators: '100', subsample: '0.8' }
        },
        {
          name: 'Random Forest',
          params: { n_trees: '100', max_features: 'sqrt', min_samples_split: '5' }
        },
        {
          name: 'Exponential Smoothing',
          params: { trend: 'add', seasonal: 'mul', seasonal_periods: '12' }
        }
      ];
      
      for (const algo of algorithms) {
        console.log(`アルゴリズム: ${algo.name}`);
        
        const algoSelect = page.locator('select[label*="予測手法"]').first();
        if (await algoSelect.isVisible()) {
          await algoSelect.selectOption(algo.name);
          await page.waitForTimeout(500);
          
          // ハイパーパラメータ設定
          for (const [param, value] of Object.entries(algo.params)) {
            const input = page.locator(`input[label*="${param}"]`).first();
            if (await input.isVisible()) {
              if (typeof value === 'boolean') {
                const checkbox = page.locator(`input[type="checkbox"][label*="${param}"]`).first();
                if (await checkbox.isVisible()) {
                  if (value) await checkbox.check();
                  else await checkbox.uncheck();
                }
              } else {
                await input.fill(value.toString());
              }
              await page.waitForTimeout(200);
            }
          }
          
          // 交差検証設定
          await page.fill('input[label*="検証分割"]', '5');
          await page.fill('input[label*="テストサイズ"]', '0.2');
          await page.waitForTimeout(300);
          
          // モデル訓練
          await page.click('button:has-text("モデル訓練")');
          await page.waitForTimeout(5000);
          
          // 評価指標の確認
          const metrics = ['RMSE', 'MAE', 'MAPE', 'R²'];
          for (const metric of metrics) {
            const metricValue = await page.locator(`text=${metric}:`).isVisible();
            if (metricValue) {
              console.log(`${algo.name} - ${metric}: 表示`);
            }
          }
        }
      }
    });
  });

  test('全システム間のデータ形式互換性テスト', async ({ page }) => {
    console.log('🔄 データ形式互換性テスト');
    
    // 各システムでサポートされるデータ形式
    const dataFormats = {
      '在庫管理': ['CSV', 'Excel', 'JSON', 'XML'],
      '物流ネットワーク設計': ['CSV', 'KML', 'GeoJSON', 'Shapefile'],
      'シフト最適化': ['CSV', 'Excel', 'iCal'],
      'PyVRP': ['CSV', 'VRPLIB', 'JSON'],
      '分析': ['CSV', 'Excel', 'Parquet', 'HDF5']
    };
    
    for (const [system, formats] of Object.entries(dataFormats)) {
      console.log(`\nシステム: ${system}`);
      
      await page.click(`text=${system}`);
      await page.waitForTimeout(2000);
      
      // データインポート/エクスポートタブを探す
      const dataTab = page.locator('text=データ管理, text=データ, text=インポート').first();
      if (await dataTab.isVisible()) {
        await dataTab.click();
        await page.waitForTimeout(1000);
        
        for (const format of formats) {
          // インポートテスト
          const importBtn = page.locator(`button:has-text("${format}"), button:has-text("インポート")`).first();
          if (await importBtn.isVisible()) {
            console.log(`${format}: インポート可能`);
          }
          
          // エクスポートテスト
          const exportBtn = page.locator(`button:has-text("${format}"), button:has-text("エクスポート")`).first();
          if (await exportBtn.isVisible()) {
            console.log(`${format}: エクスポート可能`);
          }
        }
      }
    }
  });
});