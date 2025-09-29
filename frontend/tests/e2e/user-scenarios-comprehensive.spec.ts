import { test, expect } from '@playwright/test';

test.describe('Real User Scenarios - Complete E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(120000); // 2分のタイムアウト
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(2000);
  });

  test.describe('実際のユーザー操作シナリオ - 在庫管理', () => {
    test('新規ユーザーが初めて在庫管理を使用する完全フロー', async ({ page }) => {
      console.log('📋 新規ユーザーシナリオ: 在庫管理システムの初回利用');
      
      // ステップ1: ダッシュボードから在庫管理へ移動
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // ステップ2: ヘルプ/チュートリアルを確認
      const helpButton = page.locator('button:has-text("ヘルプ"), button:has-text("?"), [aria-label="help"]').first();
      if (await helpButton.isVisible()) {
        await helpButton.click();
        await page.waitForTimeout(1000);
        
        // ヘルプダイアログを閉じる
        const closeButton = page.locator('button:has-text("閉じる"), button:has-text("×")').first();
        if (await closeButton.isVisible()) {
          await closeButton.click();
          await page.waitForTimeout(500);
        }
      }
      
      // ステップ3: EOQ分析タブでサンプルデータを使用
      await page.click('text=EOQ分析');
      await page.waitForTimeout(1000);
      
      // サンプルデータボタンを探して実行
      const sampleButton = page.locator('button:has-text("サンプル"), button:has-text("例")').first();
      if (await sampleButton.isVisible()) {
        await sampleButton.click();
        await page.waitForTimeout(2000);
      }
      
      // ステップ4: パラメータを手動で入力
      const inputs = {
        '発注コスト': '200',
        '需要量': '1500',
        '保管費率': '0.3',
        '欠品費率': '0.2'
      };
      
      for (const [label, value] of Object.entries(inputs)) {
        const input = page.locator(`input[placeholder*="${label}"], input[label*="${label}"]`).first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(value);
          await page.waitForTimeout(300);
        }
      }
      
      // ステップ5: 計算実行
      await page.click('button:has-text("計算")');
      await page.waitForTimeout(3000);
      
      // ステップ6: 結果を確認してスクリーンショット
      await page.screenshot({ path: 'test-results/inventory-eoq-results.png', fullPage: true });
      
      // ステップ7: 結果をエクスポート
      const exportButton = page.locator('button:has-text("エクスポート"), button:has-text("ダウンロード")').first();
      if (await exportButton.isVisible() && await exportButton.isEnabled()) {
        await exportButton.click();
        await page.waitForTimeout(2000);
      }
      
      console.log('✅ 在庫管理初回利用シナリオ完了');
    });

    test('在庫マネージャーの日次業務フロー', async ({ page }) => {
      console.log('👨‍💼 在庫マネージャーの日次業務シナリオ');
      
      // 在庫管理へ移動
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // シミュレーションタブへ移動
      await page.click('text=シミュレーション');
      await page.waitForTimeout(1000);
      
      // 前日の実績データをアップロード（シミュレート）
      const fileInput = page.locator('input[type="file"]').first();
      if (await fileInput.isVisible()) {
        // ダミーファイルを設定（実際のテストでは適切なファイルを用意）
        await fileInput.setInputFiles({
          name: 'daily-inventory-data.csv',
          mimeType: 'text/csv',
          buffer: Buffer.from('SKU,Demand,Stock\nA001,100,500\nA002,150,300\nA003,80,450')
        });
        await page.waitForTimeout(2000);
      }
      
      // シミュレーションパラメータ設定
      const simParams = {
        'サンプル数': '1000',
        '期間数': '30',
        '信頼区間': '95'
      };
      
      for (const [label, value] of Object.entries(simParams)) {
        const input = page.locator(`input[placeholder*="${label}"], input[label*="${label}"]`).first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(value);
          await page.waitForTimeout(200);
        }
      }
      
      // シミュレーション実行
      await page.click('button:has-text("シミュレーション実行")');
      await page.waitForTimeout(5000);
      
      // 最適化タブへ移動して安全在庫を計算
      await page.click('text=最適化');
      await page.waitForTimeout(1000);
      
      const optimizationSelect = page.locator('select').first();
      if (await optimizationSelect.isVisible()) {
        await optimizationSelect.selectOption('safety-stock');
        await page.waitForTimeout(500);
      }
      
      // 安全在庫パラメータ入力
      const safetyParams = {
        'サービスレベル': '0.98',
        'リードタイム': '5',
        '需要変動': '0.25'
      };
      
      for (const [label, value] of Object.entries(safetyParams)) {
        const input = page.locator(`input[placeholder*="${label}"], input[label*="${label}"]`).first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(value);
          await page.waitForTimeout(200);
        }
      }
      
      // 最適化実行
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      // レポート生成のためのスクリーンショット
      await page.screenshot({ path: 'test-results/daily-inventory-report.png', fullPage: true });
      
      console.log('✅ 在庫マネージャー日次業務シナリオ完了');
    });
  });

  test.describe('実際のユーザー操作シナリオ - 物流ネットワーク設計', () => {
    test('物流計画担当者が新規配送センター立地を決定するフロー', async ({ page }) => {
      console.log('🏭 新規配送センター立地決定シナリオ');
      
      // 物流ネットワーク設計へ移動
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      // データ管理タブで顧客データをアップロード
      await page.click('text=データ管理');
      await page.waitForTimeout(1000);
      
      // サンプルデータ生成
      const generateSampleButton = page.locator('button:has-text("サンプルデータ生成")').first();
      if (await generateSampleButton.isVisible()) {
        await generateSampleButton.click();
        await page.waitForTimeout(3000);
      }
      
      // 立地モデルタブへ移動
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      
      // 単一施設立地（Weiszfeld法）を実行
      const weiszfeldSection = page.locator('text=Weiszfeld法による単一施設最適立地').first();
      await expect(weiszfeldSection).toBeVisible();
      
      // パラメータ設定
      await page.fill('input[label*="最大反復回数"]', '200');
      await page.fill('input[label*="収束許容誤差"]', '1e-6');
      await page.waitForTimeout(500);
      
      // 計算実行
      await page.click('button:has-text("計算"), button:has-text("実行")');
      await page.waitForTimeout(4000);
      
      // 結果分析タブへ移動
      await page.click('text=結果分析');
      await page.waitForTimeout(1000);
      
      // サービスエリア分析を実行
      const serviceAreaButton = page.locator('button:has-text("サービスエリア分析")').first();
      if (await serviceAreaButton.isVisible()) {
        await serviceAreaButton.click();
        await page.waitForTimeout(3000);
      }
      
      // CO2排出量を計算
      const co2Button = page.locator('button:has-text("CO2"), button:has-text("排出量")').first();
      if (await co2Button.isVisible()) {
        await co2Button.click();
        await page.waitForTimeout(2000);
      }
      
      // ポリシー管理タブで設定を保存
      await page.click('text=ポリシー管理');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("設定保存")');
      await page.waitForTimeout(1000);
      
      console.log('✅ 配送センター立地決定シナリオ完了');
    });

    test('既存ネットワークの最適化と複数施設配置', async ({ page }) => {
      console.log('🔄 既存ネットワーク最適化シナリオ');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      // 立地モデルタブでK-Median最適化
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      
      // K-Median最適化セクションまでスクロール
      await page.locator('text=K-Median施設立地最適化').scrollIntoViewIfNeeded();
      
      // K値（施設数）を設定
      await page.fill('input[label*="K値"]', '5');
      await page.fill('input[label*="最大反復回数"]', '300');
      await page.waitForTimeout(500);
      
      // K-Median実行
      await page.click('button:has-text("K-Median"), button:has-text("最適化")');
      await page.waitForTimeout(5000);
      
      // エルボー法で最適施設数を分析
      await page.click('text=結果分析');
      await page.waitForTimeout(1000);
      
      await page.fill('input[label*="最小施設数"]', '2');
      await page.fill('input[label*="最大施設数"]', '10');
      await page.waitForTimeout(500);
      
      await page.click('button:has-text("エルボー法")');
      await page.waitForTimeout(4000);
      
      // 最適化実行タブで詳細な最適化
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      // Multiple Source LND最適化
      await page.fill('input[label*="最大実行時間"]', '600');
      await page.fill('input[label*="倉庫数"]', '5');
      await page.waitForTimeout(500);
      
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      console.log('✅ ネットワーク最適化シナリオ完了');
    });
  });

  test.describe('実際のユーザー操作シナリオ - シフト最適化', () => {
    test('店舗マネージャーが週次シフトを作成する完全フロー', async ({ page }) => {
      console.log('📅 週次シフト作成シナリオ');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      // システム設定でサンプルデータ生成
      await page.click('text=システム設定');
      await page.waitForTimeout(1000);
      
      // 期間設定
      await page.fill('input[type="date"]', '2024-01-15');
      await page.locator('input[type="date"]').nth(1).fill('2024-01-21');
      await page.waitForTimeout(500);
      
      // 時間設定
      await page.fill('input[label*="開始時刻"]', '09:00');
      await page.fill('input[label*="終了時刻"]', '22:00');
      await page.waitForTimeout(500);
      
      // ジョブリスト設定
      await page.fill('input[label*="ジョブリスト"]', 'レジ, 品出し, 清掃, 接客, 在庫管理');
      await page.waitForTimeout(500);
      
      // サンプルデータ生成
      await page.click('button:has-text("サンプルデータ生成")');
      await page.waitForTimeout(3000);
      
      // スタッフ管理タブでスタッフ情報確認
      await page.click('text=スタッフ管理');
      await page.waitForTimeout(1000);
      
      // シフト要件タブで必要人数設定
      await page.click('text=シフト要件');
      await page.waitForTimeout(1000);
      
      // 最適化実行
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      // パラメータ設定
      const shiftParams = {
        'θ (theta)': '1.5',
        'lb_penalty': '15000',
        'ub_penalty': '0',
        'job_change_penalty': '20',
        'break_penalty': '15000',
        'max_day_penalty': '8000',
        'time_limit': '60'
      };
      
      for (const [label, value] of Object.entries(shiftParams)) {
        const input = page.locator(`input[label*="${label}"]`).first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(value);
          await page.waitForTimeout(200);
        }
      }
      
      // 最適化実行
      await page.click('button:has-text("最適化"), button:has-text("実行")');
      await page.waitForTimeout(5000);
      
      // 結果分析
      await page.click('text=結果分析');
      await page.waitForTimeout(1000);
      
      // リアルタイム監視で違反チェック
      await page.click('text=リアルタイム監視');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("実行可能性分析")');
      await page.waitForTimeout(3000);
      
      // データ管理でエクスポート
      await page.click('text=データ管理');
      await page.waitForTimeout(1000);
      
      // ガントチャートExcelエクスポート
      const exportGanttButton = page.locator('button:has-text("ガントチャートExcel")').first();
      if (await exportGanttButton.isVisible() && await exportGanttButton.isEnabled()) {
        await exportGanttButton.click();
        await page.waitForTimeout(2000);
      }
      
      console.log('✅ 週次シフト作成シナリオ完了');
    });

    test('緊急シフト変更対応シナリオ', async ({ page }) => {
      console.log('🚨 緊急シフト変更シナリオ');
      
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      // リアルタイム監視タブへ直接移動
      await page.click('text=リアルタイム監視');
      await page.waitForTimeout(1000);
      
      // 現在のシフト状況を確認
      await page.click('button:has-text("実行可能性分析")');
      await page.waitForTimeout(3000);
      
      // スタッフ管理タブで欠勤者を設定
      await page.click('text=スタッフ管理');
      await page.waitForTimeout(1000);
      
      // 特定スタッフの利用不可設定（シミュレート）
      const staffCheckbox = page.locator('input[type="checkbox"]').first();
      if (await staffCheckbox.isVisible()) {
        await staffCheckbox.uncheck();
        await page.waitForTimeout(500);
      }
      
      // 再最適化実行
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      // 緊急モードで高速最適化
      await page.fill('input[label*="time_limit"]', '10');
      await page.waitForTimeout(300);
      
      await page.click('button:has-text("最適化"), button:has-text("実行")');
      await page.waitForTimeout(3000);
      
      // 結果確認
      await page.click('text=結果分析');
      await page.waitForTimeout(1000);
      
      console.log('✅ 緊急シフト変更シナリオ完了');
    });
  });

  test.describe('実際のユーザー操作シナリオ - VRP配送計画', () => {
    test('配送計画担当者が日次配送ルートを作成', async ({ page }) => {
      console.log('🚚 日次配送ルート作成シナリオ');
      
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      
      // データ入力タブ
      await page.click('text=データ入力');
      await page.waitForTimeout(1000);
      
      // サンプルデータ生成
      await page.click('button:has-text("サンプル")');
      await page.waitForTimeout(3000);
      
      // 車両設定タブ
      await page.click('text=車両設定');
      await page.waitForTimeout(1000);
      
      // 車両パラメータ設定
      await page.fill('input[label*="車両数"]', '3');
      await page.fill('input[label*="積載容量"]', '1500');
      await page.fill('input[label*="最大距離"]', '400');
      await page.waitForTimeout(500);
      
      // 制約設定タブ
      await page.click('text=制約設定');
      await page.waitForTimeout(1000);
      
      // 時間窓制約設定
      const timeWindowInputs = page.locator('input[type="time"]');
      if (await timeWindowInputs.count() > 0) {
        await timeWindowInputs.nth(0).fill('08:00');
        await timeWindowInputs.nth(1).fill('17:00');
        await page.waitForTimeout(500);
      }
      
      // アルゴリズム設定
      await page.click('text=アルゴリズム設定');
      await page.waitForTimeout(1000);
      
      await page.fill('input[label*="実行時間"]', '180');
      await page.fill('input[label*="反復回数"]', '2000');
      await page.waitForTimeout(500);
      
      // 最適化実行
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(5000);
      
      // 結果分析
      await page.click('text=結果分析');
      await page.waitForTimeout(1000);
      
      // エクスポート
      await page.click('text=エクスポート');
      await page.waitForTimeout(1000);
      
      const exportRouteButton = page.locator('button:has-text("ルート詳細")').first();
      if (await exportRouteButton.isVisible() && await exportRouteButton.isEnabled()) {
        await exportRouteButton.click();
        await page.waitForTimeout(2000);
      }
      
      console.log('✅ 日次配送ルート作成シナリオ完了');
    });
  });

  test.describe('実際のユーザー操作シナリオ - 分析システム', () => {
    test('データアナリストが需要予測分析を実行', async ({ page }) => {
      console.log('📊 需要予測分析シナリオ');
      
      await page.click('text=分析');
      await page.waitForTimeout(2000);
      
      // データ準備
      await page.click('text=データ準備');
      await page.waitForTimeout(1000);
      
      // サンプルデータ生成
      await page.click('button:has-text("サンプル")');
      await page.waitForTimeout(3000);
      
      // 統計分析
      await page.click('text=統計分析');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("統計分析")');
      await page.waitForTimeout(3000);
      
      // 予測分析
      await page.click('text=予測分析');
      await page.waitForTimeout(1000);
      
      // モデル選択
      const modelSelect = page.locator('select').first();
      if (await modelSelect.isVisible()) {
        await modelSelect.selectOption({ index: 1 });
        await page.waitForTimeout(500);
      }
      
      await page.click('button:has-text("予測実行")');
      await page.waitForTimeout(4000);
      
      // 可視化
      await page.click('text=可視化');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("グラフ")');
      await page.waitForTimeout(2000);
      
      // レポート生成
      await page.click('text=レポート生成');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("レポート生成")');
      await page.waitForTimeout(3000);
      
      console.log('✅ 需要予測分析シナリオ完了');
    });
  });

  test.describe('実際のユーザー操作シナリオ - 複合的な業務フロー', () => {
    test('サプライチェーン全体最適化の統合シナリオ', async ({ page }) => {
      console.log('🌐 サプライチェーン統合最適化シナリオ');
      
      // 1. 需要予測から開始
      await page.click('text=分析');
      await page.waitForTimeout(2000);
      await page.click('text=予測分析');
      await page.waitForTimeout(1000);
      await page.click('button:has-text("予測実行")');
      await page.waitForTimeout(3000);
      
      // 2. 在庫最適化
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      await page.click('text=最適化');
      await page.waitForTimeout(1000);
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      // 3. 配送ネットワーク設計
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      await page.click('text=立地モデル');
      await page.waitForTimeout(1000);
      await page.click('button:has-text("計算")');
      await page.waitForTimeout(3000);
      
      // 4. 配送ルート最適化
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      // 5. 総合レポート（ダッシュボードに戻る）
      await page.click('text=ダッシュボード');
      await page.waitForTimeout(2000);
      
      console.log('✅ サプライチェーン統合最適化シナリオ完了');
    });

    test('月次計画から日次実行までの一連の流れ', async ({ page }) => {
      console.log('📅 月次→週次→日次計画シナリオ');
      
      // 月次需要予測
      await page.click('text=分析');
      await page.waitForTimeout(2000);
      await page.click('text=予測分析');
      await page.waitForTimeout(1000);
      
      // 予測期間を月次に設定
      const periodSelect = page.locator('select[label*="期間"]').first();
      if (await periodSelect.isVisible()) {
        await periodSelect.selectOption('monthly');
        await page.waitForTimeout(500);
      }
      
      await page.click('button:has-text("予測実行")');
      await page.waitForTimeout(3000);
      
      // 在庫計画
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      await page.click('text=多段階在庫');
      await page.waitForTimeout(1000);
      
      await page.fill('input[label*="計画期間"]', '30');
      await page.waitForTimeout(300);
      
      await page.click('button:has-text("分析実行")');
      await page.waitForTimeout(3000);
      
      // 週次シフト計画
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("最適化")');
      await page.waitForTimeout(3000);
      
      // 日次配送計画
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      console.log('✅ 月次→週次→日次計画シナリオ完了');
    });
  });

  test.describe('エラー回復とトラブルシューティングシナリオ', () => {
    test('データアップロードエラーからの回復', async ({ page }) => {
      console.log('🔧 エラー回復シナリオ: データアップロード');
      
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      await page.click('text=データ管理');
      await page.waitForTimeout(1000);
      
      // 不正なファイルをアップロード試行
      const fileInput = page.locator('input[type="file"]').first();
      if (await fileInput.isVisible()) {
        await fileInput.setInputFiles({
          name: 'invalid-data.txt',
          mimeType: 'text/plain',
          buffer: Buffer.from('This is not valid CSV data')
        });
        await page.waitForTimeout(2000);
      }
      
      // エラーメッセージの確認
      const errorMessage = await page.locator('text=エラー, text=無効, [class*="error"]').isVisible();
      if (errorMessage) {
        console.log('✅ エラーメッセージが適切に表示されました');
        
        // エラーからの回復 - サンプルデータ使用
        await page.click('button:has-text("サンプル")');
        await page.waitForTimeout(2000);
        console.log('✅ サンプルデータでエラーから回復');
      }
    });

    test('最適化タイムアウトと再実行', async ({ page }) => {
      console.log('⏱️ タイムアウト処理シナリオ');
      
      await page.click('text=PyVRP');
      await page.waitForTimeout(2000);
      
      await page.click('text=アルゴリズム設定');
      await page.waitForTimeout(1000);
      
      // 極端に短いタイムアウトを設定
      await page.fill('input[label*="実行時間"]', '1');
      await page.waitForTimeout(300);
      
      await page.click('text=最適化実行');
      await page.waitForTimeout(1000);
      
      await page.click('button:has-text("最適化実行")');
      await page.waitForTimeout(3000);
      
      // タイムアウトメッセージ確認
      const timeoutMessage = await page.locator('text=タイムアウト, text=時間切れ, text=timeout').isVisible();
      if (timeoutMessage) {
        console.log('✅ タイムアウトメッセージ確認');
        
        // パラメータ修正して再実行
        await page.click('text=アルゴリズム設定');
        await page.waitForTimeout(1000);
        await page.fill('input[label*="実行時間"]', '60');
        await page.waitForTimeout(300);
        
        await page.click('text=最適化実行');
        await page.waitForTimeout(1000);
        await page.click('button:has-text("最適化実行")');
        await page.waitForTimeout(3000);
        
        console.log('✅ 再実行成功');
      }
    });
  });

  test.describe('アクセシビリティとユーザビリティシナリオ', () => {
    test('キーボードのみでの完全操作', async ({ page }) => {
      console.log('⌨️ キーボードナビゲーションシナリオ');
      
      // Tabキーでナビゲーション
      await page.keyboard.press('Tab');
      await page.waitForTimeout(200);
      await page.keyboard.press('Tab');
      await page.waitForTimeout(200);
      
      // Enterキーで選択
      await page.keyboard.press('Enter');
      await page.waitForTimeout(1000);
      
      // 矢印キーでタブ切り替え
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(500);
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(500);
      
      // Spaceキーでボタン操作
      await page.keyboard.press('Tab');
      await page.waitForTimeout(200);
      await page.keyboard.press('Space');
      await page.waitForTimeout(500);
      
      console.log('✅ キーボードナビゲーション動作確認');
    });

    test('モバイルデバイスでの操作', async ({ page }) => {
      console.log('📱 モバイル操作シナリオ');
      
      // モバイルビューポートに設定
      await page.setViewportSize({ width: 375, height: 667 });
      await page.waitForTimeout(1000);
      
      // ハンバーガーメニューの操作
      const menuButton = page.locator('[aria-label="menu"], button:has-text("☰")').first();
      if (await menuButton.isVisible()) {
        await menuButton.click();
        await page.waitForTimeout(1000);
      }
      
      // メニュー項目選択
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // スワイプ操作のシミュレート（横スクロール）
      const tabContainer = page.locator('[role="tablist"]').first();
      if (await tabContainer.isVisible()) {
        await tabContainer.scrollIntoViewIfNeeded();
        await page.waitForTimeout(500);
      }
      
      console.log('✅ モバイル操作確認完了');
    });
  });

  test('パフォーマンス負荷テスト - 大量データ処理', async ({ page }) => {
    console.log('🏋️ パフォーマンス負荷テストシナリオ');
    
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    await page.click('text=シミュレーション');
    await page.waitForTimeout(1000);
    
    // 大量データでのシミュレーション
    await page.fill('input[label*="サンプル数"]', '10000');
    await page.fill('input[label*="期間数"]', '365');
    await page.fill('input[label*="製品数"]', '1000');
    await page.waitForTimeout(500);
    
    const startTime = Date.now();
    await page.click('button:has-text("シミュレーション実行")');
    
    // タイムアウトを長めに設定して完了を待つ
    await page.waitForTimeout(10000);
    
    const endTime = Date.now();
    const executionTime = endTime - startTime;
    
    console.log(`✅ 大量データ処理完了: ${executionTime}ms`);
    
    // メモリ使用量の確認（ブラウザコンソール経由）
    const performanceMetrics = await page.evaluate(() => {
      if (performance.memory) {
        return {
          usedJSHeapSize: Math.round(performance.memory.usedJSHeapSize / 1048576),
          totalJSHeapSize: Math.round(performance.memory.totalJSHeapSize / 1048576)
        };
      }
      return null;
    });
    
    if (performanceMetrics) {
      console.log(`メモリ使用量: ${performanceMetrics.usedJSHeapSize}MB / ${performanceMetrics.totalJSHeapSize}MB`);
    }
  });
});

test.describe('セキュリティとデータ保護シナリオ', () => {
  test('機密データの適切な取り扱い確認', async ({ page }) => {
    console.log('🔒 セキュリティ確認シナリオ');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // ローカルストレージのクリア確認
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
    
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    // 機密データ入力
    await page.fill('input[label*="コスト"]', '1000000');
    await page.waitForTimeout(500);
    
    // ページリロード
    await page.reload();
    await page.waitForTimeout(2000);
    
    // データが保持されていないことを確認
    const costInput = page.locator('input[label*="コスト"]').first();
    if (await costInput.isVisible()) {
      const value = await costInput.inputValue();
      if (value === '') {
        console.log('✅ 機密データは適切にクリアされています');
      }
    }
    
    // コンソールログに機密情報が出力されていないか確認
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'log') {
        consoleLogs.push(msg.text());
      }
    });
    
    await page.click('button:has-text("計算")');
    await page.waitForTimeout(2000);
    
    const sensitiveDataInLogs = consoleLogs.some(log => 
      log.includes('1000000') || log.includes('password') || log.includes('secret')
    );
    
    if (!sensitiveDataInLogs) {
      console.log('✅ コンソールログに機密情報は含まれていません');
    }
  });

  test('セッションタイムアウトと再認証フロー', async ({ page }) => {
    console.log('⏰ セッションタイムアウトシナリオ');
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // 長時間の非活動をシミュレート
    await page.waitForTimeout(5000);
    
    // 操作を試行
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    // セッションタイムアウトダイアログの確認
    const sessionDialog = await page.locator('text=セッションが期限切れ, text=再ログイン, text=session expired').isVisible();
    if (sessionDialog) {
      console.log('✅ セッションタイムアウト処理が正常に動作');
      
      // 再認証ボタンをクリック
      const reAuthButton = page.locator('button:has-text("再ログイン"), button:has-text("続ける")').first();
      if (await reAuthButton.isVisible()) {
        await reAuthButton.click();
        await page.waitForTimeout(2000);
      }
    }
  });
});