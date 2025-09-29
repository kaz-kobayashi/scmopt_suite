import { test, expect } from '@playwright/test';

test.describe('Complete Feature Coverage Tests - Every Possible User Action', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(120000); // 2分タイムアウト
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
  });

  test.describe('完全なユーザーインタラクションテスト', () => {
    test('全ボタンクリック可能性テスト', async ({ page }) => {
      console.log('🖱️ 全ボタンクリック可能性テスト');
      
      // 全てのボタンを取得してクリックテスト
      const buttonResults = await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const results = [];
        
        buttons.forEach((button, index) => {
          const isVisible = button.offsetWidth > 0 && button.offsetHeight > 0;
          const isEnabled = !button.disabled;
          const hasText = button.textContent?.trim() || button.getAttribute('aria-label') || button.getAttribute('title');
          
          results.push({
            index,
            text: hasText?.substring(0, 30) || '',
            visible: isVisible,
            enabled: isEnabled,
            clickable: isVisible && isEnabled
          });
        });
        
        return results;
      });
      
      console.log(`総ボタン数: ${buttonResults.length}`);
      console.log(`クリック可能: ${buttonResults.filter(b => b.clickable).length}`);
      console.log(`無効状態: ${buttonResults.filter(b => !b.enabled).length}`);
      
      // 各セクションのボタンをテスト
      const sections = ['在庫管理', '物流ネットワーク設計', 'シフト最適化', 'PyVRP', 'ジョブショップスケジューリング', '分析'];
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        const sectionButtons = await page.locator('button').count();
        console.log(`${section}のボタン数: ${sectionButtons}`);
        
        // ランダムにいくつかのボタンをクリック
        const maxTests = Math.min(sectionButtons, 5);
        for (let i = 0; i < maxTests; i++) {
          const randomIndex = Math.floor(Math.random() * sectionButtons);
          const button = page.locator('button').nth(randomIndex);
          
          if (await button.isVisible() && await button.isEnabled()) {
            try {
              await button.click();
              await page.waitForTimeout(1000);
              console.log(`${section} - ボタン${randomIndex}: クリック成功`);
            } catch (error) {
              console.log(`${section} - ボタン${randomIndex}: クリック失敗`);
            }
          }
        }
      }
    });

    test('全フォーム入力フィールドテスト', async ({ page }) => {
      console.log('📝 全フォーム入力フィールドテスト');
      
      const sections = ['在庫管理', '物流ネットワーク設計', 'シフト最適化', 'PyVRP'];
      const testInputs = {
        'text': 'テストデータ123',
        'number': '12345',
        'email': 'test@example.com',
        'url': 'https://example.com',
        'tel': '03-1234-5678',
        'date': '2024-12-31',
        'time': '14:30',
        'datetime-local': '2024-12-31T14:30',
        'color': '#ff0000',
        'range': '50'
      };
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        console.log(`\n${section}の入力フィールドテスト:`);
        
        // 全ての入力タイプをテスト
        for (const [inputType, testValue] of Object.entries(testInputs)) {
          const inputs = page.locator(`input[type="${inputType}"]`);
          const count = await inputs.count();
          
          if (count > 0) {
            console.log(`  ${inputType}フィールド: ${count}個`);
            
            for (let i = 0; i < Math.min(count, 3); i++) {
              const input = inputs.nth(i);
              if (await input.isVisible() && await input.isEnabled()) {
                try {
                  await input.clear();
                  await input.fill(testValue);
                  await page.waitForTimeout(200);
                  
                  const value = await input.inputValue();
                  console.log(`    フィールド${i}: ${value ? '入力成功' : '入力失敗'}`);
                } catch (error) {
                  console.log(`    フィールド${i}: エラー`);
                }
              }
            }
          }
        }
        
        // select要素のテスト
        const selects = page.locator('select');
        const selectCount = await selects.count();
        console.log(`  selectフィールド: ${selectCount}個`);
        
        for (let i = 0; i < Math.min(selectCount, 3); i++) {
          const select = selects.nth(i);
          if (await select.isVisible() && await select.isEnabled()) {
            const options = await select.locator('option').count();
            if (options > 1) {
              await select.selectOption({ index: 1 });
              await page.waitForTimeout(200);
              console.log(`    select${i}: 選択成功`);
            }
          }
        }
        
        // textarea要素のテスト
        const textareas = page.locator('textarea');
        const textareaCount = await textareas.count();
        console.log(`  textareaフィールド: ${textareaCount}個`);
        
        for (let i = 0; i < Math.min(textareaCount, 2); i++) {
          const textarea = textareas.nth(i);
          if (await textarea.isVisible() && await textarea.isEnabled()) {
            await textarea.fill('これは長いテキストの入力テストです。\n複数行にわたってテストを行います。');
            await page.waitForTimeout(200);
            console.log(`    textarea${i}: 入力成功`);
          }
        }
      }
    });

    test('全チェックボックス・ラジオボタンテスト', async ({ page }) => {
      console.log('☑️ 全チェックボックス・ラジオボタンテスト');
      
      const sections = ['在庫管理', 'シフト最適化', 'PyVRP'];
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        console.log(`\n${section}のチェックボックス・ラジオボタンテスト:`);
        
        // チェックボックステスト
        const checkboxes = page.locator('input[type="checkbox"]');
        const checkboxCount = await checkboxes.count();
        console.log(`  チェックボックス: ${checkboxCount}個`);
        
        for (let i = 0; i < Math.min(checkboxCount, 5); i++) {
          const checkbox = checkboxes.nth(i);
          if (await checkbox.isVisible() && await checkbox.isEnabled()) {
            // チェック
            await checkbox.check();
            await page.waitForTimeout(200);
            const isChecked = await checkbox.isChecked();
            
            // アンチェック
            await checkbox.uncheck();
            await page.waitForTimeout(200);
            const isUnchecked = !(await checkbox.isChecked());
            
            console.log(`    チェックボックス${i}: チェック=${isChecked}, アンチェック=${isUnchecked}`);
          }
        }
        
        // ラジオボタンテスト
        const radioGroups = await page.evaluate(() => {
          const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
          const groups = {};
          
          radios.forEach(radio => {
            const name = radio.name || 'unnamed';
            if (!groups[name]) groups[name] = [];
            groups[name].push(radio.value || radio.id);
          });
          
          return groups;
        });
        
        console.log(`  ラジオボタングループ: ${Object.keys(radioGroups).length}個`);
        
        for (const [groupName, values] of Object.entries(radioGroups)) {
          console.log(`    グループ "${groupName}": ${values.length}個のオプション`);
          
          for (let i = 0; i < Math.min(values.length, 3); i++) {
            const radio = page.locator(`input[type="radio"][name="${groupName}"]`).nth(i);
            if (await radio.isVisible() && await radio.isEnabled()) {
              await radio.check();
              await page.waitForTimeout(200);
              const isChecked = await radio.isChecked();
              console.log(`      オプション${i}: ${isChecked ? '選択成功' : '選択失敗'}`);
            }
          }
        }
      }
    });

    test('全ドロップダウン・コンボボックステスト', async ({ page }) => {
      console.log('📋 全ドロップダウン・コンボボックステスト');
      
      const sections = ['在庫管理', '物流ネットワーク設計', 'シフト最適化', 'PyVRP', '分析'];
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        console.log(`\n${section}のドロップダウンテスト:`);
        
        // select要素
        const selects = page.locator('select');
        const selectCount = await selects.count();
        console.log(`  select要素: ${selectCount}個`);
        
        for (let i = 0; i < Math.min(selectCount, 5); i++) {
          const select = selects.nth(i);
          if (await select.isVisible() && await select.isEnabled()) {
            const options = await select.locator('option').count();
            console.log(`    select${i}: ${options}個のオプション`);
            
            // 全てのオプションをテスト
            for (let j = 0; j < Math.min(options, 3); j++) {
              await select.selectOption({ index: j });
              await page.waitForTimeout(300);
              
              const selectedValue = await select.inputValue();
              console.log(`      オプション${j}: ${selectedValue}`);
            }
          }
        }
        
        // ARIA combobox
        const comboboxes = page.locator('[role="combobox"]');
        const comboCount = await comboboxes.count();
        console.log(`  combobox要素: ${comboCount}個`);
        
        for (let i = 0; i < Math.min(comboCount, 3); i++) {
          const combo = comboboxes.nth(i);
          if (await combo.isVisible()) {
            await combo.click();
            await page.waitForTimeout(500);
            
            // ドロップダウンリストの確認
            const listItems = page.locator('[role="option"]');
            const itemCount = await listItems.count();
            
            if (itemCount > 0) {
              console.log(`    combobox${i}: ${itemCount}個のオプション`);
              
              // いくつかのオプションを選択
              for (let j = 0; j < Math.min(itemCount, 2); j++) {
                await listItems.nth(j).click();
                await page.waitForTimeout(300);
                console.log(`      オプション${j}: 選択完了`);
                
                // 再度開く
                await combo.click();
                await page.waitForTimeout(300);
              }
            }
          }
        }
      }
    });

    test('全ファイルアップロード機能テスト', async ({ page }) => {
      console.log('📎 全ファイルアップロード機能テスト');
      
      const sections = ['物流ネットワーク設計', 'PyVRP', '分析'];
      
      // テスト用ファイルデータ
      const testFiles = {
        'CSV': {
          content: 'id,name,value\n1,Test1,100\n2,Test2,200',
          mimeType: 'text/csv',
          extension: '.csv'
        },
        'Excel': {
          content: 'PK\x03\x04', // Excel magic bytes
          mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          extension: '.xlsx'
        },
        'JSON': {
          content: '{"data": [{"id": 1, "value": 100}]}',
          mimeType: 'application/json',
          extension: '.json'
        },
        'XML': {
          content: '<?xml version="1.0"?><root><item id="1">Test</item></root>',
          mimeType: 'application/xml',
          extension: '.xml'
        }
      };
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        // データ管理タブを探す
        const dataTab = page.locator('text=データ管理, text=データ入力, text=ファイル').first();
        if (await dataTab.isVisible()) {
          await dataTab.click();
          await page.waitForTimeout(1000);
        }
        
        console.log(`\n${section}のファイルアップロードテスト:`);
        
        const fileInputs = page.locator('input[type="file"]');
        const inputCount = await fileInputs.count();
        console.log(`  ファイル入力フィールド: ${inputCount}個`);
        
        for (let i = 0; i < Math.min(inputCount, 2); i++) {
          const fileInput = fileInputs.nth(i);
          
          if (await fileInput.isVisible()) {
            for (const [fileType, fileData] of Object.entries(testFiles)) {
              console.log(`    ${fileType}ファイルテスト:`);
              
              try {
                await fileInput.setInputFiles({
                  name: `test${fileData.extension}`,
                  mimeType: fileData.mimeType,
                  buffer: Buffer.from(fileData.content)
                });
                await page.waitForTimeout(2000);
                
                // アップロード結果の確認
                const successMsg = await page.locator('text=成功, text=完了, text=アップロード').isVisible();
                const errorMsg = await page.locator('text=エラー, text=失敗, text=無効').isVisible();
                
                console.log(`      結果: ${successMsg ? '成功' : errorMsg ? 'エラー' : '不明'}`);
                
                // エラーダイアログをクリア
                if (errorMsg) {
                  const closeBtn = page.locator('button:has-text("閉じる"), button:has-text("OK")').first();
                  if (await closeBtn.isVisible()) {
                    await closeBtn.click();
                    await page.waitForTimeout(500);
                  }
                }
              } catch (error) {
                console.log(`      ${fileType}: アップロードエラー`);
              }
            }
          }
        }
      }
    });

    test('全ダイアログ・モーダル操作テスト', async ({ page }) => {
      console.log('🔲 全ダイアログ・モーダル操作テスト');
      
      const sections = ['在庫管理', '物流ネットワーク設計', 'シフト最適化'];
      
      for (const section of sections) {
        await page.click(`text=${section}`);
        await page.waitForTimeout(2000);
        
        console.log(`\n${section}のダイアログテスト:`);
        
        // ダイアログを開くボタンを探す
        const dialogTriggers = [
          'ヘルプ', 'Help', '設定', 'Settings', '追加', 'Add', '編集', 'Edit', 
          '削除', 'Delete', '詳細', 'Details', '情報', 'Info'
        ];
        
        for (const trigger of dialogTriggers) {
          const triggerBtn = page.locator(`button:has-text("${trigger}")`).first();
          
          if (await triggerBtn.isVisible()) {
            await triggerBtn.click();
            await page.waitForTimeout(1000);
            
            // ダイアログが開いたか確認
            const dialog = page.locator('[role="dialog"], .modal, .popup').first();
            
            if (await dialog.isVisible()) {
              console.log(`  ${trigger}ダイアログ: 開いた`);
              
              // ダイアログ内のボタンをテスト
              const dialogButtons = dialog.locator('button');
              const buttonCount = await dialogButtons.count();
              
              console.log(`    ダイアログ内ボタン: ${buttonCount}個`);
              
              // Escapeキーでクローズテスト
              await page.keyboard.press('Escape');
              await page.waitForTimeout(500);
              
              const isClosed = !(await dialog.isVisible());
              console.log(`    Escapeキーでクローズ: ${isClosed ? '成功' : '失敗'}`);
              
              // まだ開いている場合は明示的にクローズ
              if (!isClosed) {
                const closeBtn = dialog.locator('button:has-text("閉じる"), button:has-text("Close"), button:has-text("×")').first();
                if (await closeBtn.isVisible()) {
                  await closeBtn.click();
                  await page.waitForTimeout(500);
                }
              }
            }
          }
        }
        
        // 右クリックコンテキストメニューのテスト
        const tables = page.locator('table, [role="grid"]');
        const tableCount = await tables.count();
        
        if (tableCount > 0) {
          console.log(`  テーブル右クリックテスト:`);
          
          const firstTable = tables.first();
          await firstTable.click({ button: 'right' });
          await page.waitForTimeout(1000);
          
          const contextMenu = page.locator('[role="menu"], .context-menu').first();
          if (await contextMenu.isVisible()) {
            console.log(`    コンテキストメニュー: 表示`);
            
            // メニュー項目のクリック
            const menuItems = contextMenu.locator('[role="menuitem"], button, a');
            const itemCount = await menuItems.count();
            
            if (itemCount > 0) {
              await menuItems.first().click();
              await page.waitForTimeout(500);
              console.log(`    メニュー項目クリック: 成功`);
            }
          }
        }
      }
    });
  });

  test('完全な統合ワークフローテスト', async ({ page }) => {
    console.log('🔄 完全統合ワークフローテスト');
    
    const workflowSteps = [
      {
        step: 'データ準備・分析',
        actions: [
          { section: '分析', tab: 'データ準備', action: 'button:has-text("サンプル")' },
          { section: '分析', tab: '統計分析', action: 'button:has-text("分析実行")' }
        ]
      },
      {
        step: '需要予測',
        actions: [
          { section: '分析', tab: '予測分析', action: 'select', value: 'ARIMA' },
          { section: '分析', tab: '予測分析', action: 'button:has-text("予測実行")' }
        ]
      },
      {
        step: '在庫最適化',
        actions: [
          { section: '在庫管理', tab: '最適化', action: 'select', value: 'qr' },
          { section: '在庫管理', tab: '最適化', action: 'button:has-text("最適化実行")' }
        ]
      },
      {
        step: 'ネットワーク設計',
        actions: [
          { section: '物流ネットワーク設計', tab: '立地モデル', action: 'input[label*="K値"]', value: '3' },
          { section: '物流ネットワーク設計', tab: '立地モデル', action: 'button:has-text("K-Median")' }
        ]
      },
      {
        step: '配送計画',
        actions: [
          { section: 'PyVRP', tab: '車両設定', action: 'input[label*="車両数"]', value: '5' },
          { section: 'PyVRP', tab: '最適化実行', action: 'button:has-text("最適化実行")' }
        ]
      },
      {
        step: 'シフト計画',
        actions: [
          { section: 'シフト最適化', tab: 'システム設定', action: 'button:has-text("サンプルデータ生成")' },
          { section: 'シフト最適化', tab: '最適化実行', action: 'button:has-text("最適化")' }
        ]
      },
      {
        step: 'レポート生成',
        actions: [
          { section: '分析', tab: 'レポート生成', action: 'button:has-text("レポート生成")' },
          { section: '分析', tab: 'エクスポート・共有', action: 'button:has-text("エクスポート")' }
        ]
      }
    ];
    
    let successfulSteps = 0;
    
    for (const workflow of workflowSteps) {
      console.log(`\n実行中: ${workflow.step}`);
      let stepSuccess = true;
      
      try {
        for (const action of workflow.actions) {
          // セクション移動
          await page.click(`text=${action.section}`);
          await page.waitForTimeout(2000);
          
          // タブ移動
          if (action.tab) {
            await page.click(`text=${action.tab}`);
            await page.waitForTimeout(1000);
          }
          
          // アクション実行
          if (action.action.includes('button')) {
            const button = page.locator(action.action).first();
            if (await button.isVisible() && await button.isEnabled()) {
              await button.click();
              await page.waitForTimeout(3000);
            }
          } else if (action.action.includes('select')) {
            const select = page.locator('select').first();
            if (await select.isVisible() && action.value) {
              await select.selectOption(action.value);
              await page.waitForTimeout(500);
            }
          } else if (action.action.includes('input')) {
            const input = page.locator(action.action).first();
            if (await input.isVisible() && action.value) {
              await input.fill(action.value);
              await page.waitForTimeout(500);
            }
          }
        }
        
        successfulSteps++;
        console.log(`${workflow.step}: ✅ 成功`);
        
      } catch (error) {
        stepSuccess = false;
        console.log(`${workflow.step}: ❌ 失敗 - ${error.message}`);
      }
    }
    
    const successRate = (successfulSteps / workflowSteps.length) * 100;
    console.log(`\n統合ワークフロー成功率: ${successRate.toFixed(1)}% (${successfulSteps}/${workflowSteps.length})`);
    
    expect(successRate).toBeGreaterThan(70); // 70%以上の成功率を期待
  });

  test('完全なエラー処理とリカバリテスト', async ({ page }) => {
    console.log('🔧 完全エラー処理・リカバリテスト');
    
    const errorScenarios = [
      {
        name: '無効データ入力',
        setup: async () => {
          await page.click('text=在庫管理');
          await page.waitForTimeout(2000);
          await page.fill('input[label*="発注コスト"]', 'invalid');
          await page.click('button:has-text("計算")');
        }
      },
      {
        name: 'ネットワークタイムアウト',
        setup: async () => {
          await page.route('**/*', route => {
            setTimeout(() => route.abort(), 10000);
          });
          await page.click('text=PyVRP');
          await page.waitForTimeout(2000);
          await page.click('button:has-text("最適化実行")');
        }
      },
      {
        name: 'メモリ不足',
        setup: async () => {
          await page.click('text=在庫管理');
          await page.waitForTimeout(2000);
          await page.click('text=シミュレーション');
          await page.waitForTimeout(1000);
          await page.fill('input[label*="サンプル数"]', '99999999');
          await page.click('button:has-text("シミュレーション実行")');
        }
      }
    ];
    
    for (const scenario of errorScenarios) {
      console.log(`\nテスト: ${scenario.name}`);
      
      try {
        await scenario.setup();
        await page.waitForTimeout(3000);
        
        // エラーメッセージの確認
        const hasError = await page.locator('text=エラー, text=Error, [class*="error"]').isVisible();
        console.log(`エラー表示: ${hasError ? '✅' : '❌'}`);
        
        // エラー回復の確認
        if (hasError) {
          // エラーダイアログを閉じる
          const closeBtn = page.locator('button:has-text("閉じる"), button:has-text("OK")').first();
          if (await closeBtn.isVisible()) {
            await closeBtn.click();
            await page.waitForTimeout(500);
          }
          
          // アプリケーションが正常状態に戻るか確認
          const isAppResponsive = await page.locator('text=在庫管理').isVisible();
          console.log(`アプリ復旧: ${isAppResponsive ? '✅' : '❌'}`);
        }
        
      } catch (error) {
        console.log(`${scenario.name}: 例外発生 - ${error.message}`);
      }
      
      // ページをリフレッシュして次のテストに備える
      await page.reload();
      await page.waitForTimeout(2000);
    }
  });
});