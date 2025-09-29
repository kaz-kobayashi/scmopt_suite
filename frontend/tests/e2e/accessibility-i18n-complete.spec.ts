import { test, expect } from '@playwright/test';

test.describe('Accessibility and Internationalization Complete Tests', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(90000);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
  });

  test.describe('完全なアクセシビリティテスト', () => {
    test('スクリーンリーダー対応の完全テスト', async ({ page }) => {
      console.log('🔊 スクリーンリーダー完全対応テスト');
      
      // ARIAラベルの確認
      const ariaElements = await page.evaluate(() => {
        const elements = {
          buttons: 0,
          inputs: 0,
          navigation: 0,
          headings: 0,
          tables: 0,
          forms: 0
        };
        
        // ボタンのアクセシビリティ
        document.querySelectorAll('button').forEach(btn => {
          if (btn.getAttribute('aria-label') || btn.textContent.trim()) {
            elements.buttons++;
          }
        });
        
        // 入力フィールドのラベル
        document.querySelectorAll('input, select, textarea').forEach(input => {
          if (input.getAttribute('aria-label') || 
              input.getAttribute('aria-labelledby') ||
              document.querySelector(`label[for="${input.id}"]`)) {
            elements.inputs++;
          }
        });
        
        // ナビゲーション構造
        document.querySelectorAll('[role="navigation"], nav').forEach(nav => {
          elements.navigation++;
        });
        
        // 見出し構造
        document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(heading => {
          elements.headings++;
        });
        
        // テーブルのアクセシビリティ
        document.querySelectorAll('table').forEach(table => {
          if (table.querySelector('caption') || table.getAttribute('aria-label')) {
            elements.tables++;
          }
        });
        
        // フォームのアクセシビリティ
        document.querySelectorAll('form').forEach(form => {
          if (form.getAttribute('aria-label') || form.querySelector('legend')) {
            elements.forms++;
          }
        });
        
        return elements;
      });
      
      console.log('アクセシビリティ要素カウント:', ariaElements);
      
      // 各コンポーネントでのアクセシビリティテスト
      const components = [
        '在庫管理', '物流ネットワーク設計', 'シフト最適化', 
        'PyVRP', 'ジョブショップスケジューリング', '分析'
      ];
      
      for (const component of components) {
        await page.click(`text=${component}`);
        await page.waitForTimeout(2000);
        
        // フォーカス管理のテスト
        await page.keyboard.press('Tab');
        const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
        console.log(`${component}: フォーカス要素 = ${focusedElement}`);
        
        // アナウンスメントの確認
        const liveRegions = await page.locator('[aria-live], [role="alert"], [role="status"]').count();
        console.log(`${component}: ライブリージョン数 = ${liveRegions}`);
      }
    });

    test('キーボードナビゲーション完全テスト', async ({ page }) => {
      console.log('⌨️ キーボードナビゲーション完全テスト');
      
      // Tabキーでの全要素ナビゲーション
      const tabbableElements = [];
      let previousActiveElement = '';
      
      for (let i = 0; i < 50; i++) {
        await page.keyboard.press('Tab');
        await page.waitForTimeout(100);
        
        const currentElement = await page.evaluate(() => {
          const el = document.activeElement;
          return {
            tag: el?.tagName,
            text: el?.textContent?.trim().substring(0, 50),
            type: el?.getAttribute('type'),
            role: el?.getAttribute('role')
          };
        });
        
        const elementKey = JSON.stringify(currentElement);
        if (elementKey === previousActiveElement) {
          console.log('タブナビゲーション完了');
          break;
        }
        
        tabbableElements.push(currentElement);
        previousActiveElement = elementKey;
      }
      
      console.log(`タブ可能な要素数: ${tabbableElements.length}`);
      
      // 矢印キーナビゲーション
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // タブ間の矢印キー操作
      const tabs = await page.locator('[role="tab"]');
      if (await tabs.count() > 0) {
        await tabs.first().focus();
        
        for (let i = 0; i < 3; i++) {
          await page.keyboard.press('ArrowRight');
          await page.waitForTimeout(300);
          
          const activeTab = await page.evaluate(() => 
            document.activeElement?.textContent?.trim()
          );
          console.log(`アクティブタブ: ${activeTab}`);
        }
      }
      
      // Escapeキーでのダイアログクローズ
      const button = page.locator('button').first();
      if (await button.isVisible()) {
        await button.click();
        await page.waitForTimeout(1000);
        
        const dialogVisible = await page.locator('[role="dialog"]').isVisible();
        if (dialogVisible) {
          await page.keyboard.press('Escape');
          await page.waitForTimeout(500);
          
          const dialogClosed = !(await page.locator('[role="dialog"]').isVisible());
          console.log(`Escapeキーでのダイアログクローズ: ${dialogClosed ? '成功' : '失敗'}`);
        }
      }
    });

    test('色覚多様性対応テスト', async ({ page }) => {
      console.log('🎨 色覚多様性対応テスト');
      
      // コントラスト比の確認
      const contrastResults = await page.evaluate(() => {
        const results = [];
        
        function getLuminance(r, g, b) {
          const [rs, gs, bs] = [r, g, b].map(c => {
            c = c / 255;
            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
          });
          return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
        }
        
        function getContrastRatio(color1, color2) {
          const l1 = getLuminance(...color1);
          const l2 = getLuminance(...color2);
          const lmax = Math.max(l1, l2);
          const lmin = Math.min(l1, l2);
          return (lmax + 0.05) / (lmin + 0.05);
        }
        
        // テキスト要素のコントラストチェック
        document.querySelectorAll('*').forEach(el => {
          const style = window.getComputedStyle(el);
          if (el.textContent && el.textContent.trim() && style.color && style.backgroundColor) {
            const color = style.color.match(/\d+/g)?.map(Number);
            const bgColor = style.backgroundColor.match(/\d+/g)?.map(Number);
            
            if (color && bgColor && color.length >= 3 && bgColor.length >= 3) {
              const ratio = getContrastRatio(color.slice(0, 3), bgColor.slice(0, 3));
              
              if (ratio < 4.5) { // WCAG AA基準
                results.push({
                  element: el.tagName,
                  text: el.textContent.substring(0, 30),
                  ratio: ratio.toFixed(2),
                  pass: false
                });
              }
            }
          }
        });
        
        return results;
      });
      
      console.log(`低コントラスト要素数: ${contrastResults.length}`);
      contrastResults.slice(0, 5).forEach(result => {
        console.log(`- ${result.element}: "${result.text}" (比率: ${result.ratio})`);
      });
      
      // 色だけに依存しない情報伝達の確認
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      const colorOnlyElements = await page.evaluate(() => {
        const elements = [];
        
        document.querySelectorAll('[class*="error"], [class*="success"], [class*="warning"]').forEach(el => {
          const hasIcon = el.querySelector('svg, i, [class*="icon"]');
          const hasText = el.textContent && el.textContent.trim().length > 0;
          
          if (!hasIcon && !hasText) {
            elements.push({
              class: el.className,
              issue: '色のみで状態を表現'
            });
          }
        });
        
        return elements;
      });
      
      console.log(`色のみに依存する要素: ${colorOnlyElements.length}`);
    });

    test('支援技術との互換性テスト', async ({ page }) => {
      console.log('🦮 支援技術互換性テスト');
      
      // ランドマークの確認
      const landmarks = await page.evaluate(() => {
        const landmarkRoles = ['banner', 'navigation', 'main', 'complementary', 'contentinfo'];
        const found = {};
        
        landmarkRoles.forEach(role => {
          const elements = document.querySelectorAll(`[role="${role}"]`);
          found[role] = elements.length;
        });
        
        // HTML5セマンティック要素
        found['header'] = document.querySelectorAll('header').length;
        found['nav'] = document.querySelectorAll('nav').length;
        found['main'] = document.querySelectorAll('main').length;
        found['aside'] = document.querySelectorAll('aside').length;
        found['footer'] = document.querySelectorAll('footer').length;
        
        return found;
      });
      
      console.log('ランドマーク要素:', landmarks);
      
      // フォームのアクセシビリティ
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      const formAccessibility = await page.evaluate(() => {
        const results = {
          totalInputs: 0,
          labeledInputs: 0,
          requiredIndicators: 0,
          errorMessages: 0,
          fieldsets: 0
        };
        
        const inputs = document.querySelectorAll('input, select, textarea');
        results.totalInputs = inputs.length;
        
        inputs.forEach(input => {
          // ラベルの確認
          if (input.labels?.length > 0 || 
              input.getAttribute('aria-label') || 
              input.getAttribute('aria-labelledby')) {
            results.labeledInputs++;
          }
          
          // 必須フィールドの表示
          if (input.required || input.getAttribute('aria-required') === 'true') {
            results.requiredIndicators++;
          }
          
          // エラーメッセージの関連付け
          if (input.getAttribute('aria-describedby') || input.getAttribute('aria-errormessage')) {
            results.errorMessages++;
          }
        });
        
        results.fieldsets = document.querySelectorAll('fieldset').length;
        
        return results;
      });
      
      console.log('フォームアクセシビリティ:', formAccessibility);
    });
  });

  test.describe('完全な国際化テスト', () => {
    test('日本語・英語・中国語の言語切り替えテスト', async ({ page }) => {
      console.log('🌐 多言語切り替えテスト');
      
      // 言語切り替えボタンを探す
      const langButton = page.locator('button:has-text("言語"), button:has-text("Language"), [aria-label*="language"]').first();
      
      if (await langButton.isVisible()) {
        await langButton.click();
        await page.waitForTimeout(1000);
        
        // 利用可能な言語オプション
        const languages = ['日本語', 'English', '中文'];
        
        for (const lang of languages) {
          const langOption = page.locator(`text=${lang}`).first();
          if (await langOption.isVisible()) {
            await langOption.click();
            await page.waitForTimeout(2000);
            
            // UI要素が正しく翻訳されているか確認
            const translations = {
              '日本語': ['在庫管理', '物流ネットワーク設計', 'シフト最適化'],
              'English': ['Inventory Management', 'Logistics Network Design', 'Shift Optimization'],
              '中文': ['库存管理', '物流网络设计', '排班优化']
            };
            
            if (translations[lang]) {
              for (const text of translations[lang]) {
                const isVisible = await page.locator(`text=${text}`).isVisible();
                console.log(`${lang} - "${text}": ${isVisible ? '表示' : '非表示'}`);
              }
            }
          }
        }
      }
    });

    test('RTL（右から左）言語対応テスト', async ({ page }) => {
      console.log('↔️ RTL言語対応テスト');
      
      // アラビア語やヘブライ語のテストデータ
      const rtlTestData = {
        arabic: 'مُنتَج اختبار للمخزون',
        hebrew: 'מוצר בדיקה למלאי'
      };
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // RTLテキスト入力
      const textInput = page.locator('input[type="text"]').first();
      if (await textInput.isVisible()) {
        for (const [lang, text] of Object.entries(rtlTestData)) {
          await textInput.clear();
          await textInput.fill(text);
          await page.waitForTimeout(500);
          
          // テキストの方向性を確認
          const direction = await textInput.evaluate(el => 
            window.getComputedStyle(el).direction
          );
          
          console.log(`${lang}テキスト方向: ${direction}`);
          
          // RTLレイアウトの確認
          const isRtlLayout = await page.evaluate(() => {
            const body = document.body;
            return window.getComputedStyle(body).direction === 'rtl';
          });
          
          console.log(`RTLレイアウト適用: ${isRtlLayout ? 'あり' : 'なし'}`);
        }
      }
    });

    test('日付・時刻・数値・通貨のローカライゼーション', async ({ page }) => {
      console.log('📅 ローカライゼーション完全テスト');
      
      // 各ロケールでのフォーマット確認
      const locales = ['ja-JP', 'en-US', 'zh-CN', 'de-DE', 'fr-FR'];
      
      for (const locale of locales) {
        // ブラウザのロケールを変更（シミュレート）
        await page.evaluate((loc) => {
          // 日付フォーマット
          const date = new Date('2024-01-15T14:30:00');
          console.log(`${loc} 日付: ${date.toLocaleDateString(loc)}`);
          console.log(`${loc} 時刻: ${date.toLocaleTimeString(loc)}`);
          
          // 数値フォーマット
          const number = 1234567.89;
          console.log(`${loc} 数値: ${number.toLocaleString(loc)}`);
          
          // 通貨フォーマット
          const currency = new Intl.NumberFormat(loc, {
            style: 'currency',
            currency: loc.includes('JP') ? 'JPY' : 
                     loc.includes('CN') ? 'CNY' :
                     loc.includes('US') ? 'USD' : 'EUR'
          }).format(number);
          console.log(`${loc} 通貨: ${currency}`);
        }, locale);
        
        await page.waitForTimeout(500);
      }
      
      // 実際のUIでの表示確認
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      // 日付入力フィールドのフォーマット確認
      const dateInputs = page.locator('input[type="date"]');
      const dateCount = await dateInputs.count();
      
      if (dateCount > 0) {
        const dateFormat = await dateInputs.first().evaluate(el => {
          return {
            placeholder: el.placeholder,
            pattern: el.pattern,
            value: el.value
          };
        });
        
        console.log('日付入力フォーマット:', dateFormat);
      }
    });

    test('タイムゾーン対応テスト', async ({ page }) => {
      console.log('🕐 タイムゾーン対応テスト');
      
      // 異なるタイムゾーンでの表示確認
      const timezones = [
        'Asia/Tokyo',
        'America/New_York',
        'Europe/London',
        'Australia/Sydney'
      ];
      
      for (const tz of timezones) {
        await page.evaluate((timezone) => {
          // タイムゾーンを変更（シミュレート）
          const date = new Date();
          const options = { timeZone: timezone, timeZoneName: 'short' };
          console.log(`${timezone}: ${date.toLocaleString('ja-JP', options)}`);
        }, tz);
      }
      
      // シフト最適化でのタイムゾーン考慮
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      // 時刻入力でのタイムゾーン表示
      const timeInputs = page.locator('input[type="time"]');
      const timeCount = await timeInputs.count();
      
      if (timeCount > 0) {
        console.log(`時刻入力フィールド数: ${timeCount}`);
        
        // タイムゾーン表示の確認
        const tzDisplay = await page.locator('text=JST, text=UTC, text=タイムゾーン').isVisible();
        console.log(`タイムゾーン表示: ${tzDisplay ? 'あり' : 'なし'}`);
      }
    });
  });

  test.describe('文化的配慮とローカルビジネス慣習', () => {
    test('地域別ビジネス慣習の反映', async ({ page }) => {
      console.log('🏢 地域別ビジネス慣習テスト');
      
      // 営業日・休日設定
      await page.click('text=シフト最適化');
      await page.waitForTimeout(2000);
      
      await page.click('text=システム設定');
      await page.waitForTimeout(1000);
      
      // 週の開始日の確認（日本: 月曜、米国: 日曜）
      const calendarElements = await page.locator('[class*="calendar"], [role="grid"]').count();
      if (calendarElements > 0) {
        const weekStart = await page.evaluate(() => {
          const calendar = document.querySelector('[class*="calendar"], [role="grid"]');
          const firstDay = calendar?.querySelector('[role="columnheader"]')?.textContent;
          return firstDay;
        });
        
        console.log(`週の開始日: ${weekStart}`);
      }
      
      // 祝日・休日の考慮
      const holidays = {
        'ja-JP': ['元日', '成人の日', 'ゴールデンウィーク'],
        'en-US': ['New Year', 'Independence Day', 'Thanksgiving'],
        'zh-CN': ['春节', '国庆节', '中秋节']
      };
      
      // 物流ネットワーク設計での地域差
      await page.click('text=物流ネットワーク設計');
      await page.waitForTimeout(2000);
      
      // 距離単位の確認（km vs miles）
      const distanceUnit = await page.locator('text=km, text=キロメートル, text=miles, text=マイル').first().textContent();
      console.log(`距離単位: ${distanceUnit}`);
      
      // 温度単位（倉庫管理）
      const tempUnit = await page.locator('text=℃, text=°C, text=°F').first().isVisible();
      console.log(`温度単位表示: ${tempUnit ? 'あり' : 'なし'}`);
    });

    test('文字エンコーディングと特殊文字処理', async ({ page }) => {
      console.log('🔤 文字エンコーディングテスト');
      
      const specialCharacters = {
        '全角文字': 'ＡＢＣ１２３',
        '半角カナ': 'ｱｲｳｴｵ',
        '機種依存文字': '①②③㈱㈲',
        '絵文字': '📦🚚💰📊',
        'Unicode結合文字': 'é (e + ́)',
        '制御文字': '\u200B\u200C\u200D', // ゼロ幅スペース
        'サロゲートペア': '𠀋𡈽𠮟', // 拡張漢字
        'BOM付き': '\uFEFFテスト'
      };
      
      await page.click('text=在庫管理');
      await page.waitForTimeout(2000);
      
      for (const [type, chars] of Object.entries(specialCharacters)) {
        const input = page.locator('input[type="text"]').first();
        if (await input.isVisible()) {
          await input.clear();
          await input.fill(chars);
          await page.waitForTimeout(300);
          
          const value = await input.inputValue();
          const isCorrect = value === chars || value === chars.trim();
          
          console.log(`${type}: ${isCorrect ? '正常' : '問題あり'}`);
          
          // データ送信時の処理確認
          const button = page.locator('button').first();
          if (await button.isVisible()) {
            await button.click();
            await page.waitForTimeout(1000);
            
            // エラーの確認
            const hasError = await page.locator('[class*="error"]').isVisible();
            console.log(`${type}でのエラー: ${hasError ? 'あり' : 'なし'}`);
          }
        }
      }
    });
  });

  test('完全なフォントとレイアウトテスト', async ({ page }) => {
    console.log('🔤 フォントとレイアウト完全テスト');
    
    // フォントの読み込み確認
    const fontInfo = await page.evaluate(() => {
      const fonts = [];
      const elements = document.querySelectorAll('*');
      
      elements.forEach(el => {
        const style = window.getComputedStyle(el);
        const fontFamily = style.fontFamily;
        
        if (fontFamily && !fonts.includes(fontFamily)) {
          fonts.push(fontFamily);
        }
      });
      
      return fonts;
    });
    
    console.log('使用フォント:', fontInfo);
    
    // 長いテキストでのレイアウト崩れ確認
    const longTexts = {
      '日本語長文': '在庫管理システムにおける最適発注量計算アルゴリズムの実装と検証結果レポート',
      '英語長文': 'Implementation and Verification Results Report of Optimal Order Quantity Calculation Algorithm in Inventory Management System',
      '混在長文': '在庫管理システム(Inventory Management System)における最適化アルゴリズム'
    };
    
    await page.click('text=在庫管理');
    await page.waitForTimeout(2000);
    
    for (const [type, text] of Object.entries(longTexts)) {
      const input = page.locator('input[type="text"]').first();
      if (await input.isVisible()) {
        await input.clear();
        await input.fill(text);
        await page.waitForTimeout(500);
        
        // オーバーフローの確認
        const hasOverflow = await input.evaluate(el => {
          return el.scrollWidth > el.clientWidth;
        });
        
        console.log(`${type}のオーバーフロー: ${hasOverflow ? 'あり' : 'なし'}`);
      }
    }
    
    // レスポンシブフォントサイズ
    const viewportSizes = [
      { width: 320, name: 'Mobile S' },
      { width: 768, name: 'Tablet' },
      { width: 1920, name: 'Desktop' }
    ];
    
    for (const size of viewportSizes) {
      await page.setViewportSize({ width: size.width, height: 800 });
      await page.waitForTimeout(1000);
      
      const fontSize = await page.evaluate(() => {
        const body = document.body;
        return window.getComputedStyle(body).fontSize;
      });
      
      console.log(`${size.name}のフォントサイズ: ${fontSize}`);
    }
  });
});