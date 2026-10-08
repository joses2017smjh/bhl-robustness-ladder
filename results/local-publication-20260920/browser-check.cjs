const { chromium } = require('/tmp/robotics-portfolio-stvWjr/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const output = '/tmp/ml-robotics-publication-20260920';
const base = 'http://127.0.0.1:4322';
const routes = [
  ['home', '/'],
  ['spur', '/projects/depth-estimation-robotic-pruning'],
  ['bhl', '/projects/bhl-robustness-ladder'],
  ['pruning', '/projects/isaac-pruning-workflow'],
  ['metanavit', '/projects/metanavit'],
  ['folding', '/projects/isaac-folding'],
];

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/tmp/robotics-portfolio-stvWjr/browsers/chromium-1243/chrome-linux64/chrome',
  });
  const results = [];
  try {
    for (const width of [320, 390, 768, 1440]) {
      for (const [name, route] of routes) {
        const page = await browser.newPage({ viewport: { width, height: width >= 768 ? 1000 : 844 }, reducedMotion: 'reduce' });
        const errors = [], failedResponses = [], failedRequests = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('response', response => {
          if (response.status() >= 400) failedResponses.push({ url: response.url(), status: response.status() });
        });
        page.on('requestfailed', request => {
          if (request.failure()?.errorText !== 'net::ERR_ABORTED') failedRequests.push({ url: request.url(), error: request.failure()?.errorText });
        });
        try {
          const response = await page.goto(base + route, { waitUntil: 'networkidle' });
          assert.equal(response.status(), 200);
          if (name === 'home') {
            assert.match(await page.title(), /ML\s*\/\s*AI\s*&\s*Robotics/);
            assert.equal(await page.locator('#projects .project-grid > li').count(), 4);
          }
          if (name === 'folding') {
            const text = await page.locator('main').innerText();
            assert.match(text, /8\/24/);
            assert.match(text, /3\/24/);
            assert.match(text, /not.*new adaptation/i);
          }
          await page.evaluate(async () => {
            const images = [...document.images];
            for (const image of images) image.loading = 'eager';
            await Promise.all(images.map(image => image.decode().catch(() => {})));
            await document.fonts.ready;
          });
          await page.keyboard.press('Tab');
          const focused = page.locator(':focus');
          assert.equal(await focused.count(), 1);
          const focus = await focused.evaluate(element => {
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return { text: element.textContent.trim(), href: element.getAttribute('href'),
                     top: bounds.top, left: bounds.left, right: bounds.right,
                     outline: style.outlineStyle, outlineWidth: style.outlineWidth };
          });
          assert.equal(focus.href, '#main');
          assert.ok(focus.top >= 0 && focus.left >= 0 && focus.right <= width, JSON.stringify(focus));
          assert.notEqual(focus.outline, 'none');
          await page.keyboard.press('Tab');
          assert.equal(await page.locator(':focus').getAttribute('href'), '/');
          const dimensions = await page.evaluate(() => ({
            viewport: innerWidth, page: document.documentElement.scrollWidth,
            images: document.images.length,
            brokenImages: [...document.images].filter(image => !image.complete || !image.naturalWidth).map(image => image.src),
            videos: document.querySelectorAll('video').length,
            playingVideos: [...document.querySelectorAll('video')].filter(video => !video.paused).length,
            reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
          }));
          assert.ok(dimensions.page <= dimensions.viewport, JSON.stringify(dimensions));
          assert.deepEqual(dimensions.brokenImages, []);
          assert.equal(dimensions.playingVideos, 0, 'Reduced motion must suppress autoplay');
          assert.equal(dimensions.reducedMotion, true);
          if (dimensions.videos > 0) {
            assert.equal((await page.locator('#demo-playback').innerText()).trim(), 'Play demos');
            await page.locator('video').first().scrollIntoViewIfNeeded();
            await page.emulateMedia({ reducedMotion: 'no-preference' });
            await page.waitForTimeout(250);
            await page.emulateMedia({ reducedMotion: 'reduce' });
            await page.waitForFunction(() => [...document.querySelectorAll('video')].every(video => video.paused));
            assert.equal((await page.locator('#demo-playback').innerText()).trim(), 'Play demos');
          }
          await page.evaluate(() => { document.activeElement?.blur(); scrollTo(0, 0); });
          if ([390, 1440].includes(width) && ['home', 'folding'].includes(name)) {
            await page.screenshot({ path: `${output}/browser-${name}-${width}.png`, fullPage: true });
            await page.screenshot({ path: `${output}/browser-${name}-${width}-viewport.png` });
          }
          assert.deepEqual(errors, []);
          assert.deepEqual(failedResponses, []);
          assert.deepEqual(failedRequests, []);
          const result = { name, width, route, title: await page.title(), keyboardFocus: focus, ...dimensions, status: 'passed' };
          results.push(result);
          console.log(JSON.stringify({ name, width, images: dimensions.images, videos: dimensions.videos, status: 'passed' }));
        } catch (error) {
          results.push({ name, width, route, status: 'failed', error: String(error), errors, failedResponses, failedRequests });
          await page.screenshot({ path: `${output}/browser-failed-${name}-${width}.png`, fullPage: true });
          console.error(JSON.stringify(results.at(-1)));
        } finally {
          await page.close();
        }
        fs.writeFileSync(`${output}/browser-results.json`, JSON.stringify(results, null, 2) + '\n');
      }
    }
  } finally {
    await browser.close();
  }
  assert.equal(results.length, 24);
  assert.ok(results.every(result => result.status === 'passed'), 'Browser failures recorded in browser-results.json');
  console.log('PASS: 24 page/viewport combinations, keyboard focus, loaded images, JavaScript and reduced-motion video behavior.');
})().catch(error => { console.error(error); process.exit(1); });
