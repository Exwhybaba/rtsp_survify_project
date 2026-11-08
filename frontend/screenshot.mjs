// Headless-Chromium capture of the *running* dashboard, used by
// scripts/capture_dashboard_screenshot.py. Not part of the app build --
// this is a one-off dev/CI tool, which is why it lives next to
// node_modules (so `import "playwright"` resolves) instead of under src/.
//
// Usage: node screenshot.mjs <url> <output-path>
import { chromium } from "playwright";

const url = process.argv[2] ?? "http://localhost:4173";
const outPath = process.argv[3] ?? "../docs/screenshots/dashboard.png";

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1024 } });
  const consoleErrors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });

  // Real data, not a mockup: wait for at least one live-event row...
  await page.waitForSelector("table tbody tr", { timeout: 30000 });
  // ...and for the annotated detection frames to actually finish loading.
  await page.waitForFunction(
    () => {
      const imgs = Array.from(document.querySelectorAll(".detection-frame img"));
      return imgs.length > 0 && imgs.every((img) => img.complete && img.naturalWidth > 0);
    },
    { timeout: 30000 },
  );

  // Let the WS "connected" badge and any flash animations settle.
  await page.waitForTimeout(500);

  await page.screenshot({ path: outPath, fullPage: true });
  console.log(`Saved screenshot to ${outPath}`);
  if (consoleErrors.length > 0) {
    console.warn("Browser console errors during capture:", consoleErrors);
  }
} finally {
  await browser.close();
}
