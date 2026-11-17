import { expect, test } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

// 导入时的表格结构自动识别（docs/评分表结构自动识别方案.md）。验收环境用 Mock LLM，
// 不会真的识别；这里拦截会话接口，模拟“识别中 → 有差异 → 采用”，验证轮询、对照卡片与采用。
const root = resolve(process.cwd(), "../..");
const python = process.env.PGS_PYTHON || resolve(root, ".venv/bin/python");
const workbook = () => execFileSync(python, ["-c", `
from io import BytesIO
import sys
from openpyxl import Workbook
book = Workbook()
sheet = book.active
sheet.append(["评分项", "分值", "评分说明"])
sheet.append(["封面信息", 10, "每缺1项扣1分"])
sheet.append(["选题背景", 15, "充分8分"])
out = BytesIO()
book.save(out)
sys.stdout.buffer.write(out.getvalue())
`], { cwd: root });

const signals = [{ code: "E4", message: "评分项满分合计 25 与表头声明的满分 12 不一致。" }];
const comparison = {
  rows: [
    { row_number: 2, change: "changed", fields: ["max_score"], current_name: "封面信息", current_max_score: 10,
      proposed_name: "封面信息", proposed_max_score: 4, proposed_dimension: "一、基本信息" },
    { row_number: 3, change: "changed", fields: ["max_score"], current_name: "选题背景", current_max_score: 15,
      proposed_name: "选题背景", proposed_max_score: 8, proposed_dimension: "二、选题背景" },
  ],
  current_total: 25, proposed_total: 12, declared_total: 12,
  columns: [{ column: 2, header: "分值", current_field: "max_score", proposed_field: "max_score" }],
  reasons: [], consistent: false,
};

test("强信号导入：识别中轮询，出现对照后采用替换草稿", async ({ page }) => {
  let created = null;
  let reads = 0;
  await page.route(/\/api\/rubrics\/import-sessions$/, async (route) => {
    const response = await route.fetch();
    created = await response.json();
    await route.fulfill({ response, json: { ...created, structure_check: { status: "running", signals } } });
  });
  await page.route(/\/api\/rubrics\/import-sessions\/[^/]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const response = await route.fetch();
    const body = await response.json();
    reads += 1;
    await route.fulfill({ response, json: { ...body, structure_check: { status: "differs", signals, comparison } } });
  });
  await page.route(/\/structure\/apply$/, async (route) => {
    const criteria = created.criteria.map((item, index) => ({ ...item, max_score: index ? 8 : 4 }));
    await route.fulfill({ json: {
      ...created, state_version: created.state_version + 1, total_score: 12, criteria,
      structure_check: { status: "applied", signals, comparison },
    } });
  });

  await page.goto("/workbench/rubrics");
  await page.getByRole("button", { name: "新建评分标准", exact: true }).click();
  const panel = page.locator(".import-panel");
  await panel.getByLabel("标准名称", { exact: true }).fill(`结构识别-${Date.now()}`);
  await panel.getByLabel("评分表", { exact: true }).setInputFiles({
    name: "scores.xlsx",
    mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    buffer: workbook(),
  });
  await panel.getByRole("button", { name: "解析文件", exact: true }).click();

  const card = page.locator("[data-test=import-structure-check]");
  await expect(card.locator("[data-test=structure-check-running]")).toBeVisible();
  await expect(card).toContainText("满分合计 25");

  // 识别中点“下一步”要二次确认；取消则留在本页
  page.once("dialog", (dialog) => {
    expect(dialog.message()).toContain("表格结构可能识别有误");
    return dialog.dismiss();
  });
  await page.locator("[data-test=confirm-import-session]").click();
  await expect(page.locator("[data-test=rubric-import-workspace]")).toBeVisible();

  // 轮询拿到对照
  await expect(card.locator("[data-test=structure-check-comparison]")).toBeVisible({ timeout: 10_000 });
  expect(reads).toBeGreaterThan(0);
  await expect(card).toContainText("10 → 4");
  await expect(card.locator("[data-test=structure-check-total]")).toContainText("25 → 12");

  await card.locator("[data-test=structure-check-apply]").click();
  await expect(card).toContainText("已按 AI 识别的表格结构重新解析");
  await expect(page.locator(".criteria-card")).toContainText("合计 12 分");
});
