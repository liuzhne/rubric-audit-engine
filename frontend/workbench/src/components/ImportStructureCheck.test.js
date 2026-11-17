import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ImportStructureCheck from "@/components/ImportStructureCheck.vue";
import RubricImportWorkspace from "@/components/RubricImportWorkspace.vue";

const comparison = {
  rows: [
    { row_number: 3, change: "changed", fields: ["max_score"], current_name: "封面信息", current_max_score: 10,
      proposed_name: "封面信息", proposed_max_score: 4, proposed_dimension: "一、基本信息" },
    { row_number: 4, change: "same", fields: [], current_name: "填写规范", current_max_score: 2,
      proposed_name: "填写规范", proposed_max_score: 2, proposed_dimension: "一、基本信息" },
    { row_number: 8, change: "removed", fields: [], current_name: "合计（满分 25 分）", current_max_score: 25,
      proposed_name: null, proposed_max_score: null, proposed_dimension: null },
  ],
  current_total: 60, proposed_total: 25, declared_total: 25,
  columns: [{ column: 2, header: "分值", current_field: "max_score", proposed_field: null },
    { column: 4, header: "分值", current_field: null, proposed_field: "max_score" }],
  reasons: [{ ref: "C4", reason: "每行一个分值" }],
  consistent: false,
};
const signals = [{ code: "E4", message: "评分项满分合计 60 与表头声明的满分 25 不一致。" }];

describe("导入时的表格结构自动识别", () => {
  it("识别中只显示进度和原因", () => {
    const wrapper = mount(ImportStructureCheck, { props: { check: { status: "running", signals } } });
    expect(wrapper.text()).toContain("满分合计 60");
    expect(wrapper.find('[data-test="structure-check-running"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="structure-check-apply"]').exists()).toBe(false);
  });

  it("有差异时列出变化的行、合计与列用途，按钮发出采用与保留", async () => {
    const wrapper = mount(ImportStructureCheck, { props: { check: { status: "differs", signals, comparison } } });
    const rows = wrapper.findAll("tbody tr");
    expect(rows).toHaveLength(2);
    expect(rows[0].text()).toContain("10 → 4");
    expect(rows[0].text()).toContain("一、基本信息");
    expect(rows[1].text()).toContain("移除");
    expect(wrapper.get('[data-test="structure-check-total"]').text()).toContain("60 → 25");
    expect(wrapper.get('[data-test="structure-check-total"]').text()).toContain("与表上写的满分 25 一致");
    expect(wrapper.text()).toContain("分值（第 2 列）：满分 → 不使用");
    expect(wrapper.text()).toContain("分值（第 4 列）：未使用 → 满分");
    await wrapper.get('[data-test="structure-check-apply"]').trigger("click");
    await wrapper.get('[data-test="structure-check-dismiss"]').trigger("click");
    expect(wrapper.emitted("apply")).toHaveLength(1);
    expect(wrapper.emitted("dismiss")).toHaveLength(1);
  });

  it("结果一致时只显示一行小字", () => {
    const wrapper = mount(ImportStructureCheck, {
      props: { check: { status: "consistent", signals, comparison: { ...comparison, consistent: true } } },
    });
    expect(wrapper.get('[data-test="structure-check-consistent"]').text()).toContain("结果与当前一致");
    expect(wrapper.find("table").exists()).toBe(false);
  });

  it("已保留时默认收起，可以再展开，但不再显示按钮", async () => {
    const wrapper = mount(ImportStructureCheck, { props: { check: { status: "dismissed", signals, comparison } } });
    expect(wrapper.find('[data-test="structure-check-comparison"]').exists()).toBe(false);
    await wrapper.get("button").trigger("click");
    expect(wrapper.find('[data-test="structure-check-comparison"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="structure-check-apply"]').exists()).toBe(false);
  });

  it("识别失败时说明原因，不阻断核对", () => {
    const wrapper = mount(ImportStructureCheck, {
      props: { check: { status: "unavailable", signals, message: "当前没有可用的 AI 连接" } },
    });
    expect(wrapper.get('[data-test="structure-check-failed"]').text()).toContain("当前没有可用的 AI 连接");
  });

  it("导入工作区在评分项上方显示卡片，并把动作交给页面", async () => {
    const session = {
      id: "s1", status: "draft", state_version: 1, name: "开题报告", version: "v1", total_score: 60,
      criteria: [{ code: "C01", name: "封面信息", max_score: 60, source_refs: [], parse_status: "parsed" }],
      conflicts: [], structure_check: { status: "differs", signals, comparison },
    };
    const wrapper = mount(RubricImportWorkspace, { props: { session } });
    await wrapper.get('[data-test="structure-check-apply"]').trigger("click");
    await wrapper.get('[data-test="structure-check-dismiss"]').trigger("click");
    expect(wrapper.emitted("apply-structure")).toHaveLength(1);
    expect(wrapper.emitted("dismiss-structure")).toHaveLength(1);
    const persisted = mount(RubricImportWorkspace, { props: { session, persisted: true } });
    expect(persisted.find('[data-test="import-structure-check"]').exists()).toBe(false);
  });
});
