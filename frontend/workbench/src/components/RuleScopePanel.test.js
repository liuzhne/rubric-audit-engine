import { mount } from "@vue/test-utils";
import { expect, it } from "vitest";
import RuleScopePanel from "./RuleScopePanel.vue";

it("有待识别的规则才能发起识别，进行中显示进度并可停止", async () => {
  const idle = mount(RuleScopePanel, { props: { pending: 0, editable: true } });
  expect(idle.find('[data-test="rule-scope-run"]').element.disabled).toBe(true);
  expect(idle.text()).toContain("所有语义规则都已有适用范围");

  const ready = mount(RuleScopePanel, { props: { pending: 3, editable: true } });
  expect(ready.text()).toContain("3 条规则还没有适用范围");
  await ready.find('[data-test="rule-scope-run"]').trigger("click");
  expect(ready.emitted("run")).toHaveLength(1);

  const running = mount(RuleScopePanel, { props: { pending: 3, editable: true,
    task: { status: "running", total_items: 2, succeeded_count: 1, items: [] } } });
  expect(running.find('[data-test="rule-scope-run"]').element.disabled).toBe(true);
  expect(running.find('[data-test="rule-scope-progress"]').text()).toContain("1/2");
  await running.findAll("button").find((button) => button.text() === "停止").trigger("click");
  expect(running.emitted("task-action")[0]).toEqual(["cancel"]);
});

it("没有编辑权限时不能发起", () => {
  const wrapper = mount(RuleScopePanel, { props: { pending: 2, editable: false } });
  expect(wrapper.find('[data-test="rule-scope-run"]').element.disabled).toBe(true);
});
