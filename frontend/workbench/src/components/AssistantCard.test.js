import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";

import { useAssistantStore } from "@/stores/assistant.js";

import AssistantCard from "./AssistantCard.vue";

beforeEach(() => {
  setActivePinia(createPinia());
});

const router = () => createRouter({ history: createMemoryHistory(), routes: [{ path: "/:any(.*)", component: { template: "<div />" } }] });

it("页面链接在新标签页打开：选好的待评分文件只在本页内存里，跳走就丢了", () => {
  const wrapper = mount(AssistantCard, {
    props: { message: { id: "m1" }, card: { id: "l1", type: "link", label: "去配置模型连接", path: "/account" } },
    global: { plugins: [router()] },
  });
  const link = wrapper.get("a");
  expect(link.attributes("href")).toBe("/account");
  expect(link.attributes("target")).toBe("_blank");
  expect(link.text()).toBe("去配置模型连接 →");
});

it("“要放弃上一步吗”：两个按钮分别放弃并开始、继续上一步；处理过后只显示结果", async () => {
  const store = useAssistantStore();
  store.switchFlow = vi.fn();
  const card = { id: "s1", type: "confirm_switch_flow", status: "proposed", from_label: "上传评分规则与模板", to_label: "开始评分" };
  const wrapper = mount(AssistantCard, { props: { message: { id: "m1" }, card }, global: { plugins: [router()] } });
  const [switchButton, keepButton] = wrapper.findAll("button");
  expect(switchButton.text()).toBe("放弃并开始评分");
  expect(keepButton.text()).toBe("继续“上传评分规则与模板”");
  await keepButton.trigger("click");
  expect(store.switchFlow).toHaveBeenCalledWith(card, "keep");

  await wrapper.setProps({ card: { ...card, status: "dismissed" } });
  expect(wrapper.text()).toBe("已继续上一步");
});

it("选择卡片：每个选项一个按钮，正在用的那个有标记；点了交给后端，选完只显示选了哪个", async () => {
  const store = useAssistantStore();
  store.select = vi.fn();
  const card = {
    id: "c1", type: "choice", status: "proposed", purpose: "use_rubric", title: "选择评分标准",
    options: [{ id: "r-a", label: "《开题报告》v1.0（通用文档）", current: true }, { id: "r-b", label: "《毕业论文评分标准》v1.0（毕业论文）" }],
  };
  const wrapper = mount(AssistantCard, { props: { message: { id: "m1" }, card }, global: { plugins: [router()] } });
  const buttons = wrapper.findAll("button");
  expect(buttons.map((item) => item.text())).toEqual(["《开题报告》v1.0（通用文档）正在用", "《毕业论文评分标准》v1.0（毕业论文）"]);
  await buttons[1].trigger("click");
  expect(store.select).toHaveBeenCalledWith("c1", { id: "r-b" });

  await wrapper.setProps({ card: { ...card, status: "done", result: { id: "r-b" } } });
  expect(wrapper.findAll("button")).toHaveLength(0);
  expect(wrapper.text()).toBe("已选：《毕业论文评分标准》v1.0（毕业论文）");
  await wrapper.setProps({ card: { ...card, status: "expired" } });
  expect(wrapper.text()).toBe("已失效");
});
