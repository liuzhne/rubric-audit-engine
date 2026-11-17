import { describe, expect, it } from "vitest";

import { emptyForm, formFromEntry, payloadFromForm, requestPreview } from "./model-capabilities.js";

const codeEntry = {
  source: "code", id: null, protocol: "chat", host_family: "zhipu", model_prefix: "glm-5.3",
  can_disable: false, off_request: null, lowest_effort: "low", effort_request: { reasoning_effort: "low" },
  max_output_tokens: 128000, note: "强制思考", sources: ["https://docs.bigmodel.cn/x", "生产验证"],
  verified_on: "2026-11-10", enabled: true,
};

describe("model capability form", () => {
  it("覆盖代码记录时新建一条同键的管理员记录", () => {
    const form = formFromEntry(codeEntry);
    expect(form.id).toBeNull();
    expect(form.source_url).toBe("https://docs.bigmodel.cn/x");
    const body = payloadFromForm(form);
    expect(body).toMatchObject({
      protocol: "chat", host_family: "zhipu", model_prefix: "glm-5.3", can_disable: false,
      lowest_effort: "low", effort_request: { reasoning_effort: "low" }, off_request: null, max_output_tokens: 128000,
    });
    expect(body.revision).toBeUndefined();
  });

  it("多协议的代码记录取第一个协议与主机族", () => {
    const form = formFromEntry({ ...codeEntry, protocol: "chat+responses", host_family: "vertex+gemini_api" });
    expect([form.protocol, form.host_family]).toEqual(["chat", "vertex"]);
  });

  it("编辑管理员记录只发可改字段与 revision", () => {
    const body = payloadFromForm(formFromEntry({ ...codeEntry, source: "admin", id: "a1", revision: 3 }));
    expect(body.revision).toBe(3);
    expect(body.protocol).toBeUndefined();
  });

  it("能关闭思考时清空强度字段", () => {
    const form = { ...emptyForm(), can_disable: true, off_request: '{"thinking":{"type":"disabled"}}',
      lowest_effort: "low", effort_request: '{"reasoning_effort":"low"}', model_prefix: " GLM-X " };
    const body = payloadFromForm(form);
    expect(body).toMatchObject({ off_request: { thinking: { type: "disabled" } }, lowest_effort: null,
      effort_request: null, model_prefix: "glm-x" });
  });

  it("JSON 写错时报出是哪个字段", () => {
    expect(() => payloadFromForm({ ...emptyForm(), effort_request: "{oops" })).toThrow("最低强度写法");
    expect(() => payloadFromForm({ ...emptyForm(), effort_request: "[1]" })).toThrow("JSON 对象");
    expect(requestPreview({ ...emptyForm(), effort_request: "{oops" })).toContain("最低强度写法");
    expect(requestPreview({ ...emptyForm(), effort_request: '{"reasoning_effort":"low"}' })).toContain("reasoning_effort");
  });
});
