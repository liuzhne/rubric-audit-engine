import { describe, expect, it } from "vitest";

import { thinkingNotice } from "./connection-thinking.js";

describe("thinkingNotice", () => {
  it("提示关不掉思考的模型会按最低强度评分", () => {
    expect(thinkingNotice({ thinking: "observed", thinking_capability: "forced" })).toContain("最低思考强度");
  });

  it("观测到思考而能力表没有记录时提示登记", () => {
    expect(thinkingNotice({ thinking: "observed", thinking_capability: "unknown" })).toContain("平台管理员登记");
  });

  it("能关闭思考、没观测到或读不出来时不提示", () => {
    expect(thinkingNotice({ thinking: "observed", thinking_capability: "can_disable" })).toBeNull();
    expect(thinkingNotice({ thinking: "not_observed", thinking_capability: "unknown" })).toBeNull();
    expect(thinkingNotice({})).toBeNull();
    expect(thinkingNotice(null)).toBeNull();
  });
});
