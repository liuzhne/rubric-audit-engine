/**
 * 连接测试顺带报告的思考情况 → 给用户看的一句提示（docs/模型管理页与评分思考修正方案.md 3.1）。
 *
 * 探测请求不带思考参数，能关闭思考的模型这里也可能在思考，所以只在两种情况下提示：
 * 能力表记着关不掉（评分会按最低强度发送），或者观测到思考而表里没有这个模型（评分会被停下）。
 *
 * @param {{ thinking?: string, thinking_capability?: string } | null | undefined} probe
 * @returns {string | null}
 */
export function thinkingNotice(probe) {
  if (!probe) return null;
  if (probe.thinking_capability === "forced") {
    return "这个模型关不掉思考：评分会按最低思考强度发送，耗时和费用会比不思考的模型高一些。";
  }
  if (probe.thinking === "observed" && probe.thinking_capability === "unknown") {
    return "测试时观测到这个模型在思考，但模型能力表里还没有它：直接评分会因为“没有关闭思考”而停下。请先把模型名转告平台管理员在「模型管理」里登记，或换一个能关闭思考的模型。";
  }
  return null;
}
