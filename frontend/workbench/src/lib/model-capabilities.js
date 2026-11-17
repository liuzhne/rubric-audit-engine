/**
 * 模型管理页的表单换算（docs/模型管理页与评分思考修正方案.md 3.2）。
 *
 * 请求写法在表单里是 JSON 文本，提交前解析；解析不了就在前端拦下，不发请求。
 * 白名单、大小与嵌套的校验以服务端为准，这里只做能不能解析。
 */

export const PROTOCOL_LABELS = { chat: "Chat 兼容", responses: "Responses", messages: "Claude Messages" };

/** @returns {Record<string, any>} */
export function emptyForm() {
  return {
    id: null,
    revision: null,
    protocol: "chat",
    host_family: "other",
    model_prefix: "",
    can_disable: false,
    off_request: "",
    lowest_effort: "",
    effort_request: "",
    max_output_tokens: "",
    enabled: true,
    note: "",
    source_url: "",
    verified_on: new Date().toISOString().slice(0, 10),
  };
}

function jsonText(value) {
  return value == null ? "" : JSON.stringify(value, null, 2);
}

/**
 * 以一行为底稿：管理员记录是编辑它，代码记录是“覆盖”（新建一条同键的管理员记录）。
 * 代码记录可能写着多个协议或主机族（用 + 连接），覆盖时取第一个，管理员可以再改。
 */
export function formFromEntry(entry) {
  const editing = entry.source === "admin";
  return {
    id: editing ? entry.id : null,
    revision: editing ? entry.revision : null,
    protocol: String(entry.protocol).split("+")[0],
    host_family: String(entry.host_family).split("+")[0],
    model_prefix: entry.model_prefix,
    can_disable: Boolean(entry.can_disable),
    off_request: jsonText(entry.off_request),
    lowest_effort: entry.lowest_effort || "",
    effort_request: jsonText(entry.effort_request),
    max_output_tokens: entry.max_output_tokens == null ? "" : String(entry.max_output_tokens),
    enabled: editing ? Boolean(entry.enabled) : true,
    note: entry.note || "",
    source_url: (entry.sources || []).find((item) => String(item).startsWith("https://")) || "",
    verified_on: editing ? entry.verified_on : new Date().toISOString().slice(0, 10),
  };
}

function parseJson(text, label) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return null;
  try {
    const value = JSON.parse(trimmed);
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("not an object");
    }
    return value;
  } catch {
    throw new Error(`${label}不是合法的 JSON 对象`);
  }
}

/**
 * 表单 → 请求体。新建带匹配键；编辑带 revision，不带匹配键（键不能改）。
 * 能关闭的模型只发关闭写法，关不掉的只发强度：另一组字段清空，免得把旧值带上去。
 */
export function payloadFromForm(form) {
  const canDisable = Boolean(form.can_disable);
  const body = {
    can_disable: canDisable,
    off_request: canDisable ? parseJson(form.off_request, "关闭写法") : null,
    lowest_effort: canDisable ? null : String(form.lowest_effort || "").trim() || null,
    effort_request: canDisable ? null : parseJson(form.effort_request, "最低强度写法"),
    max_output_tokens: String(form.max_output_tokens || "").trim() ? Number(form.max_output_tokens) : null,
    enabled: Boolean(form.enabled),
    note: String(form.note || "").trim(),
    source_url: String(form.source_url || "").trim(),
    verified_on: form.verified_on,
  };
  if (form.id) return { ...body, revision: form.revision };
  return {
    ...body,
    protocol: form.protocol,
    host_family: form.host_family,
    model_prefix: String(form.model_prefix || "").trim().toLowerCase(),
  };
}

/** 保存前预览：这条记录会合并进请求体的字段。 */
export function requestPreview(form) {
  try {
    const body = payloadFromForm(form);
    return JSON.stringify((body.can_disable ? body.off_request : body.effort_request) || {}, null, 2);
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}
