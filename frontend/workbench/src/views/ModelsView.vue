<script setup>
import { computed, onMounted, reactive, ref } from "vue";

import { api, StaleContextError } from "@/api/client.js";
import {
  PROTOCOL_LABELS,
  emptyForm,
  formFromEntry,
  payloadFromForm,
  requestPreview,
} from "@/lib/model-capabilities.js";
import { useSessionStore } from "@/stores/session.js";

/**
 * 模型管理（docs/模型管理页与评分思考修正方案.md 3.2）。
 *
 * 只给平台管理员：能力记录全平台共享，改一条会影响所有学校。代码里的能力表是默认值，
 * 这里只能新增“管理员记录”去覆盖或补充它；停用即退回代码表。真正的权限由服务端判定。
 */
const session = useSessionStore();
const allowed = computed(() => session.can("view_platform_ops"));

const data = ref(null);
const loadError = ref(null);
const filter = ref("");
const form = reactive(emptyForm());
const editing = ref(false);
const busy = ref(false);
const formError = ref(null);
const notice = ref(null);

const items = computed(() => {
  const keyword = filter.value.trim().toLowerCase();
  const rows = data.value?.items || [];
  if (!keyword) return rows;
  return rows.filter((row) =>
    [row.model_prefix, row.host_family, row.protocol, row.note].some((text) =>
      String(text || "").toLowerCase().includes(keyword),
    ),
  );
});
const preview = computed(() => requestPreview(form));

async function load() {
  loadError.value = null;
  try {
    data.value = await api.get("/admin/model-capabilities");
  } catch (error) {
    if (!(error instanceof StaleContextError)) loadError.value = error?.message || "加载失败";
  }
}

function openForm(entry) {
  Object.assign(form, entry ? formFromEntry(entry) : emptyForm());
  formError.value = null;
  notice.value = null;
  editing.value = true;
}

function closeForm() {
  editing.value = false;
}

async function save() {
  formError.value = null;
  let body;
  try {
    body = payloadFromForm(form);
  } catch (error) {
    formError.value = error instanceof Error ? error.message : String(error);
    return;
  }
  busy.value = true;
  try {
    if (form.id) await api.patch(`/admin/model-capabilities/${form.id}`, body);
    else await api.post("/admin/model-capabilities", body);
    editing.value = false;
    notice.value = "已保存。新建的评分与 AI 任务最迟 30 秒后按这条记录发送；已经在跑的任务不受影响。";
    await load();
  } catch (error) {
    if (!(error instanceof StaleContextError)) formError.value = error?.message || "保存失败";
  } finally {
    busy.value = false;
  }
}

function capabilityText(row) {
  if (row.can_disable) return "能关闭";
  return row.lowest_effort ? `关不掉 · 最低 ${row.lowest_effort}` : "关不掉";
}

function protocolText(value) {
  return String(value)
    .split("+")
    .map((item) => PROTOCOL_LABELS[item] || item)
    .join(" / ");
}

onMounted(() => {
  if (allowed.value) load();
});
</script>

<template>
  <div class="models-page">
    <header class="page-head">
      <div class="page-eyebrow">运维</div>
      <div class="page-head-row">
        <h1 class="page-title">模型管理</h1>
        <button v-if="allowed" class="btn btn-primary" type="button" data-test="model-new" @click="openForm(null)">
          登记模型
        </button>
      </div>
      <p class="page-sub">
        记录每个模型能不能关闭思考、关不掉时怎么发最低强度。表里没有的模型按“能关闭”处理，
        关不掉的话评分会因为“没有关闭思考”而停下：遇到新模型先在这里登记。只依据厂商文档或生产日志填写。
      </p>
    </header>

    <p v-if="!allowed" class="notice notice-warn" role="alert">这一页只给平台管理员。</p>

    <template v-else>
      <p v-if="loadError" class="notice notice-danger" role="alert">{{ loadError }}</p>
      <p v-if="notice" class="notice" role="status" data-test="model-saved">{{ notice }}</p>

      <section v-if="editing" class="card card-pad editor" aria-label="登记模型" data-test="model-form">
        <h2 class="card-title">{{ form.id ? "编辑管理员记录" : "新建管理员记录" }}</h2>
        <div class="form-grid">
          <label class="field">
            <span class="field-label">接口协议</span>
            <select v-model="form.protocol" class="select" :disabled="Boolean(form.id)">
              <option v-for="(label, key) in PROTOCOL_LABELS" :key="key" :value="key">{{ label }}</option>
            </select>
          </label>
          <label class="field">
            <span class="field-label">主机族</span>
            <select v-model="form.host_family" class="select" :disabled="Boolean(form.id)">
              <option v-for="family in data?.host_families || []" :key="family" :value="family">{{ family }}</option>
            </select>
            <span class="field-hint">自建网关等认不出的地址都算 other。</span>
          </label>
          <label class="field">
            <span class="field-label">模型名前缀</span>
            <input v-model="form.model_prefix" class="input" :disabled="Boolean(form.id)" placeholder="glm-5.3" />
            <span class="field-hint">小写；前缀后紧跟版本号（如 -2、.1）不算命中，新版本单独登记。</span>
          </label>
          <label class="field">
            <span class="field-label">能否关闭思考</span>
            <select v-model="form.can_disable" class="select" data-test="model-can-disable">
              <option :value="true">能关闭</option>
              <option :value="false">关不掉（发最低强度）</option>
            </select>
          </label>
          <label v-if="form.can_disable" class="field wide">
            <span class="field-label">关闭写法（JSON，可留空表示省略即关闭）</span>
            <textarea v-model="form.off_request" class="input mono" rows="3" placeholder='{"thinking": {"type": "disabled"}}' />
          </label>
          <template v-else>
            <label class="field">
              <span class="field-label">最低强度名</span>
              <input v-model="form.lowest_effort" class="input" placeholder="low" />
            </label>
            <label class="field wide">
              <span class="field-label">最低强度写法（JSON）</span>
              <textarea v-model="form.effort_request" class="input mono" rows="3" placeholder='{"reasoning_effort": "low"}' />
            </label>
          </template>
          <label class="field">
            <span class="field-label">最大输出 token（可空）</span>
            <input v-model="form.max_output_tokens" class="input" inputmode="numeric" placeholder="不写取 32768" />
          </label>
          <label class="field">
            <span class="field-label">核实日期</span>
            <input v-model="form.verified_on" class="input" type="date" />
          </label>
          <label class="field wide">
            <span class="field-label">来源链接（厂商文档）</span>
            <input v-model="form.source_url" class="input" placeholder="https://" />
          </label>
          <label class="field wide">
            <span class="field-label">备注</span>
            <input v-model="form.note" class="input" maxlength="500" />
          </label>
          <label class="field">
            <span class="field-label">状态</span>
            <select v-model="form.enabled" class="select">
              <option :value="true">启用</option>
              <option :value="false">停用（退回代码表）</option>
            </select>
          </label>
        </div>
        <div class="preview">
          <span class="field-label">会合并进请求体的字段</span>
          <pre class="mono" data-test="model-preview">{{ preview }}</pre>
        </div>
        <p v-if="formError" class="notice notice-danger" role="alert">{{ formError }}</p>
        <div class="form-actions">
          <button class="btn btn-outline" type="button" @click="closeForm">取消</button>
          <button class="btn btn-primary" type="button" :disabled="busy" data-test="model-save" @click="save">
            {{ busy ? "保存中…" : "保存" }}
          </button>
        </div>
      </section>

      <section class="card">
        <div class="card-head">
          <h2 class="card-title">能力记录</h2>
          <input v-model="filter" class="input filter" placeholder="按模型名、主机族筛选" aria-label="筛选" />
        </div>
        <div class="table-wrap">
          <table class="table" data-test="model-table">
            <thead>
              <tr>
                <th>来源</th>
                <th>模型名前缀</th>
                <th>协议 · 主机族</th>
                <th>思考</th>
                <th>最大输出</th>
                <th>启用中连接</th>
                <th>核实</th>
                <th />
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in items" :key="row.id || row.key + row.protocol" :class="{ dim: row.overridden || !row.enabled }">
                <td>
                  <span class="tag" :class="row.source === 'admin' ? 'tag-accent' : 'tag-muted'">
                    {{ row.source === "admin" ? "管理员登记" : "代码" }}
                  </span>
                  <span v-if="row.overridden" class="faint"> · 已被覆盖</span>
                  <span v-if="!row.enabled" class="faint"> · 已停用</span>
                </td>
                <td class="mono">{{ row.model_prefix || "（整族默认）" }}</td>
                <td>{{ protocolText(row.protocol) }} · {{ row.host_family }}</td>
                <td>{{ capabilityText(row) }}</td>
                <td class="mono">{{ row.max_output_tokens ?? "—" }}</td>
                <td class="mono">{{ row.active_connections }}</td>
                <td>
                  {{ row.verified_on }}
                  <a v-for="link in row.sources.filter((item) => item.startsWith('https://')).slice(0, 1)" :key="link"
                     :href="link" target="_blank" rel="noopener noreferrer" class="faint">来源</a>
                </td>
                <td>
                  <button class="btn btn-sm btn-outline" type="button" @click="openForm(row)">
                    {{ row.source === "admin" ? "编辑" : "覆盖" }}
                  </button>
                </td>
              </tr>
              <tr v-if="data && !items.length">
                <td class="table-empty" colspan="8">没有匹配的记录</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section v-if="data" class="card card-pad">
        <h2 class="card-title">评分助手 Agent</h2>
        <p class="card-note">
          只有评测达标的模型才会进入 Agent 模式，其余模型由“先问模型、再按规则兜底”识别意图。
          放行新模型要先用真实模型跑评测，这里只读。
        </p>
        <p class="mono">{{ data.agent_evaluated.join("、") || "（暂无）" }}</p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.models-page { display: flex; flex-direction: column; gap: 18px; }
.editor .card-title { margin: 0 0 16px; }
.editor .preview { margin-top: 16px; }
.editor .form-actions { margin-top: 14px; }
.editor .form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.field.wide { grid-column: 1 / -1; }
.mono { font-family: var(--font-mono); font-size: 12px; }
textarea.mono { resize: vertical; }
.preview pre { margin: 6px 0 0; padding: 10px 12px; border-radius: var(--radius); background: var(--surface-muted, #f4f4f1); white-space: pre-wrap; }
.filter { max-width: 240px; }
tr.dim td { opacity: 0.55; }
</style>
