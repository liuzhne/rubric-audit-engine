<script setup>
import { computed, ref } from "vue";

/**
 * 导入时的表格结构自动识别（docs/评分表结构自动识别方案.md §6）。
 *
 * 只展示代码按模型结构从原表取出的数据：每行的名称、满分（当前 → AI）、所属维度，
 * 以及列用途的变化；模型的判断依据放在折叠区。采用后才替换草稿，保留只记选择。
 */
const props = defineProps({
  check: { type: Object, default: null },
  busy: { type: Boolean, default: false },
  readonly: { type: Boolean, default: false },
});
const emit = defineEmits(["apply", "dismiss"]);

const FIELD = {
  name: "评分项名称", max_score: "满分", dimension: "所属维度", description: "说明", code: "编号",
  item_label: "打分项", deduction_rules: "扣分规则", rubric_levels: "分档", evidence_hints: "依据",
  weight: "权重", display_order: "顺序", criterion_type: "判定类型", applies_to: "适用范围", sub_checks: "子检查",
};
const CHANGE = { changed: "有变化", added: "新增", removed: "移除", same: "一致" };

const expanded = ref(false);
const status = computed(() => props.check?.status || "");
const comparison = computed(() => props.check?.comparison || null);
const reason = computed(() => {
  const parts = (props.check?.signals || []).map(item => String(item.message || "").replace(/[。.]$/, "")).filter(Boolean);
  return parts.length ? `${parts.join("；")}。` : "";
});
const title = computed(() => ({
  applied: "已按 AI 识别的表格结构重新解析",
  dismissed: "表格结构核对",
}[status.value] || "表格结构可能识别有误"));
const changedRows = computed(() => (comparison.value?.rows || []).filter(row => row.change !== "same"));
const open = computed(() => status.value === "differs" || expanded.value);
const columnText = (column) => {
  const label = column.header ? `${column.header}（第 ${column.column} 列）` : `第 ${column.column} 列`;
  const before = column.current_field ? FIELD[column.current_field] || column.current_field : "未使用";
  const after = column.proposed_field ? FIELD[column.proposed_field] || column.proposed_field : "不使用";
  return `${label}：${before} → ${after}`;
};
const score = (value) => (value == null ? "—" : Number(value).toLocaleString("zh-CN"));
const totalMatches = computed(() => comparison.value?.declared_total != null
  && Number(comparison.value.proposed_total) === Number(comparison.value.declared_total));
</script>

<template>
  <section v-if="check" class="card structure-check" :class="`status-${status}`" data-test="import-structure-check">
    <template v-if="status === 'consistent'">
      <p class="faint quiet" data-test="structure-check-consistent">AI 已核对表格结构，结果与当前一致。</p>
    </template>
    <template v-else>
      <div class="head">
        <div>
          <h2 class="card-title">{{ title }}</h2>
          <p class="card-note">{{ reason }}</p>
        </div>
        <button v-if="status === 'dismissed' || status === 'applied' || status === 'stale'" class="btn btn-sm" type="button"
          @click="expanded = !expanded">{{ expanded ? "收起对照" : "查看对照" }}</button>
      </div>

      <p v-if="status === 'running'" class="progress" role="status" data-test="structure-check-running">AI 正在识别表格结构，完成后在这里显示对照。你可以先核对下方结果。</p>
      <p v-else-if="status === 'unavailable' || status === 'failed'" class="notice notice-warn" data-test="structure-check-failed">
        没能自动识别表格结构：{{ String(check.message || "AI 不可用").replace(/[。.]$/, "") }}。请对照原表核对下方评分项。
      </p>
      <p v-else-if="status === 'applied'" class="faint">已采用 AI 识别的结构，下方评分项已按新结构重新解析。</p>
      <p v-else-if="status === 'dismissed'" class="faint">已保留当前解析结果。</p>
      <p v-else-if="status === 'stale'" class="notice notice-warn">{{ check.message }}</p>

      <div v-if="comparison && open" class="comparison" data-test="structure-check-comparison">
        <div class="table-wrap">
          <table>
            <thead><tr><th>行</th><th>评分项</th><th class="num">满分（当前 → AI）</th><th>所属维度（AI）</th><th>变化</th></tr></thead>
            <tbody>
              <tr v-for="row in changedRows" :key="row.row_number" :class="row.change">
                <td>{{ row.row_number }}</td>
                <td>
                  <template v-if="row.change === 'changed' && row.fields.includes('name')">{{ row.current_name }} → {{ row.proposed_name }}</template>
                  <template v-else>{{ row.proposed_name || row.current_name }}</template>
                </td>
                <td class="num">{{ score(row.current_max_score) }} → {{ score(row.proposed_max_score) }}</td>
                <td>{{ row.proposed_dimension || "—" }}</td>
                <td>{{ CHANGE[row.change] }}</td>
              </tr>
              <tr v-if="!changedRows.length"><td colspan="5" class="faint">逐行结果一致。</td></tr>
            </tbody>
          </table>
        </div>
        <p class="total" data-test="structure-check-total">
          合计：{{ score(comparison.current_total) }} → {{ score(comparison.proposed_total) }}
          <span v-if="comparison.declared_total != null" :class="totalMatches ? 'ok' : 'warn'">
            {{ totalMatches ? `与表上写的满分 ${score(comparison.declared_total)} 一致` : `表上写的满分是 ${score(comparison.declared_total)}` }}
          </span>
        </p>
        <details v-if="comparison.columns?.length" class="detail">
          <summary>列用途的变化（{{ comparison.columns.length }}）</summary>
          <ul><li v-for="column in comparison.columns" :key="column.column">{{ columnText(column) }}</li></ul>
        </details>
        <details v-if="comparison.reasons?.length" class="detail">
          <summary>AI 的判断依据</summary>
          <ul><li v-for="item in comparison.reasons" :key="item.ref"><code>{{ item.ref }}</code> {{ item.reason }}</li></ul>
        </details>
        <div v-if="status === 'differs' && !readonly" class="actions">
          <button class="btn" type="button" data-test="structure-check-dismiss" :disabled="busy" @click="emit('dismiss')">保留当前结果</button>
          <button class="btn btn-primary" type="button" data-test="structure-check-apply" :disabled="busy" @click="emit('apply')">采用 AI 识别结果</button>
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.structure-check { padding: 18px 22px; border-color: #e9d9b4; background: #fffcf4; }
.structure-check.status-consistent, .structure-check.status-applied, .structure-check.status-dismissed { border-color: var(--line, #e2e5de); background: var(--surface, #fff); }
.structure-check.status-consistent { padding: 10px 22px; }
.quiet { margin: 0; font-size: 12px; }
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.progress { font-size: 13px; color: #956624; margin: 10px 0 0; }
.comparison { margin-top: 12px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #eee8da; vertical-align: top; }
th { font-weight: 500; color: #7b8274; white-space: nowrap; }
td.num, th.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
tr.removed td { color: #8d9587; text-decoration: line-through; }
.total { font-size: 13px; margin: 10px 0 0; display: flex; gap: 10px; flex-wrap: wrap; }
.total .ok { color: #175c50; }
.total .warn { color: #a34a2a; }
.detail { margin-top: 10px; font-size: 13px; }
.detail summary { cursor: pointer; color: #175c50; }
.detail ul { margin: 6px 0 0; padding-left: 18px; }
.actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
@media (max-width: 700px) { .structure-check { padding: 14px 16px; } .actions .btn { flex: 1; } }
</style>
