<script setup>
import { computed } from "vue";

import { canRetryAiTask, isActiveAiTask, scopeTaskProgress } from "@/lib/ai-tasks.js";

/**
 * 适用范围识别（docs/评分细则适用范围方案.md 第 6 节）。
 *
 * 适用范围由模型给出、老师不填写：起草的细则自带范围；不经过起草的规则（Excel 已解析的
 * 扣分规则、导入的原子规则）在这里发起一次识别。识别只看评分项与规则文字，不看论文；
 * 完成后写进草稿规则，不改变规则的确认状态。没有范围的规则评分时按全文处理。
 */
const props = defineProps({
  /** 还没有范围的语义规则数（参与评分的）。 */
  pending: { type: Number, default: 0 },
  busy: { type: Boolean, default: false },
  editable: { type: Boolean, default: false },
  /** 最近一次识别任务（进行中、失败或刚完成）。 */
  task: { type: Object, default: null },
});
const emit = defineEmits(["run", "task-action"]);
const running = computed(() => isActiveAiTask(props.task));
const progress = computed(() => scopeTaskProgress(props.task));
</script>

<template>
  <section class="card card-pad" data-test="rule-scope-panel">
    <h3 class="card-title">适用范围</h3>
    <p class="card-note">
      每条规则评分时看论文的哪一部分（全文、某一章或几章对照）由 AI 识别，不需要手动填写。起草的细则已自带范围；
      其余规则可在这里识别，只看评分项与规则文字，不看论文。没有范围的规则按全文处理。
    </p>
    <p data-test="rule-scope-pending">{{ pending ? `${pending} 条规则还没有适用范围。` : '所有语义规则都已有适用范围。' }}</p>
    <p v-if="progress" class="faint" data-test="rule-scope-progress">{{ progress }}</p>
    <div class="actions">
      <button class="btn" type="button" data-test="rule-scope-run"
        :disabled="busy || !editable || running || !pending" @click="emit('run')">识别适用范围</button>
      <button v-if="running" class="btn btn-sm" type="button" :disabled="busy" @click="emit('task-action', 'cancel')">停止</button>
      <button v-if="canRetryAiTask(task)" class="btn btn-sm" type="button" :disabled="busy" @click="emit('task-action', 'retry')">重试失败的评分项</button>
    </div>
  </section>
</template>
