<script setup>
/**
 * 选择卡片：在几个对象里点一个（docs/评分助手选择卡片方案.md）。
 *
 * 一种卡片承担所有选择：用途（换评分标准、说的是哪个任务、哪一份材料）只决定后端点了之后做什么，界面一样。
 * 选项的文字来自后端的工具结果；点了通过 `select` 交给后端，写动作由后端经共享守卫执行。
 */
import { computed } from "vue";

import { useAssistantStore } from "@/stores/assistant.js";

const props = defineProps({
  card: { type: Object, required: true },
});

const store = useAssistantStore();
const proposed = computed(() => props.card.status === "proposed");
const disabled = computed(() => store.sending || store.uploading);
const options = computed(() => props.card.options || []);

const STATUS_TEXT = { expired: "已失效", dismissed: "已放弃" };
const resultText = computed(() => {
  const picked = options.value.find((item) => item.id === (props.card.result || {}).id);
  if (picked) return `已选：${picked.label}`;
  return STATUS_TEXT[props.card.status] || "已完成";
});
</script>

<template>
  <div class="choice-card">
    <div v-if="proposed" class="choices" role="group" :aria-label="card.title || '请选择'">
      <button v-for="item in options" :key="item.id" class="choice" :class="{ 'is-current': item.current }" type="button"
              :disabled="disabled" @click="store.select(card.id, { id: item.id })">
        <span>{{ item.label }}</span>
        <span v-if="item.current" class="current">正在用</span>
      </button>
    </div>
    <p v-else class="faint small">{{ resultText }}</p>
  </div>
</template>

<style scoped>
.choices { display: flex; flex-wrap: wrap; gap: 6px; }
.choice { display: inline-flex; align-items: center; gap: 6px; max-width: 100%; padding: 7px 11px;
  border: 1px solid var(--border-input); border-radius: 8px; background: var(--surface); cursor: pointer;
  text-align: left; font: inherit; font-size: 13px; color: var(--text); overflow-wrap: anywhere; }
.choice:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.choice:disabled { opacity: 0.55; cursor: default; }
.choice.is-current { border-color: var(--accent); }
.current { flex: none; padding: 0 6px; border-radius: 8px; background: var(--accent-soft, var(--surface-muted)); color: var(--accent);
  font-size: 11.5px; line-height: 18px; }
.small { margin: 0; font-size: 12px; }
</style>
