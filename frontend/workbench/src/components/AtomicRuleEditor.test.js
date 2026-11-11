import { mount } from '@vue/test-utils';
import { expect, it } from 'vitest';
import AtomicRuleEditor from './AtomicRuleEditor.vue';

it('单次旧上限可显式清除，累计封顶输入仍可编辑；清除不改扣分或命中方式', async () => {
  const rule = { id: 'r', rule_code: 'T01.r1', changes: {
    name:'合成规则', rule_text:'合成条件', direction:'deduct', effect_type:'score',
    repeat_policy:'once', max_points:2, cap_points:6, levels:[],
  } };
  const wrapper = mount(AtomicRuleEditor, { props: { rules: [rule] } });
  const cap = wrapper.findAll('label').find(l => l.text().includes('单条累计上限')).find('input');
  expect(cap.element.disabled).toBe(true);
  const clear = wrapper.findAll('button').find(b => b.text() === '清除不适用的单条上限');
  await clear.trigger('click');
  expect(rule.changes.cap_points).toBeNull();
  expect(rule.changes.max_points).toBe(2);
  expect(rule.changes.repeat_policy).toBe('once');
  const repeat = wrapper.findAll('label').find(l => l.text().includes('重复命中方式')).find('select');
  await repeat.setValue('capped');
  expect(cap.element.disabled).toBe(false);
  await cap.setValue(6);
  expect(rule.changes.cap_points).toBe(6);
});

it('语义规则只读显示适用范围和来源，没有可编辑的范围输入', () => {
  const rule = { id: 'r', rule_code: 'T01.r1',
    scope: { type: 'cross_section', sections: ['绪论', '结论'], source: 'ai_draft' }, scope_label: '跨章节：绪论、结论',
    changes: { name:'研究问题与结论呼应', rule_text:'绪论的研究问题在结论中没有回应', direction:'deduct', effect_type:'score',
      judge_type:'semantic', repeat_policy:'once', max_points:2, levels:[],
      evidence_policy: { mode:'source_quote', requirement:'required', minimum_coverage:'1' } } };
  const wrapper = mount(AtomicRuleEditor, { props: { rules: [rule] } });
  const field = wrapper.find('[data-test="rule-scope"]');
  expect(field.text()).toContain('跨章节：绪论、结论');
  expect(field.text()).toContain('起草时 AI 识别');
  expect(field.find('input').exists()).toBe(false);
  expect(field.find('select').exists()).toBe(false);
  expect(wrapper.findAll('label').some(l => l.text().startsWith('判定范围'))).toBe(false);
});

it('确定性规则不显示适用范围；没有范围的语义规则按全文显示', () => {
  const deterministic = { id: 'd', rule_code: 'T01.d1', changes: { name:'必备章节', rule_text:'缺少结论章', direction:'deduct',
    effect_type:'score', judge_type:'deterministic', levels:[] } };
  const semantic = { id: 's', rule_code: 'T01.s1', scope: null, scope_label: '全文',
    changes: { name:'论证充分', rule_text:'论证不充分', direction:'deduct', effect_type:'score', judge_type:'semantic', levels:[] } };
  const wrapper = mount(AtomicRuleEditor, { props: { rules: [deterministic, semantic] } });
  const scopes = wrapper.findAll('[data-test="rule-scope"]');
  expect(scopes).toHaveLength(1);
  expect(scopes[0].text()).toContain('全文');
  expect(scopes[0].text()).toContain('尚未识别，按全文');
});
