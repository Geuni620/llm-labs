import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

interface Snippet { file: string; anchor: string; lines: number; label: string }
interface Section {
  title: string;
  kind?: string;
  nodes: [string, string][];
  paragraphs: string[];
  bullets?: string[];
  formula?: string;
  snippets: Snippet[];
}
const root = fileURLToPath(new URL('../', import.meta.url));
const definitions = JSON.parse(readFileSync(join(root, 'docs/implementation-sections.json'), 'utf8')) as Section[];
const prepared = definitions.map(section => ({
  ...section,
  snippets: section.snippets.map(snippet => {
    const source = readFileSync(join(root, snippet.file), 'utf8').split('\n');
    const start = source.findIndex(line => line.includes(snippet.anchor));
    assert.ok(start >= 0, `${snippet.file}: 코드 구간을 찾을 수 없습니다: ${snippet.anchor}`);
    const end = Math.min(start + snippet.lines, source.length);
    return { ...snippet, start: start + 1, end, code: source.slice(start, end).join('\n') };
  }),
}));
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const guideData = { sections: prepared, runtimeDependencies: Object.keys(pkg.dependencies ?? {}), devDependencies: pkg.devDependencies };
const template = readFileSync(join(root, 'docs/implementation-guide.template.html'), 'utf8');
const svg = readFileSync(join(root, 'src/svg.js'), 'utf8');
const player = readFileSync(join(root, 'docs/implementation-guide.js'), 'utf8');
const fragment = template.replace('__SVG_SOURCE__', () => svg).replace('__GUIDE_PLAYER__', () => player)
  .replace('__GUIDE_DATA__', () => JSON.stringify(guideData).replaceAll('<', '\\u003c'));
assert.ok(Buffer.byteLength(fragment) < 512000);
const full = (style = '') => `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Node 언어 모델 구현 해설</title>${style}</head><body>${fragment}</body></html>`;
const theme = `<style>:root{color-scheme:light dark;--foreground:light-dark(#18181b,#f4f4f5);--background:light-dark(#fff,#18181b);--muted:light-dark(#f4f4f5,#27272a);--muted-foreground:light-dark(#52525b,#b4b4be);--border:light-dark(#d4d4d8,#484850);--code-background:var(--muted);--code-foreground:var(--foreground);--font-sans:system-ui,sans-serif;--font-mono:ui-monospace,monospace;--font-size-base:14px}body{max-width:940px;margin:24px auto;padding:0 16px;background:var(--background);color:var(--foreground);font:14px var(--font-sans)}button,select{font:inherit;padding:7px 10px}button{background:var(--background);color:var(--foreground);border:1px solid var(--border);border-radius:6px}</style>`;
mkdirSync(join(root, 'results'), { recursive: true });
writeFileSync(join(root, 'results/implementation-guide.inline.html'), full());
writeFileSync(join(root, 'docs/implementation-guide.html'), full(theme));
const ascii = (section: Section) => section.kind === 'branches'
  ? '                 ┌→ 트라이그램 ─┐\n6문장 → 24학습쌍 ─┤              ├→ 생성 → 분류·리포트\n                 └→ NPLM ──────┘'
  : section.nodes.map(node => node[0]).join(' → ');
const markdown = '# Node·TypeScript 언어 모델 전체 구현 해설\n\n' +
  '실행·학습·시각화는 외부 라이브러리 없이 구현했습니다. TypeScript와 Node 타입 정의는 개발 도구로 유지합니다. 코드 블록은 실제 파일에서 추출한 발췌입니다. 전체 코드는 각 파일 링크에서 확인할 수 있습니다.\n\n' +
  prepared.map((section, index) => `## ${index + 1}. ${section.title}\n\n\`\`\`text\n${ascii(section)}\n\`\`\`\n\n` +
    section.paragraphs.join('\n\n') + '\n\n' +
    (section.bullets?.map(text => `- ${text}`).join('\n') ?? '') + '\n\n' +
    (section.formula ? `\`\`\`text\n${section.formula}\n\`\`\`\n\n` : '') +
    section.snippets.map(snippet => `**${snippet.label}** — [${snippet.file}](${snippet.file}), ${snippet.start}~${snippet.end}행\n\n\`\`\`${snippet.file.endsWith('.html') ? 'html' : snippet.file.endsWith('.json') ? 'json' : snippet.file.endsWith('.js') ? 'javascript' : 'typescript'}\n${snippet.code}\n\`\`\`\n`).join('\n'))
    .join('\n');
writeFileSync(join(root, 'IMPLEMENTATION.md'), markdown);
console.log(`${prepared.length}개 해설 · 실제 코드 ${prepared.reduce((sum, section) => sum + section.snippets.length, 0)}개 발췌 · 실행 의존성 ${guideData.runtimeDependencies.length}개`);
console.log(join(root, 'docs/implementation-guide.html'));
