// 원고(마크다운) → 공모전 제출용 docx 변환기
// 사용: node md2docx.js <manuscript.md> <figs_dir> <out.docx>
// 제출본은 "# [부록" 또는 "# [편집 메모" 이전까지만 변환한다.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, Footer, PageNumber, LevelFormat, PageBreak,
} = require('docx');

const [, , mdPath, figDir, outPath] = process.argv;
const FONT = 'Malgun Gothic';
const FONT_EA = '맑은 고딕';
const BODY = 21;        // 10.5pt
const SMALL = 17;       // 8.5pt (표)
const CAPTION = 19;     // 9.5pt
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1134;   // A4, 20mm
const CONTENT_W = PAGE_W - 2 * MARGIN;                // 9638 DXA
const ACCENT = '1B6AA5';

let md = fs.readFileSync(mdPath, 'utf8').replace(/\r\n/g, '\n');
const cut = md.search(/^# \[(부록|편집 메모)/m);
if (cut > 0) md = md.slice(0, cut);
const lines = md.split('\n');

function runs(text, base = {}) {
  // **굵게** 처리. 그 밖의 마크다운 기호(`, *)는 제거.
  const out = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const p of parts) {
    if (!p) continue;
    const bold = p.startsWith('**') && p.endsWith('**');
    // *기울임*: 굵게가 아닌 조각 안에서만 처리(참고문헌 학술지명)
    const sub = bold ? [p.slice(2, -2)] : p.split(/(\*[^*\s][^*]*\*)/g);
    for (const q of sub) {
      if (!q) continue;
      const ital = !bold && q.length > 2 && q.startsWith('*') && q.endsWith('*');
      const t = (ital ? q.slice(1, -1) : q).replace(/`/g, '');
      out.push(new TextRun({ text: t, bold: bold || base.bold, italics: ital || undefined, size: base.size || BODY, color: base.color, font: { ascii: FONT, eastAsia: FONT_EA, hAnsi: FONT } }));
    }
  }
  return out.length ? out : [new TextRun({ text: '', size: base.size || BODY })];
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

const children = [];
let numInstance = 0, inNumbered = false, firstRule = true, titleDone = false, afterTitle = 0, inRefs = false;

function para(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts.run || {}),
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 80, line: opts.line || 276 },
    keepNext: opts.keepNext,
    numbering: opts.numbering,
    heading: opts.heading,
    indent: opts.indent,
  });
}

function parseRow(l) {
  return l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}

function makeTable(rowsTxt) {
  const header = parseRow(rowsTxt[0]);
  const body = rowsTxt.slice(2).map(parseRow);
  const ncol = header.length;
  // 열 너비: 글자 수 기반(최소·최대 제한)
  const len = Array(ncol).fill(0);
  for (const r of [header, ...body]) r.forEach((c, i) => { if (i < ncol) len[i] = Math.max(len[i], Math.min(40, c.replace(/\*\*/g, '').length)); });
  // 머리글 글자 수보다 좁아지지 않게(한글 1자 ≈ 1.6 단위), 최소 4자
  const weights = len.map((x, i) => Math.max(4, x, header[i].replace(/\*\*/g, '').length * 1.6 + 1));
  const sum = weights.reduce((a, b) => a + b, 0);
  let widths = weights.map((w) => Math.max(620, Math.floor((w / sum) * CONTENT_W)));
  const over = widths.reduce((a, b) => a + b, 0) - CONTENT_W;
  if (over > 0) { const big = widths.indexOf(Math.max(...widths)); widths[big] -= over; }
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: 'A6A6A6' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkCell = (txt, i, isHead) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    shading: isHead ? { type: ShadingType.CLEAR, fill: 'E8EEF4', color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(txt || '', { size: SMALL, bold: isHead }), alignment: AlignmentType.LEFT, spacing: { after: 0, line: 240 } })],
  });
  const rows = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((c, i) => mkCell(c, i, true)) })];
  for (const r of body) {
    const cells = [];
    for (let i = 0; i < ncol; i++) cells.push(mkCell(r[i] || '', i, false));
    rows.push(new TableRow({ cantSplit: true, children: cells }));
  }
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows });
}

for (let i = 0; i < lines.length; i++) {
  const raw = lines[i];
  const l = raw.trimEnd();
  if (!l.trim()) { inNumbered = false; continue; }

  // 표
  if (l.trim().startsWith('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
    const block = [];
    while (i < lines.length && lines[i].trim().startsWith('|')) { block.push(lines[i]); i++; }
    i--;
    children.push(makeTable(block));
    children.push(new Paragraph({ children: [], spacing: { after: 60 } }));
    continue;
  }
  // 구분선
  if (/^-{3,}$/.test(l.trim())) {
    if (firstRule) { children.push(new Paragraph({ children: [new PageBreak()] })); firstRule = false; }
    continue;
  }
  // 그림 자리표시
  const fm = l.match(/^\[그림\s*(\d+)\s*[:：]\s*([^\]\s]+\.png)\s*\]?(.*)$/);
  if (fm) {
    const file = path.join(figDir, fm[2]);
    if (fs.existsSync(file)) {
      const { w, h, data } = pngSize(file);
      let ww, hh;
      if (process.env.FIG_NATIVE === '1') {
        // 그림을 그린 물리 크기 그대로 배치(300dpi 가정 → 96px/in). 본문 폭(약 642px)을 넘으면 비율 유지 축소.
        ww = Math.round((w / 300) * 96); hh = Math.round((h / 300) * 96);
        const maxW = 642;
        if (ww > maxW) { hh = Math.round(hh * maxW / ww); ww = maxW; }
      } else {
        const maxW = 620, maxH = Number(process.env.FIG_MAXH || 430);   // v2: FIG_MAXH로 그림 최대 높이(px) 조정
        ww = maxW; hh = Math.round((h / w) * maxW);
        if (hh > maxH) { hh = maxH; ww = Math.round((w / h) * maxH); }
      }
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 60, after: 40 }, children: [new ImageRun({ type: 'png', data, transformation: { width: ww, height: hh } })] }));
    } else {
      children.push(para(`[그림 파일 없음: ${fm[2]}]`, { run: { color: 'C00000' } }));
    }
    if (fm[3] && fm[3].trim()) children.push(para(fm[3].trim(), { run: { size: CAPTION }, align: AlignmentType.CENTER }));
    continue;
  }
  // 제목
  const h = l.match(/^(#{1,3})\s+(.*)$/);
  if (h) {
    const level = h[1].length, text = h[2].trim();
    if (!titleDone && level === 1) {
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 120 }, children: [new TextRun({ text, bold: true, size: 34, color: ACCENT, font: { ascii: FONT, eastAsia: FONT_EA, hAnsi: FONT } })] }));
      titleDone = true; afterTitle = 2; continue;
    }
    inRefs = /참고문헌/.test(text);
    const hl = level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3;
    children.push(new Paragraph({ heading: hl, keepNext: true, children: runs(text, { bold: true, size: level === 1 ? 28 : level === 2 ? 24 : 22, color: level === 1 ? ACCENT : '1F1F1F' }), spacing: { before: level === 1 ? 240 : 160, after: 80 } }));
    continue;
  }
  // 제목 바로 아래 부제·팀명 줄은 가운데 정렬
  if (afterTitle > 0) { children.push(para(l.trim(), { align: AlignmentType.CENTER, after: 60 })); afterTitle--; continue; }
  // 표·그림 캡션
  if (/^\*\*(표|그림|박스)\s*\d+/.test(l.trim()) || /^(그림|표)\s*\d+\./.test(l.trim())) {
    const isTable = /^\*\*표/.test(l.trim()) || /^표\s*\d+/.test(l.trim());
    children.push(para(l.trim(), { run: { size: CAPTION, bold: true }, align: isTable ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: isTable, after: 60 }));
    continue;
  }
  // 주석(주: …, ¹ …)
  if (/^(주[:：]|¹|²|³|출처[:：])/.test(l.trim())) { children.push(para(l.trim(), { run: { size: SMALL, color: '595959' }, after: 80 })); continue; }
  // 글머리표
  const b = raw.match(/^(\s*)[-*]\s+(.*)$/);
  if (b) {
    const lvl = Math.min(2, Math.floor(b[1].length / 2));
    // 참고문헌 목록은 줄 간격을 촘촘하게(지면)
    children.push(para(b[2], inRefs ? { numbering: { reference: 'bullets', level: lvl }, after: 20, line: 240, run: { size: SMALL } } : { numbering: { reference: 'bullets', level: lvl }, after: 40 }));
    continue;
  }
  // 번호 목록
  const n = raw.match(/^(\s*)(\d+)\.\s+(.*)$/);
  if (n) {
    if (!inNumbered) { numInstance++; inNumbered = true; }
    const lvl = Math.min(2, Math.floor(n[1].length / 3));
    children.push(para(n[3], { numbering: { reference: 'numbers', level: lvl, instance: numInstance }, after: 40 }));
    continue;
  }
  inNumbered = false;
  children.push(para(l.trim()));
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: FONT, eastAsia: FONT_EA, hAnsi: FONT }, size: BODY } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 28, bold: true, color: ACCENT, font: FONT }, paragraph: { outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, font: FONT }, paragraph: { outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 22, bold: true, font: FONT }, paragraph: { outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [0, 1, 2].map((lv) => ({ level: lv, format: LevelFormat.BULLET, text: lv === 0 ? '•' : lv === 1 ? '–' : '·', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360 + lv * 360, hanging: 240 } } } })) },
      { reference: 'numbers', levels: [0, 1, 2].map((lv) => ({ level: lv, format: lv === 1 ? LevelFormat.LOWER_LETTER : LevelFormat.DECIMAL, text: lv === 1 ? '%2)' : `%${lv + 1}.`, alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400 + lv * 400, hanging: 300 } } } })) },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ['- ', PageNumber.CURRENT, ' -'], size: 18, font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log('written', outPath, children.length, 'blocks'); });
