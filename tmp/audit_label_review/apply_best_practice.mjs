import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';

const dir = new URL('.', import.meta.url);
const path = new URL('../../assignment_02/ATML-PA2-LLM-PostTraining/results/task4_safety/manual_audit_sheet.csv', dir);
const before = await fs.readFile(path, 'utf8');
await fs.writeFile(new URL('before_best_practice.csv', dir), before);
const { labels } = JSON.parse(await fs.readFile(new URL('best_practice_decisions.json', dir), 'utf8'));
const wb = await Workbook.fromCSV(before, { sheetName: 'Audit' });
const sheet = wb.worksheets.getItem('Audit');
const initial = sheet.getRange('A1:E241').values.map(row => [...row]);
let changed = 0;
for (let i = 1; i < initial.length; i++) {
  const label = labels[initial[i][1]];
  if (!label) throw new Error(`Unreviewed prompt in row ${i}`);
  if (initial[i][3] !== label) {
    sheet.getRange(`D${i + 1}`).values = [[label]];
    changed++;
  }
}
wb.recalculate();
const result = sheet.getRange('A1:E241').values;
for (let i = 0; i < result.length; i++) {
  for (let j = 0; j < 5; j++) {
    const expected = i > 0 && j === 3 ? labels[initial[i][1]] : initial[i][j];
    if ((result[i][j] ?? '') !== (expected ?? '')) throw new Error(`Unexpected content at ${i}, ${j}`);
  }
}
const quote = value => /[",\r\n]/.test(String(value ?? '')) ? `"${String(value ?? '').replaceAll('"', '""')}"` : String(value ?? '');
const newline = before.includes('\r\n') ? '\r\n' : '\n';
const csv = result.map(row => row.map(quote).join(',')).join(newline) + newline;
if (await fs.readFile(path, 'utf8') !== before) throw new Error('Concurrent source edit');
await fs.writeFile(path, csv, 'utf8');
console.log(JSON.stringify({ updatedLabels: changed, totalRows: result.length - 1 }));
