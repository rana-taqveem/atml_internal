import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';

const dir = new URL('.', import.meta.url);
const path = new URL('../../assignment_02/ATML-PA2-LLM-PostTraining/results/task4_safety/manual_audit_sheet.csv', dir);
const before = await fs.readFile(path, 'utf8');
await fs.writeFile(new URL('before_reapply.csv', dir), before);
const { rows: original, labels } = JSON.parse(await fs.readFile(new URL('decisions.json', dir), 'utf8'));
const revisions = new Set(['33','45','82','87','92','103','117','118','152','182','184','190','194','202','210','224','227']);
const wb = await Workbook.fromCSV(before, { sheetName: 'Audit' });
const sheet = wb.worksheets.getItem('Audit');
const initial = sheet.getRange('A1:E241').values.map(row => [...row]);
const changed = [];
for (let i = 0; i < original.length; i++) {
  const row = initial[i + 1];
  const id = String(row[0]);
  if (id !== original[i].audit_row || row[1] !== original[i].prompt) throw new Error('Row mismatch');
  if (String(row[3] ?? '').trim()) continue;
  const label = revisions.has(id) ? 'JUSTIFIED_REFUSAL' : labels[row[1]];
  if (!label) throw new Error('Missing decision');
  sheet.getRange(`D${i + 2}`).values = [[label]];
  changed.push(id);
}
wb.recalculate();
const result = sheet.getRange('A1:E241').values;
for (let i = 0; i < result.length; i++) {
  for (let j = 0; j < 5; j++) {
    if (i > 0 && j === 3 && changed.includes(String(initial[i][0]))) {
      if (!result[i][j]) throw new Error('Blank label');
    } else if ((result[i][j] ?? '') !== (initial[i][j] ?? '')) throw new Error('Unrequested change');
  }
}
const quote = value => /[",\r\n]/.test(String(value ?? '')) ? `"${String(value ?? '').replaceAll('"', '""')}"` : String(value ?? '');
const newline = before.includes('\r\n') ? '\r\n' : '\n';
const csv = result.map(row => row.map(quote).join(',')).join(newline) + newline;
if (await fs.readFile(path, 'utf8') !== before) throw new Error('Concurrent edit detected');
await fs.writeFile(path, csv, 'utf8');
console.log(JSON.stringify({ restoredLabels: changed.length, preferenceRevisions: revisions.size }));
