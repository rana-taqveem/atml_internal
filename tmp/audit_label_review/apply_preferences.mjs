import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';

const taskDir = new URL('.', import.meta.url);
const sourcePath = new URL('../../assignment_02/ATML-PA2-LLM-PostTraining/results/task4_safety/manual_audit_sheet.csv', taskDir);
const before = await fs.readFile(sourcePath, 'utf8');
await fs.writeFile(new URL('before_preferences.csv', taskDir), before);
const { rows: initial } = JSON.parse(await fs.readFile(new URL('decisions.json', taskDir), 'utf8'));
const preferred = new Set(initial.filter(row => row.manual_label.trim() === 'JUSTIFIED_REFUSAL').map(row => row.prompt));
const workbook = await Workbook.fromCSV(before, { sheetName: 'Audit' });
const sheet = workbook.worksheets.getItem('Audit');
const originalValues = sheet.getRange('A1:E241').values.map(row => [...row]);
const changed = [];
for (let i = 0; i < initial.length; i++) {
  const row = originalValues[i + 1];
  if (String(row[0]) !== initial[i].audit_row || row[1] !== initial[i].prompt) throw new Error('Row mismatch');
  if (!initial[i].manual_label && row[3] === 'OVER_REFUSAL' && preferred.has(row[1])) {
    sheet.getRange(`D${i + 2}`).values = [['JUSTIFIED_REFUSAL']];
    changed.push(String(row[0]));
  }
}
workbook.recalculate();
const result = sheet.getRange('A1:E241').values;
for (let i = 0; i < result.length; i++) {
  for (let j = 0; j < 5; j++) {
    const expected = i > 0 && j === 3 && changed.includes(String(originalValues[i][0])) ? 'JUSTIFIED_REFUSAL' : originalValues[i][j];
    if ((result[i][j] ?? '') !== (expected ?? '')) throw new Error('Unexpected change');
  }
}
const quote = value => /[",\r\n]/.test(String(value ?? '')) ? `"${String(value ?? '').replaceAll('"', '""')}"` : String(value ?? '');
const newline = before.includes('\r\n') ? '\r\n' : '\n';
const csv = result.map(row => row.map(quote).join(',')).join(newline) + newline;
if (await fs.readFile(sourcePath, 'utf8') !== before) throw new Error('Concurrent source change');
await fs.writeFile(sourcePath, csv, 'utf8');
console.log(JSON.stringify({ changedCount: changed.length, auditRows: changed }));
