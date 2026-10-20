import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';

const taskDir = new URL('.', import.meta.url);
const sourcePath = new URL('../../assignment_02/ATML-PA2-LLM-PostTraining/results/task4_safety/manual_audit_sheet.csv', taskDir);
const original = await fs.readFile(new URL('original.csv', taskDir), 'utf8');
const { rows, labels } = JSON.parse(await fs.readFile(new URL('decisions.json', taskDir), 'utf8'));
const workbook = await Workbook.fromCSV(original, { sheetName: 'Audit' });
const sheet = workbook.worksheets.getItem('Audit');
let changed = 0;
for (let i = 0; i < rows.length; i++) {
  if (rows[i].manual_label) continue;
  sheet.getRange(`D${i + 2}`).values = [[labels[rows[i].prompt]]];
  changed++;
}
workbook.recalculate();
const result = sheet.getRange('A1:E241').values;
if (result.length !== 241) throw new Error('Unexpected row count');
for (let i = 0; i < rows.length; i++) {
  const expected = [rows[i].audit_row, rows[i].prompt, rows[i].response, rows[i].manual_label || labels[rows[i].prompt], rows[i].notes];
  for (let j = 0; j < 5; j++) {
    if (String(result[i + 1][j] ?? '') !== expected[j]) throw new Error(`Preservation failure at row ${i + 1}, column ${j}`);
  }
}
// Serialize the verified values to the existing CSV format without adding workbook features.
const quote = value => /[",\r\n]/.test(String(value ?? '')) ? `"${String(value ?? '').replaceAll('"', '""')}"` : String(value ?? '');
const newline = original.includes('\r\n') ? '\r\n' : '\n';
const csv = result.map(row => row.map(quote).join(',')).join(newline) + newline;
if (await fs.readFile(sourcePath, 'utf8') !== original) throw new Error('Source changed during review');
await fs.writeFile(sourcePath, csv, 'utf8');
console.log(JSON.stringify({ changed, preserved: rows.length - changed, total: rows.length }));
