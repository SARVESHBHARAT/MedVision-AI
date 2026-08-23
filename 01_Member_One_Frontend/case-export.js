// CSV export helper for the case-history table; intended to be connected to the Export button.
export function buildCaseCsv(cases) {
  const quote = value => `"${String(value).replaceAll('"', '""')}"`;
  const heading = ['Case ID', 'Submitted', 'Status', 'Finding'];
  return [heading, ...cases.map(({ id, date, status, finding }) => [id, date, status, finding])]
    .map(row => row.map(quote).join(','))
    .join('\n');
}

export function downloadCaseCsv(cases, filename = 'medvision-case-history.csv') {
  const blob = new Blob([buildCaseCsv(cases)], { type: 'text/csv;charset=utf-8' });
  const link = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: filename });
  link.click();
  URL.revokeObjectURL(link.href);
}
