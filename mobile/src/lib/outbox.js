// Turns upload-queue rows (services/db.js `outbox`) into what the Uploads panel shows.
// Pure JS (no React Native imports) so it is unit-tested with Node.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** "2026-11-11T14:32:12+08:00" -> "11 Nov 2026, 14:32:12" (the time as recorded, PH time). */
export function phTimeLabel(iso) {
  if (!iso) return '';
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2}:\d{2})/.exec(iso);
  if (!m) return String(iso);
  return `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]} ${m[1]}, ${m[4]}`;
}

export function timeAgo(ms, now = Date.now()) {
  if (ms == null) return '';
  const s = Math.max(0, (now - ms) / 1000);
  if (s < 60) return 'just now';
  const min = Math.floor(s / 60);
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.floor(h / 24);
  return d === 1 ? '1 day ago' : `${d} days ago`;
}

const ERROR_TEXT = {
  network: 'No connection to the server',
  unknown_device: 'Server does not know this phone',
  bad_signature: 'Server did not accept this phone’s signature',
  device_disabled: 'This phone was turned off by an admin',
  bad_record: 'Record could not be read by the server',
  bad_seq: 'Record number is invalid',
  bad_detected_at: 'Capture time is invalid',
  missing_fields: 'Record is missing fields',
  bad_json: 'Record could not be read by the server',
  duty_rejected: 'Duty session was refused',
  storage_failed: 'Server could not save the photos',
  case_insert_failed: 'Server could not save the case',
  evidence_insert_failed: 'Server could not save the photo details',
};

/** Stored last_error ("code" or "code: detail", or a fetch message) -> { code, text }. */
export function errorLabel(lastError) {
  if (!lastError) return null;
  const raw = String(lastError);
  if (/network request failed|failed to fetch|timeout/i.test(raw)) return { code: 'network', text: ERROR_TEXT.network };
  const code = raw.split(': ')[0].trim();
  if (code.startsWith('image_too_large:')) return { code, text: `Photo too large (${code.split(':')[1]})` };
  return { code, text: ERROR_TEXT[code] ?? raw };
}

function parse(text) {
  try { return JSON.parse(text) ?? {}; } catch { return {}; }
}

/** One outbox row -> the fields a list row shows. */
export function describeRow(row) {
  const record = parse(row.body);
  const files = row.state === 'sent' ? {} : parse(row.files);
  const photoPath = files.rider ?? files.frame ?? null;
  const plate = record.plate?.status === 'read' && record.plate.text ? record.plate.text : null;
  return {
    key: String(row.id),
    violationId: record.violation_id ?? `Record ${row.seq}`,
    capturedAt: phTimeLabel(record.detected_at),
    plate,
    attempts: row.attempts ?? 0,
    error: errorLabel(row.last_error),
    caseNo: row.case_no ?? null,
    sentAt: row.sent_at ?? null,
    photo: typeof photoPath === 'string' ? `file://${photoPath}` : null,
  };
}
