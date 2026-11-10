import test from 'node:test';
import assert from 'node:assert/strict';
import { describeRow, errorLabel, phTimeLabel, timeAgo } from '../src/lib/outbox.js';

const record = {
  violation_id: 'V-20261111-143212-0007',
  detected_at: '2026-11-11T14:32:12+08:00',
  plate: { status: 'read', text: 'ABC1234', ocr_confidence: 0.93 },
};
const row = (over = {}) => ({
  id: 5, seq: 7, body: JSON.stringify(record),
  files: JSON.stringify({ frame: '/data/ev/1/frame.jpg', rider: '/data/ev/1/rider.jpg' }),
  attempts: 0, last_error: null, created_at: 0, sent_at: null, case_no: null, state: 'pending',
  ...over,
});

test('phTimeLabel shows the capture time as written (Philippine time)', () => {
  assert.equal(phTimeLabel('2026-11-11T14:32:12+08:00'), '11 Nov 2026, 14:32:12');
  assert.equal(phTimeLabel('2026-01-02T03:04:05+08:00'), '2 Jan 2026, 03:04:05');
  assert.equal(phTimeLabel('not a time'), 'not a time');
  assert.equal(phTimeLabel(null), '');
});

test('timeAgo', () => {
  const now = 10_000_000;
  assert.equal(timeAgo(now - 20_000, now), 'just now');
  assert.equal(timeAgo(now - 60_000, now), '1 min ago');
  assert.equal(timeAgo(now - 59 * 60_000, now), '59 min ago');
  assert.equal(timeAgo(now - 3 * 3600_000, now), '3 h ago');
  assert.equal(timeAgo(now - 2 * 86400_000, now), '2 days ago');
  assert.equal(timeAgo(now - 86400_000, now), '1 day ago');
  assert.equal(timeAgo(now + 5000, now), 'just now'); // clock skew
  assert.equal(timeAgo(null, now), '');
});

test('errorLabel turns stored errors into words', () => {
  assert.deepEqual(errorLabel(null), null);
  assert.deepEqual(errorLabel('network'), { code: 'network', text: 'No connection to the server' });
  assert.deepEqual(errorLabel('Network request failed'), { code: 'network', text: 'No connection to the server' });
  assert.deepEqual(errorLabel('bad_signature'), { code: 'bad_signature', text: 'Server did not accept this phone’s signature' });
  assert.deepEqual(errorLabel('storage_failed: bucket missing'), { code: 'storage_failed', text: 'Server could not save the photos' });
  assert.deepEqual(errorLabel('image_too_large:frame'), { code: 'image_too_large:frame', text: 'Photo too large (frame)' });
  assert.deepEqual(errorLabel('weird_thing: x'), { code: 'weird_thing', text: 'weird_thing: x' });
});

test('describeRow: waiting record', () => {
  const d = describeRow(row({ attempts: 2, last_error: 'Network request failed' }));
  assert.equal(d.key, '5');
  assert.equal(d.violationId, 'V-20261111-143212-0007');
  assert.equal(d.capturedAt, '11 Nov 2026, 14:32:12');
  assert.equal(d.plate, 'ABC1234');
  assert.equal(d.attempts, 2);
  assert.equal(d.error.code, 'network');
  assert.equal(d.photo, 'file:///data/ev/1/rider.jpg');
  assert.equal(d.caseNo, null);
});

test('describeRow: uploaded record has a case number and no photo', () => {
  const d = describeRow(row({ state: 'sent', case_no: 'RS-AB12-000007', sent_at: 123 }));
  assert.equal(d.caseNo, 'RS-AB12-000007');
  assert.equal(d.sentAt, 123);
  assert.equal(d.photo, null); // photos are deleted from the phone after upload
});

test('describeRow: plate not read', () => {
  for (const plate of [{ status: 'unreadable', text: null }, { status: 'not_visible' }, undefined]) {
    const d = describeRow(row({ body: JSON.stringify({ ...record, plate }) }));
    assert.equal(d.plate, null);
  }
});

test('describeRow: falls back to frame photo, and survives broken text', () => {
  assert.equal(describeRow(row({ files: JSON.stringify({ frame: '/x/frame.jpg' }) })).photo, 'file:///x/frame.jpg');
  const d = describeRow(row({ body: '{broken', files: 'also broken' }));
  assert.equal(d.violationId, 'Record 7');
  assert.equal(d.capturedAt, '');
  assert.equal(d.photo, null);
});
