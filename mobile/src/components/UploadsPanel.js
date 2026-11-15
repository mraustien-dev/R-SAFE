// Uploads: read-only view of the upload queue. It opens ON TOP of the duty screen (not as a
// separate route) so the camera keeps detecting underneath. There is deliberately no button
// to upload, retry, edit or delete: uploading is automatic and evidence cannot be changed.
import { useCallback, useEffect, useState } from 'react';
import { BackHandler, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { listOutbox } from '../services/db';
import { onSyncStatus } from '../services/sync';
import { describeRow, timeAgo } from '../lib/outbox';
import { Screen, colors, styles as ui } from '../ui';

const PAGE = 50;
const TABS = [
  { state: 'pending', label: 'Waiting', count: 'pending' },
  { state: 'sent', label: 'Uploaded', count: 'sent' },
  { state: 'rejected', label: 'Refused', count: 'rejected' },
];

function useTick(ms) {
  const [, set] = useState(0);
  useEffect(() => {
    const t = setInterval(() => set((n) => n + 1), ms);
    return () => clearInterval(t);
  }, [ms]);
}

function Banner({ sync }) {
  let tone = colors.good;
  let title = 'Online';
  let sub = sync.lastSyncAt ? `Last synced ${timeAgo(sync.lastSyncAt)}` : 'Not synced yet';
  if (sync.blocked === 'unknown_device') {
    // The server has no camera row for this phone's key. Linking again keeps the same key,
    // so records already signed on this phone still pass the server's check.
    return (
      <View style={[ui.card, { borderLeftWidth: 4, borderLeftColor: colors.danger, marginBottom: 12 }]}>
        <Text style={{ fontSize: 17, fontWeight: '700', color: colors.danger }}>Phone not linked on the server</Text>
        <Text style={{ color: colors.muted, marginTop: 2 }}>
          Ask an admin for a new registration code, then link this phone again. Records stay safe on this phone and upload after linking.
        </Text>
        <Pressable
          onPress={() => router.push('/register')}
          style={({ pressed }) => [ui.button, { marginTop: 10 }, pressed && { opacity: 0.7 }]}
          accessibilityRole="button"
        >
          <Text style={ui.buttonText}>Link this phone again</Text>
        </Pressable>
      </View>
    );
  }
  if (sync.blocked) {
    tone = colors.danger;
    title = 'Phone blocked by server';
    sub = `Reason: ${sync.blocked}. Tell an admin. Records stay safe on this phone.`;
  } else if (sync.syncing) {
    tone = colors.primary;
    title = 'Uploading now…';
  } else if (!sync.online) {
    tone = colors.warn;
    title = 'Offline';
    sub = 'Uploads start by themselves when the phone is back online.';
  }
  return (
    <View style={[ui.card, { borderLeftWidth: 4, borderLeftColor: tone, marginBottom: 12 }]}>
      <Text style={{ fontSize: 17, fontWeight: '700', color: tone }}>{title}</Text>
      <Text style={{ color: colors.muted, marginTop: 2 }}>{sub}</Text>
    </View>
  );
}

function Thumb({ uri }) {
  const [failed, setFailed] = useState(false);
  if (!uri || failed) return null;
  return <Image source={{ uri }} style={s.thumb} onError={() => setFailed(true)} />;
}

function Row({ item, state }) {
  const d = describeRow(item);
  return (
    <View style={s.row}>
      {state !== 'sent' ? <Thumb uri={d.photo} /> : null}
      <View style={{ flex: 1, minWidth: 0 }}>
        <Text style={s.id} numberOfLines={1}>{d.violationId}</Text>
        <Text style={s.meta}>
          {d.capturedAt}{' · '}{d.plate ? `Plate ${d.plate}` : 'Plate not read'}
        </Text>
        {state === 'sent' ? (
          <Text style={[s.meta, { color: colors.good }]}>
            {`Case ${d.caseNo ?? '—'} · uploaded ${timeAgo(d.sentAt)}`}
          </Text>
        ) : null}
        {state === 'pending' ? (
          <Text style={s.meta}>
            {d.attempts === 0 ? 'Not tried yet' : `Tried ${d.attempts} time${d.attempts === 1 ? '' : 's'}`}
            {d.error ? ` · ${d.error.text}` : ''}
          </Text>
        ) : null}
        {state === 'rejected' ? (
          <Text style={[s.meta, { color: colors.danger }]}>
            {`Refused: ${d.error ? `${d.error.text} (${d.error.code})` : 'no reason given'}`}
          </Text>
        ) : null}
      </View>
    </View>
  );
}

export default function UploadsPanel({ onClose }) {
  const [sync, setSync] = useState(null);
  const [tab, setTab] = useState('pending');
  const [rows, setRows] = useState([]);
  const [more, setMore] = useState(false);
  useTick(30000); // keeps "2 min ago" current

  useEffect(() => onSyncStatus(setSync), []);
  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => { onClose(); return true; });
    return () => sub.remove();
  }, [onClose]);

  const load = useCallback(async (count) => {
    const list = await listOutbox(tab, count + 1, 0);
    setMore(list.length > count);
    setRows(list.slice(0, count));
  }, [tab]);

  // reload when the tab changes and whenever the queue changes (a sync pass, a new capture)
  const c = sync?.counts;
  useEffect(() => {
    load(PAGE).catch((e) => console.warn('[R-SAFE] could not read upload queue', e));
  }, [load, c?.pending, c?.sent, c?.rejected, sync?.lastSyncAt]);

  return (
    <View style={StyleSheet.absoluteFill}>
      <Screen style={{ paddingBottom: 0 }}>
        <View style={s.header}>
          <Text style={ui.title}>Uploads</Text>
          <Pressable onPress={onClose} style={({ pressed }) => [s.back, pressed && { opacity: 0.7 }]} accessibilityRole="button">
            <Text style={{ fontSize: 16, fontWeight: '600', color: colors.ink }}>Back</Text>
          </Pressable>
        </View>
        <Text style={[ui.sub, { marginTop: 0, marginBottom: 12 }]}>
          Violations upload automatically. This page only shows their progress.
        </Text>

        {sync ? <Banner sync={sync} /> : null}

        <View style={s.tabs} accessibilityRole="tablist">
          {TABS.map((t) => {
            const on = t.state === tab;
            const n = c?.[t.count] ?? 0;
            const tone = t.state === 'rejected' && n > 0 ? colors.danger : on ? colors.primary : colors.ink;
            return (
              <Pressable
                key={t.state}
                onPress={() => setTab(t.state)}
                accessibilityRole="tab"
                accessibilityState={{ selected: on }}
                style={[s.tab, on && s.tabOn]}
              >
                <Text style={{ fontSize: 22, fontWeight: '700', color: tone }}>{n}</Text>
                <Text style={{ color: on ? colors.primary : colors.muted, fontWeight: on ? '600' : '400' }}>{t.label}</Text>
              </Pressable>
            );
          })}
        </View>

        <FlatList
          data={rows}
          keyExtractor={(r) => String(r.id)}
          renderItem={({ item }) => <Row item={item} state={tab} />}
          contentContainerStyle={{ paddingBottom: 24 }}
          ListEmptyComponent={
            <Text style={s.empty}>
              {tab === 'pending' ? 'Nothing waiting. Every captured violation has left the phone.'
                : tab === 'sent' ? 'No uploads yet.'
                  : 'No refused records.'}
            </Text>
          }
          ListFooterComponent={more ? (
            <Pressable onPress={() => load(rows.length + PAGE)} style={[ui.button, s.more]}>
              <Text style={[ui.buttonText, { color: colors.ink }]}>Load more</Text>
            </Pressable>
          ) : null}
        />
      </Screen>
    </View>
  );
}

const s = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 },
  back: { paddingVertical: 10, paddingHorizontal: 16, borderRadius: 10, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.card },
  tabs: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  tab: { flex: 1, alignItems: 'center', paddingVertical: 10, borderRadius: 10, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.card },
  tabOn: { borderColor: colors.primary, borderWidth: 2 },
  row: { flexDirection: 'row', gap: 12, padding: 12, marginBottom: 8, borderRadius: 10, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.card },
  thumb: { width: 56, height: 56, borderRadius: 6, backgroundColor: colors.line },
  id: { fontSize: 15, fontWeight: '600', color: colors.ink, fontVariant: ['tabular-nums'] },
  meta: { fontSize: 13.5, color: colors.muted, marginTop: 2 },
  empty: { color: colors.muted, textAlign: 'center', marginTop: 32, paddingHorizontal: 16 },
  more: { backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line },
});
