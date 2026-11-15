// On duty: live camera detection. Violations are saved, signed and uploaded automatically;
// the officer cannot edit, delete or hold back evidence from this screen. The Uploads panel
// only shows upload progress.
import { useEffect, useRef, useState } from 'react';
import { PermissionsAndroid, Pressable, ScrollView, Text, View, useWindowDimensions } from 'react-native';
import { router } from 'expo-router';
import { useKeepAwake } from 'expo-keep-awake';
import { RsafeCameraView } from '../../modules/rsafe-detector';
import { currentSession, endDuty, onSession, recordViolation, refreshLocation, signOut, startDuty } from '../services/session';
import { onSyncStatus } from '../services/sync';
import { Button, ErrorText, Screen, colors, styles } from '../ui';
import UploadsPanel from '../components/UploadsPanel';

function KeepAwake() {
  useKeepAwake();
  return null;
}

export default function Duty() {
  // start from the real session: an empty first value would bounce straight back to /login
  const [session, setSession] = useState(currentSession);
  const [sync, setSync] = useState(null);
  const [engine, setEngine] = useState({ state: 'idle' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [captured, setCaptured] = useState(0);
  const [cameraOk, setCameraOk] = useState(false);
  const [showUploads, setShowUploads] = useState(false);
  // TEST: camera size for detection + evidence photos ('1080p' | '4k')
  const [quality, setQuality] = useState('1080p');
  const { width, height } = useWindowDimensions();
  const landscape = width > height;
  const pendingWrites = useRef(Promise.resolve());
  // The detector reports motorcycles still in view AFTER duty ends: credit them to that duty.
  const lastDuty = useRef(null);
  const stoppedWaiters = useRef([]);

  useEffect(() => onSession((s) => {
    if (s.duty) lastDuty.current = { dutyId: s.duty.id, officerId: s.duty.officer_id };
    setSession(s);
  }), []);
  useEffect(() => onSyncStatus(setSync), []);
  useEffect(() => {
    PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA).then((r) =>
      setCameraOk(r === PermissionsAndroid.RESULTS.GRANTED),
    );
    refreshLocation();
    const t = setInterval(refreshLocation, 60000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (!session.officer) router.replace('/login');
  }, [session.officer]);

  const onDuty = !!session.duty;

  function onViolation({ nativeEvent }) {
    // queue one at a time so violation numbers follow capture order
    const ctx = lastDuty.current ?? {};
    // same text the detector wrote to event.json, so live and recovered records match
    pendingWrites.current = pendingWrites.current
      .then(() => recordViolation(JSON.parse(nativeEvent.json), ctx))
      .then(() => setCaptured((n) => n + 1))
      .catch((e) => setError(`Could not save a violation: ${e.message}`));
  }

  async function toggleDuty() {
    setBusy(true);
    setError('');
    try {
      if (onDuty) await endDuty();
      else {
        setCaptured(0);
        await startDuty();
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function onStatus({ nativeEvent }) {
    setEngine(nativeEvent);
    if (nativeEvent.state === 'stopped') {
      stoppedWaiters.current.forEach((resolve) => resolve());
      stoppedWaiters.current = [];
    }
  }

  async function leave() {
    setBusy(true);
    if (onDuty) {
      // wait (max 25 s) for the detector to hand over motorcycles still in view; their plates
      // are read and photos saved in the background first
      const stopped = new Promise((resolve) => {
        stoppedWaiters.current.push(resolve);
        setTimeout(resolve, 25000);
      });
      await endDuty();
      await stopped;
    }
    await pendingWrites.current;
    await signOut();
    setBusy(false);
  }

  const syncLine = !sync
    ? ''
    : sync.blocked
      ? `Uploads refused by server (${sync.blocked}). Tell an admin.`
      : sync.counts.pending > 0
        ? `${sync.counts.pending} waiting to upload${sync.online ? (sync.syncing ? ' · uploading…' : '') : ' · offline'}`
        : sync.online ? 'All violations uploaded' : 'Offline · nothing waiting';

  // Portrait: camera on top, controls below. Landscape: camera left, controls in a side panel.
  // The camera stays the first child in both layouts, so turning the phone never restarts it.
  return (
    <Screen style={{ padding: 0, flexDirection: landscape ? 'row' : 'column' }}>
      {onDuty ? <KeepAwake /> : null}
      <View style={{ flex: 1, backgroundColor: '#000' }}>
        {cameraOk ? (
          <RsafeCameraView
            style={{ flex: 1 }}
            active={onDuty}
            quality={quality}
            onStatus={onStatus}
            onError={({ nativeEvent }) => setError(nativeEvent.message)}
            onViolation={onViolation}
          />
        ) : (
          <Text style={{ color: '#fff', padding: 20 }}>Camera permission is needed for detection.</Text>
        )}
        {onDuty ? (
          <View style={{ position: 'absolute', top: 10, left: 10, backgroundColor: '#000a', borderRadius: 8, padding: 8 }}>
            <Text style={{ color: '#fff', fontWeight: '600' }}>
              {engine.state === 'loading'
                ? 'Loading models…'
                : engine.state === 'failed'
                  ? 'Detector not running (see error below)'
                  : engine.fps ? `${engine.fps.toFixed(1)} frames/s` : 'Starting…'}
            </Text>
            {engine.fps ? (
              // per-stage timing: what to read out when tuning on a real road
              <Text style={{ color: '#ddd', fontSize: 11, marginTop: 2 }}>
                {`vehicle ${engine.vehicleMs.toFixed(0)} ms · helmet ${engine.helmetPerSec.toFixed(1)}/s`
                  + ` (${engine.helmetMs.toFixed(0)} ms, delay ${engine.helmetLagMs.toFixed(0)} ms)`
                  + `\nqueue ${engine.queue} · tracks ${engine.tracks}`
                  + (engine.plateMs ? ` · plate ${engine.plateMs} ms` : '')
                  + ` · ${engine.exposure}`
                  + (engine.frameW
                    ? `
${engine.frameW}×${engine.frameH} · convert ${engine.convertMs.toFixed(0)} ms`
                      + ` · memory ${engine.memoryMb.toFixed(0)} MB`
                      + (engine.photoKb ? ` · last photo ${(engine.photoKb / 1024).toFixed(1)} MB` : '')
                    : '')}
              </Text>
            ) : null}
          </View>
        ) : null}
        {onDuty ? (
          // TEST switch: compare speed, memory and photo size on the real road
          <Pressable
            onPress={() => setQuality((q) => (q === '4k' ? '1080p' : '4k'))}
            accessibilityRole="button"
            accessibilityLabel={`Camera quality ${quality === '4k' ? '4K' : '1080p'}. Tap to switch.`}
            style={({ pressed }) => ({
              position: 'absolute', top: 10, right: 10, backgroundColor: quality === '4k' ? '#c0392bdd' : '#000a',
              borderRadius: 8, paddingVertical: 10, paddingHorizontal: 14, opacity: pressed ? 0.7 : 1,
            })}
          >
            <Text style={{ color: '#fff', fontWeight: '700' }}>{quality === '4k' ? '4K' : '1080p'}</Text>
          </Pressable>
        ) : null}
      </View>

      <ScrollView
        style={landscape ? { width: Math.min(360, Math.max(280, width * 0.36)), flexGrow: 0 } : { flexGrow: 0 }}
        contentContainerStyle={{ padding: 16, paddingTop: 10 }}
      >
        <View style={{ marginBottom: 10 }}>
          <Text style={{ fontSize: 18, fontWeight: '700', color: colors.ink }}>
            {session.officer?.full_name ?? ''}
          </Text>
          <Text style={{ color: onDuty ? colors.good : colors.muted, marginTop: 2 }}>
            {onDuty ? `On duty since ${new Date(session.duty.started_at).toLocaleTimeString()}` : 'Off duty'}
          </Text>
        </View>
        <Pressable
          onPress={() => setShowUploads(true)}
          accessibilityRole="button"
          accessibilityHint="Shows the upload progress of each violation"
          style={({ pressed }) => [styles.card, { flexDirection: 'row', justifyContent: 'space-between', opacity: pressed ? 0.8 : 1 }]}
        >
          <View>
            <Text style={{ color: colors.muted }}>Captured this duty</Text>
            <Text style={{ fontSize: 22, fontWeight: '700', color: colors.ink }}>{captured}</Text>
          </View>
          <View style={{ alignItems: 'flex-end', flexShrink: 1, marginLeft: 12 }}>
            <Text style={{ color: colors.muted }}>Upload</Text>
            <Text style={{ color: sync?.blocked ? colors.danger : colors.ink, textAlign: 'right' }}>{syncLine}</Text>
            <Text style={{ color: colors.primary, fontWeight: '600', marginTop: 4 }}>See uploads ›</Text>
          </View>
        </Pressable>
        <ErrorText>{error}</ErrorText>
        <Button title={onDuty ? 'End duty' : 'Start duty'} kind={onDuty ? 'danger' : 'primary'} onPress={toggleDuty} busy={busy} disabled={!cameraOk} />
        <Button title="Sign out" kind="secondary" onPress={leave} disabled={busy} />
      </ScrollView>
      {/* drawn over the duty screen, not a new route: the camera keeps detecting underneath */}
      {showUploads ? <UploadsPanel onClose={() => setShowUploads(false)} /> : null}
    </Screen>
  );
}
