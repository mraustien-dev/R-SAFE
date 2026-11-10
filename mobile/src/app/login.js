// Officer sign-in: pick your name, enter your PIN. Works offline.
import { useCallback, useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { router, useFocusEffect } from 'expo-router';
import { listOfficers, offlineHoursLeft, signInWithPin } from '../services/session';
import { ensureIdentity } from '../services/device';
import { Button, ErrorText, Field, Form, Screen, Title, colors, styles } from '../ui';

export default function Login() {
  const [officers, setOfficers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [pin, setPin] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [camera, setCamera] = useState(null);
  const [hoursLeft, setHoursLeft] = useState(null);

  useFocusEffect(
    useCallback(() => {
      listOfficers().then(setOfficers);
      ensureIdentity().then((id) => setCamera(id.camera));
      offlineHoursLeft().then(setHoursLeft);
    }, []),
  );

  async function submit() {
    setBusy(true);
    setError('');
    try {
      await signInWithPin(selected.id, pin);
      setPin('');
      router.replace('/duty');
    } catch (e) {
      setError(e.message);
      listOfficers().then(setOfficers);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen>
      <Form>
      <Title sub={camera ? `Phone: ${camera.name}${camera.location_name ? ` · ${camera.location_name}` : ''}` : undefined}>
        Sign in
      </Title>

      {officers.length === 0 ? (
        <Text style={styles.sub}>No officers on this phone yet. Add yourself once while online.</Text>
      ) : (
        <View style={{ marginBottom: 16 }}>
          {officers.map((item) => (
            <Pressable
              key={item.id}
              onPress={() => { setSelected(item); setError(''); setPin(''); }}
              style={[
                styles.card,
                { marginBottom: 8 },
                selected?.id === item.id && { borderColor: colors.primary, borderWidth: 2 },
              ]}
            >
              <Text style={{ fontSize: 16, fontWeight: '600', color: colors.ink }}>{item.full_name}</Text>
              <Text style={{ color: colors.muted, marginTop: 2 }}>{item.email}</Text>
            </Pressable>
          ))}
        </View>
      )}

      {selected ? (
        <View>
          <Field
            label={`PIN for ${selected.full_name}`}
            value={pin}
            onChangeText={(t) => setPin(t.replace(/\D/g, ''))}
            keyboardType="number-pad"
            secureTextEntry
            maxLength={6}
            autoFocus
          />
          <Button title="Sign in" onPress={submit} busy={busy} disabled={pin.length < 4} />
        </View>
      ) : null}
      <ErrorText>{error}</ErrorText>

      <View style={{ flex: 1 }} />
      {hoursLeft !== null && hoursLeft < 24 ? (
        <Text style={{ color: colors.warn, marginBottom: 8 }}>
          {hoursLeft <= 0
            ? 'Offline sign-in is locked until this phone connects to the internet.'
            : `Connect to the internet within ${Math.ceil(hoursLeft)} h to keep offline sign-in working.`}
        </Text>
      ) : null}
      <Button title="Add officer to this phone (online)" kind="secondary" onPress={() => router.push('/enroll')} />
      </Form>
    </Screen>
  );
}
