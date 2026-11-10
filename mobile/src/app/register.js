// One-time phone registration with the 8-character code from an admin.
import { useState } from 'react';
import { router } from 'expo-router';
import { registerDevice } from '../services/api';
import { saveCamera } from '../services/device';
import { Button, ErrorText, Field, Form, Screen, Title } from '../ui';

const MESSAGES = {
  invalid_or_expired_code: 'That code is wrong or has expired. Ask an admin for a new one.',
  device_id_in_use: 'This phone is already registered to another camera.',
  network: 'No internet. Registration needs a connection once.',
};

export default function Register() {
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit() {
    setBusy(true);
    setError('');
    try {
      const camera = await registerDevice(code);
      await saveCamera(camera);
      router.replace('/login');
    } catch (e) {
      setError(MESSAGES[e.code] ?? (e.code ? e.message : MESSAGES.network));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen>
      <Form>
      <Title sub="An admin creates this phone in the R-SAFE dashboard and gives you a one-time code. This is done once, with internet.">
        Register this phone
      </Title>
      <Field
        label="Registration code"
        value={code}
        onChangeText={(t) => setCode(t.toUpperCase())}
        autoCapitalize="characters"
        autoCorrect={false}
        maxLength={8}
        placeholder="e.g. 7F3A9C21"
      />
      <Button title="Register" onPress={submit} busy={busy} disabled={code.trim().length !== 8} />
      <ErrorText>{error}</ErrorText>
      </Form>
    </Screen>
  );
}
