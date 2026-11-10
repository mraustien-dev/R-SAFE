// Online, once per officer per phone: email + password, then choose a PIN for offline use.
import { useState } from 'react';
import { router } from 'expo-router';
import { enrollWithPassword } from '../services/session';
import { Button, ErrorText, Field, Form, Screen, Title } from '../ui';

const MESSAGES = {
  not_an_officer: 'This account is not an officer or admin.',
  officer_revoked_on_device: 'An admin removed you from this phone.',
  bad_officer_token: 'Sign-in failed. Try again.',
};

export default function Enroll() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [pin, setPin] = useState('');
  const [pin2, setPin2] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function submit() {
    setError('');
    if (pin !== pin2) return setError('The two PINs do not match.');
    setBusy(true);
    try {
      await enrollWithPassword(email, password, pin);
      router.back();
    } catch (e) {
      setError(MESSAGES[e.code] ?? (e.code || e.status ? e.message : 'No internet. Adding an officer needs a connection.'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen>
      <Form>
        <Title sub="Use your R-SAFE dashboard account. After this, you sign in on this phone with your PIN, even without internet.">
          Add officer
        </Title>
        <Field label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" keyboardType="email-address" autoComplete="email" />
        <Field label="Password" value={password} onChangeText={setPassword} secureTextEntry />
        <Field label="New PIN (4–6 digits)" value={pin} onChangeText={(t) => setPin(t.replace(/\D/g, ''))} keyboardType="number-pad" secureTextEntry maxLength={6} />
        <Field label="Repeat PIN" value={pin2} onChangeText={(t) => setPin2(t.replace(/\D/g, ''))} keyboardType="number-pad" secureTextEntry maxLength={6} />
        <Button title="Add officer" onPress={submit} busy={busy} disabled={!email || !password || pin.length < 4} />
        <Button title="Cancel" kind="secondary" onPress={() => router.back()} />
        <ErrorText>{error}</ErrorText>
      </Form>
    </Screen>
  );
}
