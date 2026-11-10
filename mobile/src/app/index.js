// Decides where to go: register the phone -> sign in -> duty.
import { useEffect, useState } from 'react';
import { ActivityIndicator } from 'react-native';
import { Redirect } from 'expo-router';
import { ensureIdentity } from '../services/device';
import { currentSession } from '../services/session';
import { Screen, colors } from '../ui';

export default function Index() {
  const [target, setTarget] = useState(null);

  useEffect(() => {
    ensureIdentity().then((id) => {
      if (!id.camera) setTarget('/register');
      else if (!currentSession().officer) setTarget('/login');
      else setTarget('/duty');
    });
  }, []);

  if (target) return <Redirect href={target} />;
  return (
    <Screen style={{ justifyContent: 'center' }}>
      <ActivityIndicator size="large" color={colors.primary} />
    </Screen>
  );
}
