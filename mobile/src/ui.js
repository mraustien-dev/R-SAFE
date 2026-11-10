// Shared look for every screen.
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export const colors = {
  bg: '#F5F7FA',
  card: '#FFFFFF',
  ink: '#111827',
  muted: '#6B7280',
  line: '#E5E7EB',
  primary: '#0B5FFF',
  danger: '#DC2626',
  good: '#15803D',
  warn: '#B45309',
};

export function Screen({ children, style }) {
  return <SafeAreaView style={[styles.screen, style]}>{children}</SafeAreaView>;
}

/** Scrolling form column: fits short landscape screens, and stays readable on wide ones. */
export function Form({ children }) {
  return (
    <ScrollView
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={{ flexGrow: 1, width: '100%', maxWidth: 560, alignSelf: 'center' }}
    >
      {children}
    </ScrollView>
  );
}

export function Title({ children, sub }) {
  return (
    <View style={{ marginBottom: 20 }}>
      <Text style={styles.title}>{children}</Text>
      {sub ? <Text style={styles.sub}>{sub}</Text> : null}
    </View>
  );
}

export function Button({ title, onPress, busy, kind = 'primary', disabled }) {
  const bg = kind === 'primary' ? colors.primary : kind === 'danger' ? colors.danger : colors.card;
  const fg = kind === 'secondary' ? colors.ink : '#fff';
  return (
    <Pressable
      onPress={onPress}
      disabled={busy || disabled}
      style={({ pressed }) => [
        styles.button,
        { backgroundColor: bg, opacity: disabled ? 0.5 : pressed ? 0.85 : 1 },
        kind === 'secondary' && { borderWidth: 1, borderColor: colors.line },
      ]}
    >
      {busy ? <ActivityIndicator color={fg} /> : <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>}
    </Pressable>
  );
}

export function Field({ label, ...props }) {
  return (
    <View style={{ marginBottom: 14 }}>
      <Text style={styles.label}>{label}</Text>
      <TextInput placeholderTextColor={colors.muted} style={styles.input} {...props} />
    </View>
  );
}

export function ErrorText({ children }) {
  return children ? <Text style={styles.error}>{children}</Text> : null;
}

export const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg, padding: 20 },
  title: { fontSize: 26, fontWeight: '700', color: colors.ink },
  sub: { fontSize: 15, color: colors.muted, marginTop: 6, lineHeight: 21 },
  label: { fontSize: 14, fontWeight: '600', color: colors.ink, marginBottom: 6 },
  input: {
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.line, borderRadius: 10,
    paddingHorizontal: 14, paddingVertical: 12, fontSize: 16, color: colors.ink,
  },
  button: { borderRadius: 10, paddingVertical: 14, alignItems: 'center', marginTop: 8, minHeight: 50, justifyContent: 'center' },
  buttonText: { fontSize: 16, fontWeight: '600' },
  error: { color: colors.danger, marginTop: 10, fontSize: 14 },
  card: { backgroundColor: colors.card, borderRadius: 12, padding: 16, borderWidth: 1, borderColor: colors.line },
});
