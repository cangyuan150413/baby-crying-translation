import React, { useEffect, useRef, useState } from 'react';
import { FlatList, SafeAreaView, StyleSheet, Text, TextInput, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';

const REASONS = {
  hungry: '饿了', tired: '困了', discomfort: '不舒服（尿布/冷热）',
  pain: '疼痛/肠绞痛', burp: '需要拍嗝',
};

export default function App() {
  const [host, setHost] = useState('192.168.1.100:8000');
  const [status, setStatus] = useState('连接中…');
  const [events, setEvents] = useState([]);
  const timer = useRef(null);

  useEffect(() => {
    let ws; let closed = false;
    const connect = () => {
      ws = new WebSocket(`ws://${host}/ws/subscribe`);
      ws.onopen = () => setStatus('已连接');
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === 'cry') setEvents((p) => [{ id: String(m.ts), ...m }, ...p].slice(0, 100));
      };
      ws.onclose = () => {
        setStatus('已断开，重连中…');
        if (!closed) timer.current = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => { closed = true; clearTimeout(timer.current); ws && ws.close(); };
  }, [host]);

  const latest = events[0];
  return (
    <SafeAreaView style={s.c}>
      <StatusBar style="auto" />
      <TextInput style={s.in} value={host} onChangeText={setHost} autoCapitalize="none" />
      <Text style={s.st}>{status}</Text>
      {latest && (
        <View style={s.card}>
          <Text style={s.big}>{REASONS[latest.label] || latest.label}</Text>
          <Text>置信度 {(latest.confidence * 100).toFixed(0)}%</Text>
        </View>
      )}
      <FlatList data={events} keyExtractor={(i) => i.id} renderItem={({ item }) => (
        <Text style={s.row}>
          {new Date(item.ts * 1000).toLocaleTimeString()}  {REASONS[item.label] || item.label}
          {' '}({(item.confidence * 100).toFixed(0)}%)
        </Text>
      )} />
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  c: { flex: 1, padding: 16, paddingTop: 48 },
  in: { borderWidth: 1, borderColor: '#ccc', padding: 8, borderRadius: 6 },
  st: { marginVertical: 8, color: '#666' },
  card: { padding: 16, backgroundColor: '#ffe9ef', borderRadius: 10, marginBottom: 12 },
  big: { fontSize: 28, fontWeight: 'bold' },
  row: { paddingVertical: 6 },
});
