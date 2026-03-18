import { useEffect, useMemo, useState } from "react";
import Arena from "./Arena";

const DEFAULT_STATE = {
  ball: [0, 0, 1.5],
  bot1: [-8, 0, 0.45, 0],
  bot2: [8, 0, 0.45, Math.PI],
  score: { bot1: 0, bot2: 0 },
  training_iteration: 0,
};

export default function App() {
  const [state, setState] = useState(DEFAULT_STATE);
  const [connected, setConnected] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState(null);

  const wsUrl = useMemo(() => {
    return import.meta.env.VITE_WS_URL || "ws://localhost:8001";
  }, []);

  useEffect(() => {
    let socket;
    let mounted = true;

    const connect = () => {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        if (!mounted) return;
        setConnected(true);
      };

      socket.onmessage = (event) => {
        if (!mounted) return;
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === "connected") return;
          if (!payload.ball || !payload.bot1 || !payload.bot2) return;
          setState(payload);
          setLastMessageAt(Date.now());
        } catch (err) {
          console.error("WS parse error", err);
        }
      };

      socket.onclose = () => {
        if (!mounted) return;
        setConnected(false);
        setTimeout(connect, 1000);
      };

      socket.onerror = () => {
        socket.close();
      };
    };

    connect();

    return () => {
      mounted = false;
      if (socket && socket.readyState <= 1) socket.close();
    };
  }, [wsUrl]);

  const freshness = lastMessageAt ? Math.max(0, (Date.now() - lastMessageAt) / 1000).toFixed(1) : "-";

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={styles.title}>Rocket League 1v1 RL Playground</h1>
        <div style={styles.pills}>
          <span style={{ ...styles.pill, background: connected ? "#166534" : "#991b1b" }}>
            {connected ? "WS Connected" : "WS Disconnected"}
          </span>
          <span style={styles.pill}>Iteration: {state.training_iteration ?? 0}</span>
          <span style={styles.pill}>Score {state.score?.bot1 ?? 0} - {state.score?.bot2 ?? 0}</span>
          <span style={styles.pill}>Last Tick: {freshness}s</span>
        </div>
        <div style={styles.boostRow}>
          <div style={styles.boostItem}>
            <span style={styles.boostLabel}>Blue Boost</span>
            <div style={styles.boostTrack}>
              <div style={{...styles.boostFill, width: `${state.bot1_boost ?? 33}%`, background: "linear-gradient(90deg, #1d4ed8, #38bdf8)"}} />
            </div>
          </div>
          <div style={styles.boostItem}>
            <span style={styles.boostLabel}>Orange Boost</span>
            <div style={styles.boostTrack}>
              <div style={{...styles.boostFill, width: `${state.bot2_boost ?? 33}%`, background: "linear-gradient(90deg, #ea580c, #fb923c)"}} />
            </div>
          </div>
        </div>
      </div>
      <div style={styles.viewport}>
        <Arena state={state} />
      </div>
    </div>
  );
}

const styles = {
  page: {
    width: "100vw",
    height: "100vh",
    margin: 0,
    overflow: "hidden",
    fontFamily: "Segoe UI, Tahoma, sans-serif",
    background: "radial-gradient(circle at 20% 0%, #153452 0%, #081522 55%, #050b12 100%)",
    color: "#e5eef8",
    display: "grid",
    gridTemplateRows: "auto 1fr",
  },
  header: {
    padding: "10px 14px",
    backdropFilter: "blur(8px)",
    background: "rgba(5, 14, 26, 0.75)",
    borderBottom: "1px solid rgba(255,255,255,0.1)",
  },
  title: {
    margin: "0 0 8px 0",
    fontWeight: 700,
    fontSize: "1.0rem",
    letterSpacing: "0.02em",
  },
  pills: {
    display: "flex",
    flexWrap: "wrap",
    gap: "8px",
  },
  pill: {
    padding: "4px 8px",
    borderRadius: "999px",
    background: "#1f2937",
    border: "1px solid rgba(255,255,255,0.18)",
    fontSize: "0.8rem",
  },
  viewport: {
    minHeight: 0,
  },
  boostRow: {
    display: "flex",
    gap: "12px",
    marginTop: "8px",
  },
  boostItem: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    gap: "6px",
  },
  boostLabel: {
    fontSize: "0.72rem",
    whiteSpace: "nowrap",
    opacity: 0.8,
  },
  boostTrack: {
    flex: 1,
    height: "8px",
    borderRadius: "4px",
    background: "rgba(255,255,255,0.1)",
    overflow: "hidden",
  },
  boostFill: {
    height: "100%",
    borderRadius: "4px",
    transition: "width 0.15s ease",
  },
};
