import React, { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";
import {
  ShieldAlert, ShieldCheck, Activity, Cpu, Play, Pause, RotateCcw, Zap, Network
} from "lucide-react";

export default function App() {
  const wsRef = useRef(null);
  const graphContainerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(true);
  const [windowSize, setWindowSize] = useState(2000);
  const [threshold, setThreshold] = useState(0.15);
  const [speedSec, setSpeedSec] = useState(1.0);

  const [stats, setStats] = useState({
    totalScanned: 0,
    totalAttacks: 0,
    latencyMs: 0,
    throughput: 0,
    progress: 0,
    windowId: 0,
  });

  const [history, setHistory] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/stream");
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setIsRunning(data.is_running);

      setStats((prev) => ({
        totalScanned: prev.totalScanned + data.total_flows,
        totalAttacks: prev.totalAttacks + data.attack_flows,
        latencyMs: data.latency_ms,
        throughput: data.throughput,
        progress: data.progress,
        windowId: data.window_id,
      }));

      setHistory((prev) => [
        ...prev.slice(-29),
        {
          win: `#${data.window_id}`,
          Normal: data.normal_flows,
          Attack: data.attack_flows,
        },
      ]);

      setGraphData(data.graph);

      if (data.alerts && data.alerts.length > 0) {
        setAlerts((prev) => [...data.alerts, ...prev].slice(0, 40));
      }
    };

    return () => ws.close();
  }, []);

  const sendCommand = (payload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  };

  const handleApplyConfig = (newWin, newTh, newSpeed) => {
    setWindowSize(newWin);
    setThreshold(newTh);
    setSpeedSec(newSpeed);
    sendCommand({
      action: "config",
      window_size: newWin,
      threshold: newTh,
      speed_sec: newSpeed,
    });
  };

  const attackRate =
    stats.totalScanned > 0
      ? ((stats.totalAttacks / stats.totalScanned) * 100).toFixed(2)
      : "0.00";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 font-sans">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between bg-slate-900/90 border border-slate-800 rounded-xl px-5 py-3.5 mb-4 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600/20 border border-indigo-500/40 rounded-lg">
            <Network className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide text-white">
              E-GraphSAGE Streaming IDS — SOC Monitor
            </h1>
            <p className="text-xs text-slate-400">
              Phát hiện xâm nhập mạng thời gian thực bằng Đồ thị Động (CIC-IDS2017)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-2 sm:mt-0">
          <span
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full font-medium border ${
              connected
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : "bg-rose-500/10 text-rose-400 border-rose-500/30"
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`} />
            {connected ? `Stream Active (${stats.progress}%)` : "Disconnected"}
          </span>

          <button
            onClick={() => sendCommand({ action: "jump_attack" })}
            className="flex items-center gap-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer"
          >
            <Zap className="w-3.5 h-3.5" /> Nhảy tới đợt Tấn công
          </button>

          <button
            onClick={() => sendCommand({ action: "toggle" })}
            className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 text-white px-3.5 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer"
          >
            {isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isRunning ? "Tạm dừng" : "Tiếp tục"}
          </button>

          <button
            onClick={() => {
              sendCommand({ action: "reset" });
              setStats({ totalScanned: 0, totalAttacks: 0, latencyMs: 0, throughput: 0, progress: 0, windowId: 0 });
              setHistory([]);
              setAlerts([]);
            }}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 transition cursor-pointer"
            title="Reset luồng dữ liệu"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>TỔNG LƯU LƯỢNG ĐÃ QUÉT</span>
            <Activity className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-white">{stats.totalScanned.toLocaleString()}</div>
          <div className="text-xs text-slate-400 mt-1">Cửa sổ hiện tại: #{stats.windowId}</div>
        </div>

        <div className="bg-slate-900 border border-rose-900/50 rounded-xl p-4">
          <div className="flex items-center justify-between text-rose-400 text-xs mb-1">
            <span>LUỒNG TẤN CÔNG PHÁT HIỆN</span>
            <ShieldAlert className="w-4 h-4 text-rose-500" />
          </div>
          <div className="text-2xl font-bold text-rose-400">{stats.totalAttacks.toLocaleString()}</div>
          <div className="text-xs text-rose-300/80 mt-1">Chiếm {attackRate}% tổng lưu lượng</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>ĐỘ TRỄ SUY LUẬN GNN</span>
            <Cpu className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-300">{stats.latencyMs} ms</div>
          <div className="text-xs text-slate-400 mt-1">Xử lý 1 cửa sổ {windowSize} flows</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>THÔNG LƯỢNG HỆ THỐNG</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">{stats.throughput.toLocaleString()}</div>
          <div className="text-xs text-slate-400 mt-1">Flows / giây (Real-time)</div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-slate-900/70 border border-slate-800 rounded-xl px-5 py-3 mb-4 text-xs">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-slate-400">Ngưỡng cảnh báo (Threshold):</span>
            <span className="font-bold text-indigo-400">{threshold.toFixed(2)}</span>
          </div>
          <input
            type="range" min="0.05" max="0.80" step="0.05"
            value={threshold}
            onChange={(e) => handleApplyConfig(windowSize, parseFloat(e.target.value), speedSec)}
            className="w-full accent-indigo-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <span className="text-slate-400">Kích thước cửa sổ (Window Size):</span>
            <span className="font-bold text-sky-400">{windowSize} flows</span>
          </div>
          <input
            type="range" min="500" max="5000" step="500"
            value={windowSize}
            onChange={(e) => handleApplyConfig(parseInt(e.target.value), threshold, speedSec)}
            className="w-full accent-sky-500 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <span className="text-slate-400">Tốc độ bắn luồng (Delay):</span>
            <span className="font-bold text-emerald-400">{speedSec.toFixed(1)}s / window</span>
          </div>
          <input
            type="range" min="0.3" max="2.5" step="0.2"
            value={speedSec}
            onChange={(e) => handleApplyConfig(windowSize, threshold, parseFloat(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>
      </div>

      {/* Charts & Live Network Graph */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 mb-4">
        {/* Biểu đồ luồng lưu lượng */}
        <div className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">
            Biến động Lưu lượng Bình thường vs Tấn công theo Cửa sổ
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="win" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }} />
                <Area type="monotone" dataKey="Normal" stackId="1" stroke="#38bdf8" fill="#0284c7" fillOpacity={0.35} />
                <Area type="monotone" dataKey="Attack" stackId="1" stroke="#f43f5e" fill="#e11d48" fillOpacity={0.65} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Đồ thị mạng động */}
        <div ref={graphContainerRef} className="lg:col-span-6 bg-slate-900 border border-slate-800 rounded-xl p-4 overflow-hidden">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-slate-200">
              Đồ thị Tương tác IP Thời gian Thực ({graphData.nodes.length} Nodes)
            </h2>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Attacker</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" /> Target</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" /> Normal</span>
            </div>
          </div>
          <div className="h-72 bg-slate-950 rounded-lg border border-slate-800/80 flex items-center justify-center">
            <ForceGraph2D
              width={520}
              height={280}
              graphData={graphData}
              nodeRelSize={5}
              nodeColor={(n) =>
                n.role === "attacker" ? "#f43f5e" : n.role === "victim" ? "#fbbf24" : "#38bdf8"
              }
              nodeLabel={(n) => `${n.id} (${n.role.toUpperCase()})`}
              linkColor={(l) => (l.is_atk ? "#f43f5e" : "#1e293b")}
              linkWidth={(l) => (l.is_atk ? 2 : 1)}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              cooldownTicks={40}
            />
          </div>
        </div>
      </div>

      {/* Live Alert Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-rose-400 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4" /> Nhật ký Cảnh báo Xâm nhập Trực tiếp (Live SOC Alerts)
          </h2>
          <span className="text-xs text-slate-400">Hiển thị {alerts.length} cảnh báo gần nhất</span>
        </div>

        <div className="overflow-x-auto max-h-60">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 sticky top-0">
              <tr>
                <th className="py-2.5 px-3">Thời gian (Timestamp)</th>
                <th className="py-2.5 px-3">IP Nguồn (Attacker)</th>
                <th className="py-2.5 px-3">IP Đích (Target)</th>
                <th className="py-2.5 px-3">Độ tin cậy GNN</th>
                <th className="py-2.5 px-3">Phân loại tấn công</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {alerts.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-6 text-center text-slate-500">
                    Chưa phát hiện tấn công trong cửa sổ hiện tại — Hệ thống an toàn (hoặc bấm "Nhảy tới đợt Tấn công" để test ngay)
                  </td>
                </tr>
              ) : (
                alerts.map((a) => (
                  <tr key={a.id} className="hover:bg-rose-950/20 transition">
                    <td className="py-2 px-3 text-slate-300 font-mono">{a.timestamp}</td>
                    <td className="py-2 px-3 text-rose-400 font-mono font-semibold">{a.src_ip}</td>
                    <td className="py-2 px-3 text-amber-300 font-mono">{a.dst_ip}</td>
                    <td className="py-2 px-3">
                      <span className="bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2 py-0.5 rounded font-semibold">
                        {a.prob}%
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-200 font-medium">{a.true_label}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}