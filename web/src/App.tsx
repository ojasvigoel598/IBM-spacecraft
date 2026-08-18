import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import {
  Activity,
  AlertTriangle,
  Battery,
  Cpu,
  Gauge,
  Radio,
  Satellite,
  Thermometer,
  Zap,
  Play,
  Pause,
  RefreshCw,
  ChevronRight,
} from 'lucide-react'

// Same-origin by default (/api/*): served by the Vite dev proxy locally and
// by the Vercel Python function in production. Set VITE_API_URL to point at a
// separate API origin when the backend is hosted elsewhere.
const API = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '')

type Row = Record<string, number>
type Summary = {
  t: number
  max_time: number
  label: string
  telemetry: Record<string, number>
  physics: { power: string[] | null; thermal: string[] | null }
}

const SCENARIOS = [
  { id: 'none', label: 'NOMINAL', short: 'NOM' },
  { id: 'solar_degradation', label: 'SOLAR DEGRADATION', short: 'SOL' },
  { id: 'radiator_degradation', label: 'RADIATOR DEGRADATION', short: 'RAD' },
]

const fmtTime = (s: number) => {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `T+${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

const fmt = (v: number | undefined | null, d = 1) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : v.toFixed(d)

/* ---------------------------- SVG line chart ---------------------------- */
function LineChart({
  data,
  height = 110,
  color = '#5ac8fa',
  windowStart,
  windowEnd,
  current,
  min,
  max,
  unit,
}: {
  data: { t: number; v: number }[]
  height?: number
  color?: string
  windowStart: number
  windowEnd: number
  current: number
  min?: number
  max?: number
  unit?: string
}) {
  const width = 640
  const pad = 6
  const pts = data.filter((p) => p.t >= windowStart && p.t <= windowEnd)
  const lo = min ?? Math.min(...pts.map((p) => p.v), 0)
  const hi = max ?? Math.max(...pts.map((p) => p.v), 1)
  const span = Math.max(hi - lo, 1e-9)
  const x = (t: number) =>
    pad + ((t - windowStart) / Math.max(windowEnd - windowStart, 1)) * (width - pad * 2)
  const y = (v: number) => height - pad - ((v - lo) / span) * (height - pad * 2)
  // downsample for perf
  const step = Math.max(1, Math.floor(pts.length / 500))
  const line = pts
    .filter((_, i) => i % step === 0)
    .map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`)
    .join(' ')
  const cur = pts.length ? pts[pts.length - 1] : null
  const last = cur ?? { t: windowStart, v: lo }
  return (
    <div className="relative w-full min-w-0">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full min-w-0"
        preserveAspectRatio="none"
        style={{ height }}
      >
        <line x1={pad} y1={y(lo)} x2={width - pad} y2={y(lo)} stroke="oklch(0.3 0.02 250)" strokeWidth="1" />
        <line x1={pad} y1={y((lo + hi) / 2)} x2={width - pad} y2={y((lo + hi) / 2)} stroke="oklch(0.28 0.02 250)" strokeWidth="1" strokeDasharray="2 4" />
        <line x1={pad} y1={y(hi)} x2={width - pad} y2={y(hi)} stroke="oklch(0.3 0.02 250)" strokeWidth="1" />
        <line x1={x(current)} y1={pad} x2={x(current)} y2={height - pad} stroke="oklch(0.6 0.05 205 / 0.5)" strokeWidth="1" strokeDasharray="3 3" />
        {line && <polyline points={line} fill="none" stroke={color} strokeWidth="1.6" />}
      </svg>
      <div className="pointer-events-none absolute right-1 top-0 font-mono text-[10px] text-muted-foreground tnum">
        {unit ? `${fmt(last.v)} ${unit}` : fmt(last.v)}
      </div>
    </div>
  )
}

function Kpi({
  icon,
  label,
  value,
  unit,
  sub,
  tone = 'default',
}: {
  icon: React.ReactNode
  label: string
  value: string
  unit?: string
  sub?: string
  tone?: 'default' | 'ok' | 'warn' | 'bad'
}) {
  const toneColor =
    tone === 'ok'
      ? 'text-emerald-400'
      : tone === 'warn'
        ? 'text-amber-400'
        : tone === 'bad'
          ? 'text-red-400'
          : 'text-foreground'
  return (
    <Card size="sm">
      <CardContent className="flex flex-col gap-1">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5 uppercase tracking-wide">
            {icon}
            {label}
          </span>
        </div>
        <div className={`font-mono text-2xl font-semibold tnum ${toneColor}`}>
          {value}
          {unit && <span className="ml-1 text-xs font-normal text-muted-foreground">{unit}</span>}
        </div>
        {sub && <div className="font-mono text-[10px] text-muted-foreground tnum">{sub}</div>}
      </CardContent>
    </Card>
  )
}

/* ------------------------------ main app -------------------------------- */
export default function App() {
  const [mode, setMode] = useState('solar_degradation')
  const [rows, setRows] = useState<Row[] | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [t, setT] = useState(900)
  const [maxT, setMaxT] = useState(3600)
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)
  // live ingest
  const [liveFrames, setLiveFrames] = useState<Row[]>([])
  const [liveScore, setLiveScore] = useState<{ score: number; flag: number; source: number } | null>(null)
  const [liveRunning, setLiveRunning] = useState(false)
  const [models, setModels] = useState<any[] | null>(null)
  // code trace
  const [traceEvents, setTraceEvents] = useState<any[]>([])
  const [traceCursor, setTraceCursor] = useState(0)
  const [traceLive, setTraceLive] = useState(false)
  const [traceError, setTraceError] = useState<string | null>(null)
  const timerRef = useRef<number | null>(null)

  const WINDOW = 900

  // Honest Granite state from /api/health. "LIVE" is only shown after a real
  // watsonx call has actually succeeded — a present key alone is not proof
  // IBM answered (and a failed real call must never look like LIVE).
  const graniteLabel = (h: Record<string, unknown> | null) => {
    const g = (h?.granite ?? {}) as Record<string, unknown>
    const mode = String(g.mode ?? 'MOCK')
    const last = String(g.last_real_request ?? 'not_attempted')
    if (mode === 'REAL_READY' && last === 'succeeded') return 'GRANITE LIVE'
    if (mode === 'REAL_FAILED') return 'REAL FAILED · mock fallback'
    if (mode === 'REAL_READY') return 'REAL READY · untested'
    return 'MOCK FALLBACK'
  }

  const fetchJson = useCallback(async (path: string) => {
    const r = await fetch(`${API}${path}`)
    if (!r.ok) throw new Error(`${r.status} ${path}`)
    return r.json()
  }, [])

  // scenario data
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchJson(`/api/scenario/${mode}`)
      .then((d) => {
        if (cancelled) return
        setRows(d.rows)
        setMaxT(d.max_time)
        setT(Math.min(t, d.max_time))
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  // health + models once
  useEffect(() => {
    fetchJson('/api/health').then(setHealth).catch(() => {})
    fetchJson('/api/models').then((d) => setModels(d.models)).catch(() => {})
  }, [fetchJson])

  // summary at current t (debounced)
  useEffect(() => {
    const id = window.setTimeout(() => {
      fetchJson(`/api/summary/${mode}?t=${t}`)
        .then(setSummary)
        .catch(() => {})
    }, 120)
    return () => window.clearTimeout(id)
  }, [t, mode, fetchJson])

  // auto-advance for live stream
  useEffect(() => {
    if (!liveRunning) return
    timerRef.current = window.setInterval(() => {
      fetchJson(`/api/live/next?mode=${mode}&n=10`)
        .then((d) => {
          setLiveFrames((prev) => [...prev, ...d.frames].slice(-400))
          if (d.window_scored)
            setLiveScore({ score: d.anomaly_score, flag: d.anomaly_flag, source: d.anomaly_source })
        })
        .catch(() => {})
    }, 1500)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [liveRunning, mode, fetchJson])

  const refreshTrace = useCallback(
    (silent = true) => {
      fetchJson(`/api/trace?since=${traceCursor}&limit=300`)
        .then((d) => {
          setTraceError(null)
          if (d.events?.length) {
            setTraceEvents((prev) => [...prev, ...d.events].slice(-300))
            setTraceCursor(d.last_seq)
          }
        })
        .catch(() => !silent && setTraceError('trace unavailable'))
    },
    [fetchJson, traceCursor]
  )

  // poll the trace while LIVE is on
  useEffect(() => {
    if (!traceLive) return
    const id = window.setInterval(() => refreshTrace(), 2000)
    return () => window.clearInterval(id)
  }, [traceLive, refreshTrace])

  const advanceLive = useCallback(() => {
    fetchJson(`/api/live/next?mode=${mode}&n=30`)
      .then((d) => {
        setLiveFrames((prev) => [...prev, ...d.frames].slice(-400))
        if (d.window_scored)
          setLiveScore({ score: d.anomaly_score, flag: d.anomaly_flag, source: d.anomaly_source })
      })
      .catch(() => {})
  }, [mode, fetchJson])

  // derived: telemetry series + fault windows
  const series = useMemo(() => {
    if (!rows) return { solar: [], soc: [], temp: [], score: [], volt: [], heat: [] }
    const pick = (k: string) => rows.map((r) => ({ t: r.time_s, v: r[k] }))
    return {
      solar: pick('solar_power_w'),
      soc: pick('battery_soc'),
      temp: pick('temperature_c'),
      volt: pick('battery_voltage_v'),
      heat: pick('heat_in_w'),
      score: pick('anomaly_score'),
    }
  }, [rows])

  const faultWindows = useMemo(() => {
    if (!rows) return []
    const win: { start: number; end: number }[] = []
    let cur: { start: number } | null = null
    for (const r of rows) {
      if (r.anomaly_flag === 1 && !cur) cur = { start: r.time_s }
      if ((r.anomaly_flag === 0 || r.time_s === rows[rows.length - 1].time_s) && cur) {
        win.push({ start: cur.start, end: r.time_s })
        cur = null
      }
    }
    return win
  }, [rows])

  const activeWindow = faultWindows.find((w) => t >= w.start && t <= w.end)

  const tdata = summary?.telemetry
  const telemetryTone =
    !tdata ? 'default' : tdata.anomaly_flag === 1 ? 'bad' : tdata.anomaly_score < -0.2 ? 'warn' : 'ok'

  const sourceLabel = tdata?.anomaly_source === 1 ? 'POWER' : tdata?.anomaly_source === 2 ? 'THERMAL' : 'FULL'

  return (
    <div className="min-h-screen">
      {/* header */}
      <header className="sticky top-0 z-10 border-b border-border/60 bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-5 py-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/15 ring-1 ring-primary/40">
              <Satellite className="h-5 w-5 text-primary" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-[0.2em] text-foreground">
                MISSIONMIND
              </div>
              <div className="hidden text-[10px] uppercase tracking-wider text-muted-foreground md:block">
                Spacecraft Health &amp; Reliability Console
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge
              variant={activeWindow || (tdata?.anomaly_flag ?? 0) === 1 ? 'destructive' : 'outline'}
              className="font-mono"
            >
              <span
                className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full ${
                  activeWindow ? 'animate-pulse bg-red-400' : 'bg-emerald-400'
                }`}
              />
              {activeWindow ? 'FAULT ACTIVE' : 'NOMINAL'}
            </Badge>
            <div className="hidden font-mono text-xs text-muted-foreground sm:block tnum">
              {health?.status === 'ok' ? 'API ONLINE' : 'API …'}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 pb-16 pt-5">
        {/* scenario strip */}
        <div className="mb-4 flex min-w-0 flex-wrap items-center gap-2">
          {SCENARIOS.map((s) => (
            <Button
              key={s.id}
              variant={mode === s.id ? 'default' : 'outline'}
              size="sm"
              className="font-mono text-xs"
              onClick={() => setMode(s.id)}
            >
              {s.label}
            </Button>
          ))}
          <div className="ml-auto flex items-center gap-2">
            <Button variant="ghost" size="sm" className="font-mono text-xs" onClick={() => setT(600)}>
              ⚡ FAULT ONSET
            </Button>
            <Button variant="ghost" size="sm" className="font-mono text-xs" onClick={() => setT(0)}>
              ↺ START
            </Button>
            <Button variant="ghost" size="sm" className="font-mono text-xs" onClick={() => setT(maxT)}>
              ⟶ END
            </Button>
          </div>
        </div>

        {error && (
          <Card className="mb-4 border-destructive/50">
            <CardContent className="text-sm text-red-300">
              Backend unreachable — start it with{' '}
              <code className="font-mono text-xs">
                .venv/Scripts/python.exe -m uvicorn missionmind.viz.api_server:app --port 8100
              </code>
            </CardContent>
          </Card>
        )}

        {/* KPI row */}
        <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Kpi icon={<Zap className="h-3.5 w-3.5" />} label="Solar Power" value={fmt(tdata?.solar_power_w)} unit="W" tone={tdata && tdata.solar_power_w < 200 ? 'warn' : 'default'} />
          <Kpi icon={<Battery className="h-3.5 w-3.5" />} label="Battery SOC" value={fmt(tdata ? tdata.battery_soc * 100 : undefined, 1)} unit="%" tone={tdata && tdata.battery_soc < 0.4 ? 'bad' : tdata && tdata.battery_soc < 0.6 ? 'warn' : 'ok'} />
          <Kpi icon={<Activity className="h-3.5 w-3.5" />} label="Bus Voltage" value={fmt(tdata?.battery_voltage_v, 2)} unit="V" />
          <Kpi icon={<Thermometer className="h-3.5 w-3.5" />} label="Panel Temp" value={fmt(tdata?.temperature_c, 1)} unit="°C" tone={tdata && tdata.temperature_c > 20 ? 'bad' : tdata && tdata.temperature_c > 10 ? 'warn' : 'ok'} />
          <Kpi icon={<Gauge className="h-3.5 w-3.5" />} label="Anomaly Score" value={fmt(tdata?.anomaly_score, 3)} tone={telemetryTone} sub={tdata?.anomaly_flag === 1 ? `FLAG · SRC ${sourceLabel}` : 'NO FLAG'} />
          <Kpi icon={<Cpu className="h-3.5 w-3.5" />} label="Heat Load" value={fmt(tdata?.heat_in_w)} unit="W" sub={`out ${fmt(tdata?.heat_out_w)} W`} />
        </div>

        {/* main grid: charts + alerts */}
        <div className="mb-5 grid gap-4 lg:grid-cols-3">
          <div className="flex flex-col gap-4 lg:col-span-2">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div>
                  <CardTitle className="text-sm">TELEMETRY · {fmtTime(t)}</CardTitle>
                  <CardDescription className="font-mono text-[10px]">
                    {summary?.label ?? SCENARIOS.find((s) => s.id === mode)?.label} — window {fmtTime(Math.max(0, t - WINDOW))} → {fmtTime(t)}
                  </CardDescription>
                </div>
                <Badge variant="outline" className="font-mono text-[10px]">
                  {loading ? 'SOLVING…' : `${rows?.length ?? 0} samples`}
                </Badge>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">Solar power (W)</div>
                  <LineChart data={series.solar} windowStart={Math.max(0, t - WINDOW)} windowEnd={t} current={t} color="#5ac8fa" unit="W" />
                </div>
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">Battery SOC (%)</div>
                  <LineChart data={series.soc.map((p) => ({ t: p.t, v: p.v * 100 }))} windowStart={Math.max(0, t - WINDOW)} windowEnd={t} current={t} color="#4ade80" min={0} max={100} unit="%" />
                </div>
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">Panel temperature (°C)</div>
                  <LineChart data={series.temp} windowStart={Math.max(0, t - WINDOW)} windowEnd={t} current={t} color="#fb923c" unit="°C" />
                </div>
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    Anomaly score (ensemble)
                  </div>
                  <LineChart data={series.score} windowStart={Math.max(0, t - WINDOW)} windowEnd={t} current={t} color={tdata?.anomaly_flag === 1 ? '#f87171' : '#5ac8fa'} />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* right column: alerts + system state */}
          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  ACTIVE ALERTS
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {activeWindow ? (
                  <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3">
                    <div className="font-mono text-xs font-semibold text-red-300">
                      {mode === 'solar_degradation'
                        ? 'WARNING — SOLAR ARRAY DEGRADATION'
                        : mode === 'radiator_degradation'
                          ? 'WARNING — RADIATOR DEGRADATION'
                          : 'WARNING — ANOMALY'}
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-muted-foreground tnum">
                      Detected T+{fmtTime(activeWindow.start)} · Source {sourceLabel}
                    </div>
                    {summary?.physics && (
                      <ul className="mt-2 space-y-1 text-[11px] text-foreground/80">
                        {(summary.physics.power ?? []).map((p, i) => (
                          <li key={`p${i}`} className="flex items-start gap-1">
                            <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-amber-400" />
                            {p}
                          </li>
                        ))}
                        {(summary.physics.thermal ?? []).map((p, i) => (
                          <li key={`t${i}`} className="flex items-start gap-1">
                            <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-amber-400" />
                            {p}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ) : (
                  <div className="rounded-lg border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                    No active alert at {fmtTime(t)}.
                    {faultWindows.length > 0 && (
                      <span className="mt-1 block font-mono text-[10px] tnum">
                        {faultWindows.length} fault episode{faultWindows.length > 1 ? 's' : ''} this mission:
                        {faultWindows.map((w) => ` ${fmtTime(w.start)}→${fmtTime(w.end)}`).join(' · ')}
                      </span>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Radio className="h-4 w-4 text-primary" />
                  SYSTEM STATE
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Mission clock</span>
                  <span className="font-mono tnum">{fmtTime(t)} / {fmtTime(maxT)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Ensemble models</span>
                  <span className="font-mono tnum">{health?.models ? 'LOADED' : 'MISSING'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">watsonx.ai (Granite)</span>
                  <span className="font-mono tnum">{graniteLabel(health)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Edge stream</span>
                  <span className="font-mono tnum">{liveFrames.length > 0 ? `${liveFrames.length} frames` : 'idle'}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Gauge className="h-4 w-4 text-primary" />
                  HEALTH INDEX
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-1 flex justify-between font-mono text-[10px] text-muted-foreground tnum">
                  <span>DEGRADED</span>
                  <span>NOMINAL</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className={`h-full rounded-full transition-all ${
                      telemetryTone === 'bad'
                        ? 'bg-red-500'
                        : telemetryTone === 'warn'
                          ? 'bg-amber-400'
                          : 'bg-emerald-400'
                    }`}
                    style={{
                      width: `${Math.max(2, Math.min(98, 50 - (tdata?.anomaly_score ?? 0) * 40))}%`,
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* time scrubber */}
        <Card className="mb-5">
          <CardContent className="pt-4">
            <div className="flex items-center gap-4">
              <span className="font-mono text-sm text-primary tnum">{fmtTime(t)}</span>
              <input
                type="range"
                min={0}
                max={maxT}
                step={1}
                value={t}
                onChange={(e) => setT(Number(e.target.value))}
                className="w-full"
                style={{ ['--fill' as string]: `${(t / maxT) * 100}%` }}
                aria-label="Mission time scrubber"
              />
              <span className="font-mono text-xs text-muted-foreground tnum">{fmtTime(maxT)}</span>
            </div>
            <div className="mt-2 flex gap-2">
              {faultWindows.map((w, i) => (
                <Button key={i} variant="outline" size="sm" className="font-mono text-[10px]" onClick={() => setT(w.start)}>
                  FAULT #{i + 1} @ {fmtTime(w.start)}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* tabs */}
        <Tabs defaultValue="telemetry" className="flex-col">
          <TabsList className="flex-wrap max-w-full min-w-0">
            <TabsTrigger value="telemetry">
              <Activity className="mr-1.5 h-3.5 w-3.5" /> Mission
            </TabsTrigger>
            <TabsTrigger value="live">
              <Radio className="mr-1.5 h-3.5 w-3.5" /> Live Ingest
            </TabsTrigger>
            <TabsTrigger value="models">
              <Cpu className="mr-1.5 h-3.5 w-3.5" /> Model Diagnostics
            </TabsTrigger>
            <TabsTrigger value="trace">
              <Activity className="mr-1.5 h-3.5 w-3.5" /> Code Trace
            </TabsTrigger>
          </TabsList>

          <TabsContent value="telemetry" className="mt-3">
            <div className="grid gap-3 md:grid-cols-2">
              {[
                { title: 'BUS VOLTAGE', data: series.volt, color: '#c084fc', unit: 'V' },
                { title: 'HEAT LOAD', data: series.heat, color: '#fb923c', unit: 'W' },
              ].map((c) => (
                <Card key={c.title}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs">{c.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <LineChart data={c.data} windowStart={Math.max(0, t - WINDOW)} windowEnd={t} current={t} color={c.color} unit={c.unit} />
                  </CardContent>
                </Card>
              ))}
              <Card className="md:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-xs">FAULT EPISODES — MISSION PROFILE</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="relative h-4 w-full rounded bg-muted">
                    <div
                      className="absolute inset-y-0 left-0 rounded-l bg-primary/60"
                      style={{ width: `${(maxT / maxT) * 100}%` }}
                    />
                    {faultWindows.map((w, i) => (
                      <div
                        key={i}
                        className="absolute inset-y-0 bg-red-500/70"
                        style={{
                          left: `${(w.start / maxT) * 100}%`,
                          width: `${Math.max(0.5, ((w.end - w.start) / maxT) * 100)}%`,
                        }}
                      />
                    ))}
                    <div
                      className="absolute top-[-3px] h-[22px] w-0.5 bg-white/80"
                      style={{ left: `${(t / maxT) * 100}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground tnum">
                    <span>T+00:00</span>
                    <span>RED = anomaly episodes · cursor = current time</span>
                    <span>{fmtTime(maxT)}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="live" className="mt-3">
            <div className="grid gap-3 lg:grid-cols-3">
              <Card className="lg:col-span-2">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <div>
                    <CardTitle className="text-sm">VIRTUAL EDGE NODE STREAM</CardTitle>
                    <CardDescription className="font-mono text-[10px]">
                      ESP32-class node · 12-bit ADC · 2% packet dropout · scored through production ensemble
                    </CardDescription>
                  </div>
                  <div className="flex min-w-0 flex-wrap gap-2">
                    <Button size="sm" variant="outline" onClick={advanceLive}>
                      <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> ADVANCE 30
                    </Button>
                    <Button size="sm" variant={liveRunning ? 'destructive' : 'default'} onClick={() => setLiveRunning(!liveRunning)}>
                      {liveRunning ? <Pause className="mr-1.5 h-3.5 w-3.5" /> : <Play className="mr-1.5 h-3.5 w-3.5" />}
                      {liveRunning ? 'STOP' : 'AUTO'}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {liveScore && (
                    <div className="mb-3 flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3">
                      <Badge variant={liveScore.flag === 1 ? 'destructive' : 'outline'} className="font-mono">
                        {liveScore.flag === 1 ? 'ANOMALY' : 'NOMINAL'}
                      </Badge>
                      <span className="font-mono text-sm tnum">
                        score {fmt(liveScore.score, 3)}
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        source {liveScore.source === 1 ? 'POWER' : liveScore.source === 2 ? 'THERMAL' : 'FULL'}
                      </span>
                      {liveScore.flag === 1 && liveFrames.length > 0 && (
                        <span className="font-mono text-[10px] text-muted-foreground tnum">
                          detected @ {fmtTime(liveFrames[liveFrames.length - 1].time_s)}
                        </span>
                      )}
                    </div>
                  )}
                  <div className="h-56 overflow-auto rounded border border-border">
                    <table className="w-full font-mono text-[11px] tnum">
                      <thead className="sticky top-0 bg-muted text-left text-[10px] uppercase text-muted-foreground">
                        <tr>
                          <th className="px-3 py-2">T</th>
                          <th className="px-3 py-2">SOLAR W</th>
                          <th className="px-3 py-2">SOC %</th>
                          <th className="px-3 py-2">BUS V</th>
                          <th className="px-3 py-2">TEMP °C</th>
                          <th className="px-3 py-2">HEAT IN/OUT W</th>
                        </tr>
                      </thead>
                      <tbody>
                        {liveFrames.length === 0 && (
                          <tr>
                            <td colSpan={6} className="px-3 py-6 text-center text-muted-foreground">
                              No frames yet — press ADVANCE or AUTO to stream live telemetry.
                            </td>
                          </tr>
                        )}
                        {liveFrames.map((f, i) => (
                          <tr key={i} className="border-t border-border/40">
                            <td className="px-3 py-1.5 text-primary">{fmtTime(f.time_s)}</td>
                            <td className="px-3 py-1.5">{fmt(f.solar_power_w)}</td>
                            <td className="px-3 py-1.5">{fmt(f.battery_soc * 100, 1)}</td>
                            <td className="px-3 py-1.5">{fmt(f.battery_voltage_v, 2)}</td>
                            <td className="px-3 py-1.5">{fmt(f.temperature_c, 1)}</td>
                            <td className="px-3 py-1.5">
                              {fmt(f.heat_in_w)} / {fmt(f.heat_out_w)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">LIVE STREAM TELEMETRY</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  <LineChart
                    data={liveFrames.map((f) => ({ t: f.time_s, v: f.solar_power_w }))}
                    windowStart={liveFrames.length ? liveFrames[0].time_s : 0}
                    windowEnd={liveFrames.length ? liveFrames[liveFrames.length - 1].time_s : 1}
                    current={liveFrames.length ? liveFrames[liveFrames.length - 1].time_s : 0}
                    color="#5ac8fa"
                    min={0}
                    unit="W"
                  />
                  <LineChart
                    data={liveFrames.map((f) => ({ t: f.time_s, v: f.temperature_c }))}
                    windowStart={liveFrames.length ? liveFrames[0].time_s : 0}
                    windowEnd={liveFrames.length ? liveFrames[liveFrames.length - 1].time_s : 1}
                    current={liveFrames.length ? liveFrames[liveFrames.length - 1].time_s : 0}
                    color="#fb923c"
                    unit="°C"
                  />
                  <div className="rounded-lg border border-border bg-muted/30 p-3 text-[11px] leading-relaxed text-muted-foreground">
                    Same ensemble (Isolation Forests over full/power/thermal) that scores the recorded
                    scenarios — a live feed from a real ESP32/Raspberry Pi edge node would replace the
                    virtual node with zero changes to this view.
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trace" className="mt-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div>
                  <CardTitle className="text-sm">RUNTIME CODE TRACE</CardTitle>
                  <CardDescription className="font-mono text-[10px]">
                    every event recorded by missionmind.trace — which pipeline code actually
                    executes as telemetry flows (scoring, physics rules, RAG, narrative)
                  </CardDescription>
                </div>
                <div className="flex min-w-0 flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => refreshTrace(false)}>
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> REFRESH
                  </Button>
                  <Button
                    size="sm"
                    variant={traceLive ? 'destructive' : 'default'}
                    onClick={() => {
                      if (!traceLive) {
                        refreshTrace(false)
                        setTraceLive(true)
                      } else {
                        setTraceLive(false)
                      }
                    }}
                  >
                    <Radio className="mr-1.5 h-3.5 w-3.5" />
                    {traceLive ? 'LIVE ON' : 'LIVE OFF'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {traceError && <div className="mb-2 text-xs text-red-300">{traceError}</div>}
                <div className="flex items-center gap-2 pb-2 font-mono text-[10px] text-muted-foreground">
                  <span
                    className={`inline-block h-1.5 w-1.5 rounded-full ${traceLive ? 'animate-pulse bg-red-400' : 'bg-muted-foreground/50'}`}
                  />
                  {traceLive ? 'live — polling /api/trace every 2 s' : 'paused — press LIVE ON or REFRESH'}
                  <span className="ml-auto tnum">{traceEvents.length} events buffered</span>
                </div>
                <div className="h-[420px] overflow-auto rounded border border-border bg-black/40 p-3">
                  {traceEvents.length === 0 ? (
                    <div className="py-6 text-center font-mono text-xs text-muted-foreground">
                      No trace events yet — advance the live stream, scrub the mission clock, or
                      open an alert to see real code execution.
                    </div>
                  ) : (
                    <pre className="font-mono text-[11px] leading-relaxed tnum">
                      {traceEvents
                        .map((e) => {
                          const t =
                            e.mission_t === null || e.mission_t === undefined
                              ? '      '
                              : fmtTime(e.mission_t)
                          const val = e.value === null || e.value === undefined ? '' : ` = ${e.value}`
                          const note = e.note ? `  # ${e.note}` : ''
                          return `[${e.seq}] ${t}  ${e.module}.${e.func}${val}${note}`
                        })
                        .join('\n')}
                    </pre>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="models" className="mt-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">MODEL ZOO — SELF-TEST</CardTitle>
                <CardDescription className="text-[11px]">
                  Supervised classifiers trained on a labelled normal+anomaly mix; unsupervised
                  detectors trained on normal data only. Metrics from the same protocol as
                  <code className="mx-1 font-mono text-[10px]">advanced_models.py</code>.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!models ? (
                  <div className="py-4 text-sm text-muted-foreground">Loading…</div>
                ) : (
                  <div className="overflow-auto">
                    <table className="w-full text-left font-mono text-[11px] tnum">
                      <thead className="text-[10px] uppercase text-muted-foreground">
                        <tr>
                          <th className="px-2 py-2">Model</th>
                          <th className="px-2 py-2">Family</th>
                          <th className="px-2 py-2">Fit</th>
                          <th className="px-2 py-2">TP/10</th>
                          <th className="px-2 py-2">FP</th>
                          {['acc', 'prec', 'rec', 'F1', 'AUC'].map((m) => (
                            <th key={m} className="px-2 py-2">{m}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {models.map((m, i) => (
                          <tr key={i} className="border-t border-border/40">
                            <td className="px-2 py-1.5">{m.name}</td>
                            <td className="px-2 py-1.5">
                              <Badge variant={m.family === 'supervised' ? 'default' : 'secondary'} className="text-[9px]">
                                {m.family}
                              </Badge>
                            </td>
                            <td className="px-2 py-1.5">{m.fit}</td>
                            <td className="px-2 py-1.5">{m.tp ?? '—'}</td>
                            <td className="px-2 py-1.5">{m.fp ?? '—'}</td>
                            {['acc', 'precision', 'recall', 'f1', 'auc'].map((k) => (
                              <td key={k} className="px-2 py-1.5">
                                {m[k] === undefined || m[k] === null ? '—' : Number(m[k]).toFixed(2)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <Separator className="my-3" />
                <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                  <Badge variant="outline" className="font-mono text-[10px]">watsonx Granite: {graniteLabel(health)}</Badge>
                  <Badge variant="outline" className="font-mono text-[10px]">models on disk: {health?.models ? 'YES' : 'NO'}</Badge>
                  <Badge variant="outline" className="font-mono text-[10px]">API: {String(health?.status ?? '…')}</Badge>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t border-border/60 py-3 text-center font-mono text-[10px] text-muted-foreground tnum">
        MISSIONMIND · physics-informed spacecraft health &amp; reliability · all telemetry from the
        coupled ODE simulator, scored by the production ensemble
      </footer>
    </div>
  )
}
