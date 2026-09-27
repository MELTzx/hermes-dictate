// hermes-dictate — local-first live dictation for the Hermes desktop composer.
// Requires the sidecar: python3 sidecar.py (hermes-dictate/sidecar.py), serving
// streaming parakeet-unified-en ASR + filler cleanup on ws://127.0.0.1:8765.
//
// UX: toggle on → speech appears LIVE in the composer as a draft (it never
// sends). Each utterance (pause ≥ ~0.7s) is finalized: the raw partial is
// replaced in place by the cleaned sentence. Toggling off stops the mic;
// whatever is in the box stays for you to edit and send yourself.

import {
  Codicon,
  GlyphSpinner,
  host,
  KEYBINDS_AREA,
  COMPOSER_AREAS,
  atom,
  useValue,
} from '@hermes/plugin-sdk'
import { jsx } from 'react/jsx-runtime'

const SIDECAR = 'ws://127.0.0.1:8765'

const $running = atom(false)
const $partial = atom('')
const $status = atom('idle') // idle | connecting | listening | error
const $err = atom('')        // last error detail
const $level = atom(0)       // mic peak level 0..1 while listening
const $frames = atom(0)      // worklet frames sent (audio actually flowing)

let ws = null
let audioCtx = null
let proc = null
let stream = null
let levelTimer = null

// ── live-draft state ──────────────────────────────────────────────────
// baseText: composer content committed before dictation started (or from
// earlier finalized utterances). tail: the not-yet-final words currently
// being spoken. Draft = base + ' ' + tail, rendered into the composer on
// every partial, replaced by the cleaned sentence on final.
// USER EDITS WIN: before each render we re-read the live draft; if it differs
// from what we last wrote (user deleted/typed), the user's version becomes the
// new baseText and we append to THAT — deleted words are never resurrected.
let baseText = ''
let tail = ''
let lastWritten = ''    // the exact draft string WE last set (null-ish until first write)
let dictSid = null   // session id captured at dictation start; NEVER re-read mid-dictation

function sid() {
  return host.state?.focusedSessionId?.get?.() ?? null
}

let emptySightings = 0    // consecutive empty-draft reads (glitch vs real wipe)

async function reconcile() {
  // Runs on every partial (~300ms). User edits are adopted IMMEDIATELY on a
  // single differing read — no confirmation delay, so nothing typed between
  // partials is ever lost. Only two reads are never trusted as edits:
  //   - null/undefined: no composer surface answered (focus flicker, debounce)
  //   - empty string: could be a surface glitch wiping a paragraph; a real
  //     select-all-delete is confirmed on the very next 300ms cycle anyway.
  try {
    const live = await host.composer.getDraft(dictSid)
    if (live === null || live === undefined) return      // read failed: keep our state
    if (live === lastWritten) { emptySightings = 0; return }
    if (live === '') {                                    // suspicious empty: confirm once
      emptySightings = (emptySightings || 0) + 1
      if (emptySightings < 2) return
      emptySightings = 0
      baseText = ''
      tail = ''
      return
    }
    emptySightings = 0
    let userBase = live
    if (tail && userBase.endsWith(tail)) {
      userBase = userBase.slice(0, -tail.length).replace(/\s+$/, '')
    }
    baseText = userBase.trim()
  } catch { /* read failed: keep our state */ }
}

async function renderDraft() {
  try {
    await reconcile()
    const draft = tail ? (baseText ? baseText + ' ' : '') + tail : baseText
    await host.composer.setDraft(dictSid, draft)
    lastWritten = draft
  } catch (e) {
    console.error('[hermes-dictate] setDraft failed:', e)
  }
}

function resetDraftState() {
  baseText = ''
  tail = ''
}

function stop() {
  try { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'stop' })) } catch { /* closing */ }
  try { ws?.close() } catch { /* closing */ }
  try { stream?.getTracks().forEach((t) => t.stop()) } catch { /* stopped */ }
  try { proc?.disconnect() } catch { /* disconnected */ }
  try { audioCtx?.close() } catch { /* closed */ }
  if (levelTimer) { clearInterval(levelTimer); levelTimer = null }
  ws = null; proc = null; stream = null; audioCtx = null
  $running.set(false)
  $status.set('idle')
  $partial.set('')
}

function errText(e) {
  const name = e?.name || ''
  if (name === 'NotAllowedError') return 'mic permission denied — check OS/app mic settings'
  if (name === 'NotFoundError') return 'no microphone found'
  if (name === 'NotReadableError') return 'mic in use by another app'
  return (name ? name + (e?.message ? ' — ' + e.message : '') : String(e?.message || e || 'unknown')).slice(0, 120)
}

async function ensureSidecar(ctx) {
  // The sidecar MUST run on the machine with the mic. When the desktop's
  // backend is remote (remote gateway/dashboard), ctx.rest reaches the REMOTE
  // machine — spawning there would decode nothing. So: try a direct local WS
  // connection FIRST; only when nothing listens locally do we ask the backend
  // supervisor (/api/plugins/hermes-dictate/) to spawn one — correct for the
  // local-backend topology (ctx.rest hits localhost then).
  try {
    const wsProbe = new WebSocket(SIDECAR)
    const alive = await new Promise((res) => {
      const t = setTimeout(() => { try { wsProbe.close() } catch {} ; res(false) }, 700)
      wsProbe.onopen = () => { clearTimeout(t); try { wsProbe.close() } catch {}; res(true) }
      wsProbe.onerror = () => { clearTimeout(t); res(false) }
    })
    if (alive) return { ok: true, direct: true, already: true }
  } catch { /* probe failed — fall through to supervisor */ }
  try {
    const r = await ctx.rest('/status')
    if (r?.running) return { ok: true, already: true }
  } catch { /* no backend plugin (older/remote backend) */ }
  const t0 = Date.now()
  while (Date.now() - t0 < 150_000) {
    try {
      const r = await ctx.rest('/start', { method: 'POST', body: {}, timeoutMs: 130_000 })
      if (r?.ok) return { ok: true, already: !!r.already_running }
      if (r?.error && !/did not listen/.test(r.error)) return { ok: false, error: r.error + (r.log_tail ? ` — ${r.log_tail}` : '') }
    } catch (e) {
      return { ok: false, error: 'no local sidecar and backend cannot start one (remote backend?). Run: systemctl --user enable --now hermes-dictate' }
    }
    await new Promise((res) => setTimeout(res, 2000))
  }
  return { ok: false, error: 'sidecar start timed out (model load too slow?)' }
}

async function start(ctx) {
  $status.set('connecting')
  $err.set('')
  $frames.set(0)
  $level.set(0)
  try {
    // 0. auto-start the sidecar (local probe first; supervisor fallback)
    const boot = await ensureSidecar(ctx)
    if (!boot.ok) throw new Error(boot.error || 'sidecar failed to start')

    // 1. sidecar WS — fail fast, with a readable reason
    await new Promise((res, rej) => {
      ws = new WebSocket(SIDECAR)
      ws.binaryType = 'arraybuffer'
      let settled = false
      let opened = false
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; rej(new Error(`sidecar connect timeout — ${SIDECAR}`)) }
      }, 30000)   // systemd cold start / model load can take a moment
      ws.onopen = () => { opened = true; if (!settled) { settled = true; clearTimeout(timeout); res() } }
      ws.onerror = () => { /* onclose follows with the verdict */ }
      // provisional guard: a death during mic setup must not go unnoticed (#6)
      ws.onclose = () => {
        if (!settled) { settled = true; clearTimeout(timeout); rej(new Error(`sidecar unreachable at ${SIDECAR} — backend supervisor did not start it (is ~/.hermes/plugins/hermes-dictate/dashboard/ installed?)`)) }
        else if (opened) { stop(); $status.set('error'); $err.set('connection to sidecar lost during setup') }
      }
    })
    // pin the session for the whole dictation (#4): later focus moves must not redirect drafts
    dictSid = sid()
    // 2. snapshot the existing draft so we append, not clobber
    try {
      baseText = ((await host.composer.getDraft(sid())) ?? '').trim()
    } catch {
      baseText = ''
    }
    tail = ''
    // 3. mic + audio graph
    audioCtx = new AudioContext({ sampleRate: 16000 })
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    const src = audioCtx.createMediaStreamSource(stream)
    const workletUrl = URL.createObjectURL(new Blob([`
      class P extends AudioWorkletProcessor {
        constructor() { super(); this.buf = [] }
        process(inputs) {
          const ch = inputs[0] && inputs[0][0]
          if (ch) {
            for (let i = 0; i < ch.length; i++) this.buf.push(ch[i])
            while (this.buf.length >= 1280) this.port.postMessage(this.buf.splice(0, 1280))
          }
          return true
        }
      }
      registerProcessor('pcm', P)
    `], { type: 'application/javascript' }))
    await audioCtx.audioWorklet.addModule(workletUrl)
    proc = new AudioWorkletNode(audioCtx, 'pcm', { numberOfOutputs: 1, channelCount: 1 })
    const meter = audioCtx.createAnalyser()
    meter.fftSize = 512
    src.connect(meter)
    src.connect(proc)
    const mute = audioCtx.createGain()
    mute.gain.value = 0
    proc.connect(mute)
    mute.connect(audioCtx.destination)
    const levelBuf = new Float32Array(meter.fftSize)
    levelTimer = setInterval(() => {
      meter.getFloatTimeDomainData(levelBuf)
      let peak = 0
      for (let i = 0; i < levelBuf.length; i++) peak = Math.max(peak, Math.abs(levelBuf[i]))
      $level.set(peak)
    }, 150)
    proc.port.onmessage = (e) => {
      if (ws?.readyState === 1) {
        ws.send(new Float32Array(e.data).buffer)
        $frames.set($frames.get() + 1)   // count only frames actually sent (#5)
      }
    }
    // 4. consume transcripts → live draft in composer (NEVER sends)
    ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data)
        if (m.type === 'partial') {
          const p = (m.text || '').trim()
          if (p) {
            tail = p
            $partial.set(p)
            void renderDraft()
          }
        } else if (m.type === 'status' && m.error) {
          console.error('[hermes-dictate] sidecar status:', m.error)
        } else if (m.type === 'final') {
          const clean = (m.clean || m.text || '').trim()
          void (async () => {
            await reconcile()          // pick up any user edits BEFORE committing
            if (clean) {
              // if the user deleted this utterance's words mid-flight, don't resurrect:
              // only commit when the tail we were showing is (still) present
              const stillThere = !tail || baseText.endsWith(tail) || lastWritten.includes(tail)
              if (stillThere) {
                baseText = baseText ? baseText + ' ' + clean : clean
              }
            }
            tail = ''
            $partial.set('')
            await renderDraft()
          })()
        }
      } catch { /* malformed frame */ }
    }
    ws.onclose = () => {
      if ($running.get()) {
        stop()
        $status.set('error')
        $err.set('connection to sidecar lost')
      }
    }
    $running.set(true)
    $status.set('listening')
    host.composer.focus?.(sid())
  } catch (e) {
    console.error('[hermes-dictate] start failed:', e)
    stop()
    $status.set('error')
    $err.set(errText(e))
  }
}

function toggle(ctx) {
  if ($running.get()) stop()
  else void start(ctx)
}

export default {
  id: 'hermes-dictate',
  name: 'Dictate',
  register(ctx) {
    ctx.register({
      id: 'hermes-dictate.toggle',
      area: KEYBINDS_AREA,
      data: {
        id: 'hermes-dictate.toggle',
        label: 'Toggle dictation',
        category: 'dictate',
        defaults: ['mod+shift+d'],
        run: () => toggle(ctx),
      },
    })

    ctx.register({
      id: 'hermes-dictate.chip',
      area: COMPOSER_AREAS.underside,
      order: 50,
      render: () => {
        const running = useValue($running)
        const status = useValue($status)
        const partial = useValue($partial)
        const level = useValue($level)
        const frames = useValue($frames)
        const err = useValue($err)
        let label
        if (status === 'error') label = err || 'error'
        else if (!running) label = 'Dictate'
        else if (partial) label = partial
        else if (frames === 0) label = 'connected — no audio frames (mic?)'
        else label = `listening · ${(level * 100).toFixed(0)}%`
        return jsx('button', {
          onClick: () => toggle(ctx),
          title: `Dictate (mod+shift+D) — live draft in composer, never sends\nframes: ${frames} level: ${(level * 100).toFixed(0)}%${err ? '\nerror: ' + err : ''}`,
          style: {
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            padding: '2px 8px', fontSize: '11px', cursor: 'pointer',
            color: running ? 'var(--ui-accent)' : 'var(--ui-text-secondary)',
            background: 'transparent', border: 'none',
            maxWidth: '420px', overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis',
          },
          children: [
            jsx(Codicon, { name: running ? 'pulse' : 'mic' }),
            jsx('span', { children: label }),
            status === 'connecting' ? jsx(GlyphSpinner, {}) : null,
            status === 'error' ? jsx(Codicon, { name: 'error' }) : null,
          ],
        })
      },
    })
  },
}
