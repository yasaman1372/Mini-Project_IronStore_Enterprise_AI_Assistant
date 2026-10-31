import { useEffect, useRef, useState } from 'react'
import './index.css'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8765').replace(/\/$/, '')

const SUGGESTIONS = [
  'How long does a card refund usually take?',
  'What equipment does IronStore provide for remote work?',
  'How far in advance must I request annual leave longer than five working days?',
]

const CITATION = /[[(]\s*(?:Sources?:\s*)?([^[\]()]*?\.pdf[^[\]()]*)[\])]/gi
const PART = /([\w.\- ]+?)\.pdf(?:\s*,?\s*p(?:ages?|\.)?\s*(\d+))?/gi

function renderText(text) {
  const out = []
  let last = 0
  for (const m of text.matchAll(CITATION)) {
    out.push(text.slice(last, m.index).trimEnd())
    out.push(
      <span className="sources" key={m.index}>
        {[...m[1].matchAll(PART)].map((p, i) => (
          <span className="chip" key={i} title={`${p[1].trim()}.pdf`}>
            📄 {p[1].trim().replace(/^IronStore_/, '').replace(/_/g, ' ')}
            {p[2] && <b>p. {p[2]}</b>}
          </span>
        ))}
      </span>,
    )
    last = m.index + m[0].length
  }
  out.push(text.slice(last))
  return out
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('sessionId'))
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(text) {
    const question = text.trim()
    if (!question || loading) return
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = await res.json()
      setSessionId(data.session_id)
      localStorage.setItem('sessionId', data.session_id)
      setMessages((m) => [...m, { role: 'assistant', text: data.answer }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', text: `Sorry, something went wrong. ${err.message}`, error: true }])
    } finally {
      setLoading(false)
    }
  }

  function newChat() {
    if (sessionId) {
      fetch(`${API_URL}/chat/${sessionId}`, { method: 'DELETE' }).catch(() => {})
    }
    localStorage.removeItem('sessionId')
    setSessionId(null)
    setMessages([])
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <div className="logo">IS</div>
          <div>
            <h1>IronStore Assistant</h1>
            <span className="status"><i /> Online</span>
          </div>
        </div>
        <button onClick={newChat} disabled={loading || messages.length === 0}>New chat</button>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <div className="logo big">IS</div>
            <h2>How can I help you today?</h2>
            <p>Ask about IronStore policies and products.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`row ${m.role}`}>
            {m.role === 'assistant' && <div className="avatar">IS</div>}
            <div className={`bubble ${m.role}${m.error ? ' error' : ''}`}>{m.role === 'assistant' ? renderText(m.text) : m.text}</div>
          </div>
        ))}
        {loading && (
          <div className="row assistant">
            <div className="avatar">IS</div>
            <div className="bubble assistant dots"><span /><span /><span /></div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form onSubmit={(e) => { e.preventDefault(); send(input) }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          maxLength={2000}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()} aria-label="Send">➤</button>
      </form>
    </div>
  )
}
