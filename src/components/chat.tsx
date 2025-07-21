'use client';

import React, { useEffect, useState, useRef } from 'react';
import { exchange, createChat } from '@/lib/aiChat';
import MessageDisplayer from '@/components/misc/messageDisplayer';
import styles from './Chat.module.css';
import { usePathname } from 'next/navigation';
import { getSessionTokenHash } from '@/lib/auth/authentication';

export default function ChatElement() {
  const [question, setQuestion] = useState<string>('');
  const [messages, setMessages] = useState<exchange[]>([]);
  const [status, setStatus] = useState<string>('');
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  const isStreamingRef = useRef(false); // Tracks if we're mid-stream

  async function askQuestion() {
    if (!question.trim()) return;

    setStatus('thinking...');
    const auxQuestion = question;
    setQuestion('');
    setMessages((prev) => [...prev, { question: auxQuestion, answer: '' }]);
    const sessionId = await getSessionTokenHash();

    const payload = {
      question: auxQuestion,
      sessionId,
      modelData:
        pathname !== '/learn' && pathname !== '/docs'
          ? sessionStorage.getItem('modelResponse')
          : null,
      csvData:
        pathname !== '/learn' && pathname !== '/docs'
          ? sessionStorage.getItem('csvData')
          : null,
      context:
        pathname === '/learn' || pathname === '/docs'
          ? sessionStorage.getItem('screenContext')
          : null
    };

    // Add empty message first
    isStreamingRef.current = true;

    const response = await fetch('/api/ask', {
      method: 'POST',
      body: JSON.stringify(payload)
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    if (reader) {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        for (const char of buffer) {
          await new Promise((res) => setTimeout(res, 10));
          setMessages((prevMessages) => {
            const newMessages = [...prevMessages];
            const last = newMessages[newMessages.length - 1];
            newMessages[newMessages.length - 1] = {
              ...last,
              answer: (last.answer || '') + char
            };
            return newMessages;
          });
        }
        buffer = '';
      }

      isStreamingRef.current = false;
      setStatus('');
    } else {
      setStatus('Failed to connect to AI.');
    }
  }

  async function initChat() {
    const sessionId = await getSessionTokenHash();
    await createChat(sessionId);
  }

  useEffect(() => {
    initChat();
  }, []);

  if (
    pathname == '/' ||
    pathname == '/login' ||
    pathname == '/signup' ||
    pathname == '/handleMail' ||
    pathname == '/dash' ||
    pathname == '/profile'
  )
    return <></>;
  return (
    <div
      className={styles.chatContainer}
      style={{ bottom: open ? '0' : '-50vh' }}
    >
      <button
        className={styles.toggle}
        onClick={() => {
          setOpen(!open);
        }}
      >
        {open ? 'Close' : 'AI Chat'}
      </button>
      <div className={styles.chatArea}>
        <MessageDisplayer messages={messages} />
        <div>
          {status && status != '' ? (
            <h1 className={styles.status}>{status}</h1>
          ) : (
            ''
          )}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            askQuestion();
          }}
        >
          <input
            type={'text'}
            placeholder={'Type here...'}
            onChange={(e) => setQuestion(e.target.value)}
            value={question}
          />

          <button type={'submit'}>
            Send <i className='fa-solid fa-paper-plane-top'></i>
          </button>
        </form>
      </div>
    </div>
  );
}
