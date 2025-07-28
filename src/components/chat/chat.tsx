'use client';

import React, { useEffect, useState, useRef } from 'react';
import { exchange, createChat } from '@/lib/aiChat';
import MessageDisplayer from '@/components/chat/messageDisplayer';
import styles from './Chat.module.css';
import { usePathname } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth/authentication';
import { Trans, useLingui } from '@lingui/react/macro';

export default function ChatElement() {
  const [question, setQuestion] = useState<string>('');
  const [messages, setMessages] = useState<exchange[]>([]);
  const [status, setStatus] = useState<string>('');
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  const { t } = useLingui();

  const isStreamingRef = useRef(false); // Tracks if we're mid-stream

  async function askQuestion() {
    if (!question.trim()) return;

    setStatus(t`thinking...`);
    const auxQuestion = question;
    setQuestion('');
    setMessages((prev) => [...prev, { question: auxQuestion, answer: '' }]);
    const uId = await getCurrentUser();

    const payload = {
      question: auxQuestion,
      sessionId: uId,
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
      setStatus(t`Failed to connect to AI.`);
    }
  }

  async function initChat() {
    const uId = await getCurrentUser();
    const rezult = await createChat(uId);
    //console.log('Chat history:', rezult);

    if (rezult && Array.isArray(rezult)) {
      for (let i = 0; i < rezult.length; i += 2) {
        const question = rezult[i]?.parts?.[0]?.text || '';
        const answer = rezult[i + 1]?.parts?.[0]?.text || '';
        const newExchange: exchange = { question, answer };
        setMessages((prev) => [...prev, newExchange]);
      }
    }
  }

  useEffect(() => {
    initChat();
  }, []);

  if (
    pathname == '/' ||
    pathname == '/login' ||
    pathname == '/register' ||
    pathname == '/handleMail' ||
    pathname == '/dash' ||
    pathname == '/profile' ||
    pathname.includes('/docs')
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
        {open ? <Trans>Close</Trans> : <Trans>AI Chat</Trans>}
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
            placeholder={t`Type here...`}
            onChange={(e) => setQuestion(e.target.value)}
            value={question}
          />

          <button type={'submit'}>
            <Trans>Send</Trans> <i className='fa-solid fa-paper-plane-top'></i>
          </button>
        </form>
      </div>
    </div>
  );
}
