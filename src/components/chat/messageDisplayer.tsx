'use client';
import React, { useEffect, useRef } from 'react';
import { exchange } from '@/lib/aiChat';
import styles from './messageDisplayer.module.css';
import Message from '@/components/chat/message';

interface props {
  messages?: exchange[];
}

const MessageDisplayer = ({ messages }: props) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTo({
        top: ref.current.scrollHeight,
        left: 0,
        behavior: 'auto'
      });
    }
  }, [messages]);
  return (
    <div className={styles.container} ref={ref}>
      {messages?.map((m, i) => (
        <div key={i}>
          <Message
            icon={'fa-solid fa-user'}
            sender={'You'}
            content={m.question}
          />
          <Message
            icon='fa-solid fa-robot'
            sender={'2N2D Assistant'}
            content={m.answer}
          />
        </div>
      ))}
    </div>
  );
};

export default MessageDisplayer;
