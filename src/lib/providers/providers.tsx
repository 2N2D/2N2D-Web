'use client';

import { LinguiClientProvider } from '@/lib/providers/lingui-client-provider';
import { Messages } from '@lingui/core';
import { ReactNode } from 'react';

export default function Providers({
  lang,
  messages,
  children
}: {
  lang: string;
  messages: Messages;
  children: ReactNode;
}) {
  return (
    <LinguiClientProvider initialLocale={lang} initialMessages={messages}>
      {children}
    </LinguiClientProvider>
  );
}
