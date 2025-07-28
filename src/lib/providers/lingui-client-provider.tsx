'use client';

import { type Messages, setupI18n } from '@lingui/core';
import { I18nProvider } from '@lingui/react';
import { ReactNode, useEffect, useState } from 'react';

type Props = {
  children: ReactNode;
  initialLocale: string;
  initialMessages: Messages;
};

export function LinguiClientProvider({
  children,
  initialLocale,
  initialMessages
}: Props) {
  const [i18n] = useState(() => {
    return setupI18n({
      locale: initialLocale,
      messages: { [initialLocale]: initialMessages }
    });
  });

  useEffect(() => {
    i18n.load(initialLocale, initialMessages);
    i18n.activate(initialLocale);
  }, []);

  return <I18nProvider i18n={i18n}>{children}</I18nProvider>;
}
