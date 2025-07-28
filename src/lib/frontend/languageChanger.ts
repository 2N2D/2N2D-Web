'use server';

import { cookies } from 'next/headers';
import linguiConfig from '../../../lingui.config';

const { locales } = linguiConfig;

export async function getCurrentLocale(): Promise<string> {
  const cookieStore = await cookies();
  const lang = cookieStore.get('lang')?.value;
  return lang || 'en';
}

export async function setNewLocale(locale: string): Promise<void> {
  if (!locales.includes(locale)) {
    throw new Error(`Locale "${locale}" is not supported.`);
  }
  const cookieStore = await cookies();
  cookieStore.set('lang', locale, { expires: new Date('9999-12-31') });
}
