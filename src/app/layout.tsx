import type { Metadata } from 'next';
import {
  Geist,
  Geist_Mono,
  Special_Gothic_Expanded_One
} from 'next/font/google';
import '@/lib/fontawesome/css/fa.css';
import './globals.css';
import SideBar from '@/components/layout/sidebar';
import Chat from '@/components/chat/chat';
import Provider from '@/lib/providers/providers';
import { allMessages } from '@/lib/frontend/i18n';
import { getCurrentLocale } from '@/lib/frontend/languageChanger';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin']
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin']
});

const specialGothicExpandedOne = Special_Gothic_Expanded_One({
  variable: '--font-special-gothic-expanded-one',
  subsets: ['latin'],
  weight: '400'
});

export const metadata: Metadata = {
  title: '2N2D',
  description: 'Neural Network Development Dashboard'
};

export default async function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  const lang = await getCurrentLocale();
  return (
    <html
      lang={lang}
      className={`dark ${geistSans.variable} ${geistMono.variable} ${specialGothicExpandedOne.variable}`}
    >
      <head></head>
      <body className='antialiased'>
        <Provider lang={lang} messages={allMessages[lang]}>
          <SideBar />
          <Chat />
          {children}
          <footer>
            <p>Neural Network Development Dashboard</p>
          </footer>
        </Provider>
      </body>
    </html>
  );
}
