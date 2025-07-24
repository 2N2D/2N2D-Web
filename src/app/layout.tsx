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

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang='en'
      className={`dark ${geistSans.variable} ${geistMono.variable} ${specialGothicExpandedOne.variable}`}
    >
      <head></head>
      <body className='antialiased'>
        <SideBar />
        <Chat />
        <div>{children}</div>
        <footer>
          <p>Neural Network Development Dashboard</p>
        </footer>
      </body>
    </html>
  );
}
