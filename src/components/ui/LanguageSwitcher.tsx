'use client';

import { useEffect, useState } from 'react';
import { getCurrentLocale, setNewLocale } from '@/lib/frontend/languageChanger';

const languages = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'ar', name: 'العربية', flag: '🇸🇦' },
  { code: 'bn', name: 'বাংলা', flag: '🇧🇩' },
  { code: 'cs', name: 'Čeština', flag: '🇨🇿' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'fa', name: 'فارسی', flag: '🇮🇷' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'he', name: 'עברית', flag: '🇮🇱' },
  { code: 'hi', name: 'हिन्दी', flag: '🇮🇳' },
  { code: 'hu', name: 'Magyar', flag: '🇭🇺' },
  { code: 'id', name: 'Bahasa Indonesia', flag: '🇮🇩' },
  { code: 'it', name: 'Italiano', flag: '🇮🇹' },
  { code: 'ja', name: '日本語', flag: '🇯🇵' },
  { code: 'ko', name: '한국어', flag: '🇰🇷' },
  { code: 'ms', name: 'Bahasa Melayu', flag: '🇲🇾' },
  { code: 'pa', name: 'ਪੰਜਾਬੀ', flag: '🇮🇳' },
  { code: 'pl', name: 'Polski', flag: '🇵🇱' },
  { code: 'pt', name: 'Português', flag: '🇵🇹' },
  { code: 'pt-BR', name: 'Português (Brasil)', flag: '🇧🇷' },
  { code: 'ro', name: 'Română', flag: '🇷🇴' },
  { code: 'ru', name: 'Русский', flag: '🇷🇺' },
  { code: 'th', name: 'ไทย', flag: '🇹🇭' },
  { code: 'tl', name: 'Tagalog', flag: '🇵🇭' },
  { code: 'tr', name: 'Türkçe', flag: '🇹🇷' },
  { code: 'uk', name: 'Українська', flag: '🇺🇦' },
  { code: 'ur', name: 'اردو', flag: '🇵🇰' },
  { code: 'vi', name: 'Tiếng Việt', flag: '🇻🇳' },
  { code: 'zh', name: '中文 (简体)', flag: '🇨🇳' },
  { code: 'zh-TW', name: '中文 (繁體)', flag: '🇹🇼' }
];

export default function LanguageSwitcher() {
  const [currentLocale, setCurrentLocale] = useState('en');

  async function changeLanguage(locale: string) {
    setCurrentLocale(locale);
    await setNewLocale(locale);

    window.location.reload();
  }

  useEffect(() => {
    async function fetchLocale() {
      const locale = await getCurrentLocale();
      setCurrentLocale(locale);
    }
    fetchLocale();
  }, []);

  return (
    <div className='language-switcher'>
      <select
        value={currentLocale}
        onChange={(e) => changeLanguage(e.target.value)}
        className='language-select'
      >
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
}
