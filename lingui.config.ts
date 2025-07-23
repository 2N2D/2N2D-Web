import { defineConfig } from '@lingui/cli';

export default defineConfig({
  sourceLocale: 'en',
  locales: [
  'en', // English
  'es', // Spanish
  'fr', // French
  'de', // German
  'it', // Italian
  'pt', // Portuguese
  'pt-BR', // Brazilian Portuguese
  'ro', // Romanian 
  'pl', // Polish 
  'hu', // Hungarian 
  'cs', // Czech 
  'uk', // Ukrainian 
  'ru', // Russian 
  'ar', // Arabic (RTL) 
  'fa', // Persian/Farsi (RTL) 
  'he', // Hebrew (RTL) 
  'tr', // Turkish 
  'hi', // Hindi 
  'bn', // Bengali 
  'ur', // Urdu (RTL) 
  'pa', // Punjabi 
  'zh',     // Chinese (Simplified) 
  'zh-TW',  // Chinese (Traditional) 
  'ja',     // Japanese 
  'ko',     // Korean 
  'id',     // Indonesian 
  'ms',     // Malay 
  'vi',     // Vietnamese 
  'th',     // Thai 
  'tl'      // Tagalog/Filipino 
],
  fallbackLocales: {
    default: 'en'
  },
  catalogs: [
    {
      path: 'src/locales/{locale}/messages',
      include: ['src']
    }
  ]
});