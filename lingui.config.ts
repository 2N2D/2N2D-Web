import { defineConfig } from '@lingui/cli';

export default defineConfig({
  sourceLocale: 'en',
  locales: ['en', 'ro', 'de'],
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