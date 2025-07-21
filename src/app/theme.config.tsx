import type { DocsThemeConfig } from 'nextra-theme-docs'

const config: DocsThemeConfig = {
  logo: <span>My App Docs</span>,
  project: {
    link: 'https://github.com/my-org/my-app',
  },
  docsRepositoryBase: 'https://github.com/my-org/my-app/blob/main/src/pages',
  footer: {
    text: 'MIT © 2025 My App',
  },
}

export default config
