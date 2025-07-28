import nextra from 'nextra';
import { type NextConfig } from 'next';

const withNextra = nextra({
  search: {
    codeblocks: false
  }
});

const nextConfig: NextConfig = {
  experimental: {
    swcPlugins: [['@lingui/swc-plugin', {}]],

    turbo: {
      rules: {
        '*.po': {
          loaders: ['@lingui/loader'],
          as: '*.js'
        }
      }
    },
    serverActions: {
      bodySizeLimit: '100mb'
    }
  }
};

export default withNextra(nextConfig);
