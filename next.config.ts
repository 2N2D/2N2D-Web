import nextra from 'nextra'
import  {type NextConfig} from 'next';

const withNextra = nextra({
    search: {
        codeblocks: false
    }
})

const nextConfig : NextConfig = {
    experimental: {
        serverActions: {
            bodySizeLimit: '50mb',
        },
    },
}

export default withNextra(nextConfig)
