import { Navbar, Layout } from 'nextra-theme-docs';
import { getPageMap } from 'nextra/page-map';
import { ReactNode } from 'react';
import 'nextra-theme-docs/style.css';

const navbar = (
  <Navbar
    logo={
      <img src='logo2n2d.svg' style={{ maxWidth: '100px', height: 'auto' }} />
    }
  />
);

export default async function ({ children }: { children: ReactNode }) {
  const pageMap = await getPageMap('/docs');

  return (
    <Layout navbar={navbar} pageMap={pageMap}>
      <div className={'flex flex-col items-center justify-center'}>
        {children}
      </div>
    </Layout>
  );
}
