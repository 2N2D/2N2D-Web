import  { Navbar, Layout } from 'nextra-theme-docs';
import { getPageMap } from 'nextra/page-map';
import { ReactNode } from 'react';
import 'nextra-theme-docs/style.css';

// const banner = <Banner storageKey='some-key'>Nextra 4.0 is released 🎉</Banner>;
const navbar = (
  <Navbar logo={<b>2n2d</b>}/>
);

export default async function ({ children }: { children: ReactNode }) {
  const pageMap = await getPageMap("/docs")

  return (
    <Layout navbar={navbar} pageMap={pageMap} >
      <div className={"flex flex-col items-center justify-center"}>
      {children}
      </div>
    </Layout>
  );
}