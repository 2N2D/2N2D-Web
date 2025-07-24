import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Style from './navbar.module.css';
import { getSession } from '@/lib/auth/authentication';

export default function Navbar() {
  const [logged, setLogged] = useState<boolean>(false);
  const [showLogo, setShowLogo] = useState<boolean>(false);

  async function checkLogged() {
    if ((await getSession()) === '200') {
      sessionStorage.setItem('logged', 'true');
      setLogged(true);
    } else {
      setLogged(false);
      sessionStorage.setItem('logged', 'false');
    }
  }

  useEffect(() => {
    checkLogged();

    const handleScroll = () => {
      setShowLogo(window.scrollY > 50); // Show logo after scrolling 50px
    };

    const handleMouseEnter = () => {
      window.addEventListener('scroll', handleScroll);
    };

    const handleMouseLeave = () => {
      window.removeEventListener('scroll', handleScroll);
    };

    window.addEventListener('mouseenter', handleMouseEnter);
    window.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mouseenter', handleMouseEnter);
      window.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  return (
    <nav className={Style.navbarCont}>
      <div className={Style.navbar}>
        {showLogo && (
          <motion.a
            href='/'
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className={Style.divider}></div>
            <img src='logo2n2d.svg' alt='logo' />
          </motion.a>
        )}
        <div className={Style.links}>
          <motion.a
            href='/docs'
            className={Style.button}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            Docs
          </motion.a>
          <motion.a
            href={logged ? '/dash' : '/login'}
            className={Style.mainButton}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {logged ? 'Dashboard' : 'Login'}
          </motion.a>
        </div>
      </div>
    </nav>
  );
}
