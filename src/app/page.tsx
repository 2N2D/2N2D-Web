'use client';

import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import ParticleNetwork from '@/components/visual/particleNetwork';
import './homeStyles.css';
import Navbar from '@/components/homepage/navbar';
import Zone from '@/components/homepage/zone';

const containerVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
};

const zones = [
  {
    title: 'Dashboard Overview',
    desc: 'Monitor your neural network projects at a glance.',
    img: 'dashboard.png',
    icon: 'fa-solid fa-gauge-high',
    highlights: ['Project summaries', 'Quick stats', 'Recent activity feed']
  },
  {
    title: 'Neural network visualization',
    desc: 'Drag and drop to build and visualize neural networks.',
    img: 'vizualization.png',
    icon: 'fa-solid fa-diagram-project',
    highlights: [
      'Interactive graph visualization',
      'Layer-by-layer view',
      'Live architecture preview'
    ]
  },
  {
    title: 'Data Analysis',
    desc: 'Visualize and preprocess your datasets with ease.',
    img: 'data.png',
    icon: 'fa-solid fa-table',
    highlights: [
      'Heatmaps & distributions',
      'Missing value detection',
      'Encoding feasiability tools'
    ]
  },
  {
    title: 'Model Optimization',
    desc: 'Experiment and optimize architectures interactively.',
    img: 'optimization.png',
    icon: 'fa-solid fa-gears',
    highlights: [
      'Mutliple optimization algorithms',
      'Performance charts',
      'Export best models'
    ]
  }
];

export default function Home() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: ref });

  // Parallax for hero text
  const heroY = useTransform(scrollYProgress, [0, 0.2], [0, -80]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.1], [1, 0.5]);

  return (
    <motion.main
      className='pageHome'
      initial='hidden'
      animate='visible'
      variants={containerVariants}
      ref={ref}
      style={{ overflowY: 'auto', height: '100vh', position: 'relative' }}
    >
      <Navbar />
      <ParticleNetwork />

      {/* Hero Section */}
      <motion.section
        className='heroZone'
        style={{
          minHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          zIndex: 2,
          y: heroY,
          opacity: heroOpacity
        }}
      >
        <motion.img
          src='logo2n2d.svg'
          alt='2N2D Logo'
          className='landingLogo'
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.7 }}
          style={{ maxWidth: '300px', width: '80%' }}
        />
        <motion.p
          className='landingSubtitle'
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.7 }}
        >
          Build, analyze, and optimize neural networks visually.
        </motion.p>
        <motion.a
          href='/dash'
          className='ctaButton'
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          Get Started
        </motion.a>
        <motion.div
          className='scrollArrow'
          initial={{ opacity: 1, y: 0 }}
          animate={{ opacity: 1, y: [20, 0, 20] }}
          transition={{
            duration: 3,
            repeat: Infinity,
            repeatType: 'loop'
          }}
          style={{
            position: 'absolute',
            bottom: '2rem',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            fontSize: '2rem',
            color: '#fff',
            zIndex: 3
          }}
        >
          <i className='fas fa-arrow-down'></i>
        </motion.div>
      </motion.section>

      <div className='zonesContainer'>
        {zones.map((zone, idx) => (
          <Zone
            key={zone.title}
            title={zone.title}
            desc={zone.desc}
            img={zone.img}
            index={idx}
            highlights={zone.highlights}
            icon={zone.icon}
          />
        ))}
      </div>

      <motion.section
        className='ctaZone'
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.5 }}
        transition={{ duration: 0.7 }}
      >
        <h2>Ready to supercharge your neural network workflow?</h2>
        <a href='/dash' className='ctaButton'>
          Try 2N2D Now
        </a>
      </motion.section>
    </motion.main>
  );
}
