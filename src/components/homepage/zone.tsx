import React, { useState } from 'react';
import { motion } from 'framer-motion';

function Zone({
  title,
  desc,
  img,
  index,
  icon,
  highlights = []
}: {
  title: string;
  desc: string;
  img: string;
  index: number;
  icon?: string;
  highlights?: string[];
}) {
  const [flipped, setFlipped] = useState(false);
  const fromLeft = index % 2 === 0;

  return (
    <motion.section
      className='screenshotZone'
      initial={{ opacity: 0, x: fromLeft ? -80 : 80 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.7, delay: 0.1 * index }}
      style={{
        display: 'flex',
        flexDirection: fromLeft ? 'row' : 'row-reverse',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        gap: '3rem',
        width: '100%',
        position: 'relative'
      }}
    >
      <div className='zoneText' style={{ flex: 1, zIndex: 1 }}>
        <motion.div
          className='zoneCard'
          style={{
            perspective: 1200,
            minHeight: 320,
            position: 'relative'
          }}
        >
          <motion.div
            className='zoneCardInner'
            animate={{ rotateY: flipped ? 180 : 0 }}
            transition={{ duration: 0.7 }}
            style={{
              position: 'relative',
              width: '100%',
              minHeight: 320,
              transformStyle: 'preserve-3d'
            }}
          >
            <div
              className='zoneCardFace zoneCardFront'
              style={{
                backfaceVisibility: 'hidden',
                position: 'absolute',
                width: '100%',
                minHeight: 320,
                top: 0,
                left: 0
              }}
            >
              <h2 className='zoneTitle'>{title}</h2>
              <p className='zoneDesc'>{desc}</p>
              <a
                className='zoneMoreBtn'
                href='#!'
                onClick={(e) => {
                  e.preventDefault();
                  setFlipped(true);
                }}
              >
                View More <i className='fa-solid fa-arrow-right'></i>
              </a>
            </div>

            <div className='zoneCardFace zoneCardBack'>
              <h3 className='zoneTitle' style={{ marginBottom: '1rem' }}>
                <i className={icon + ' zoneIcon'} style={{ marginRight: 8 }} />
                {title}
              </h3>
              <ul className='zoneHighlights' style={{ marginBottom: '1.5rem' }}>
                {highlights.map((h, i) => (
                  <li key={i}>
                    <i className='fa-solid fa-check-circle highlightIcon' /> {h}
                  </li>
                ))}
              </ul>
              <a
                className='zoneMoreBtn'
                href='#!'
                onClick={(e) => {
                  e.preventDefault();
                  setFlipped(false);
                }}
              >
                <i className='fa-solid fa-arrow-left'></i> Back
              </a>
            </div>
          </motion.div>
        </motion.div>
      </div>
      <div
        className='zoneImage'
        style={{
          flex: 1,
          display: 'flex',
          justifyContent: 'center',
          zIndex: 1
        }}
      >
        <motion.img
          src={img}
          alt={title + ' screenshot'}
          style={{
            width: '100%',
            borderRadius: '1.5rem',
            boxShadow: '0 8px 32px 0 rgba(0,0,0,0.18)',
            border: '2px solid rgba(255,255,255,0.08)',
            background: 'rgba(30,30,40,0.2)'
          }}
          animate={{
            y: [0, -10, 0]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            repeatType: 'loop',
            ease: 'easeInOut',
            delay: Math.random() * 2
          }}
        />
      </div>
    </motion.section>
  );
}

export default Zone;
