import { motion } from 'framer-motion';

export default function LoadingSpinner() {
  return (
    <motion.div
      transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className='flex h-[600px] flex-col items-center justify-center gap-[1rem]'
    >
      <div className='spinner' />
      <h1>
        <b>Loading</b>
      </h1>
    </motion.div>
  );
}
