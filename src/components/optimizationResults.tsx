import React from 'react';
import { motion } from 'framer-motion';
import Styles from './optimizationResults.module.css';

type OptimizationResult = {
  best_config: {
    neurons: React.ReactNode;
    layers: React.ReactNode;
    test_loss: React.ReactNode;
    r2_score: React.ReactNode;
  };
  results: Array<{
    neurons: React.ReactNode;
    layers: React.ReactNode;
    test_loss: React.ReactNode;
    r2_score: React.ReactNode;
  }>;
  url?: string;
};

type Props = {
  result: OptimizationResult | null;
  progress: number;
  downloading: boolean;
  downloadOptimized: () => void;
};

const OptimizationResults: React.FC<Props> = ({
  result,
  progress,
  downloading,
  downloadOptimized
}) => {
  if (
    !result ||
    !result.best_config ||
    JSON.stringify(result).length <= 2 ||
    JSON.stringify(result) === 'null'
  )
    return null;

  return (
    <motion.div
      className={`${Styles.resultArea} ${progress === 100 ? 'area' : ''}`}
      style={progress === 100 ? { height: '100%' } : { height: 0 }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className={'subtitle'}>Optimization results:</h1>
      <div className={'flex w-full flex-col gap-[1rem] p-[1rem]'}>
        <h2 className={'subtitle'}>Best configuration:</h2>
        <div className={Styles.result}>
          <div className={Styles.info}>
            <h2> Neurons: </h2> {result.best_config.neurons}
          </div>
          <div className={Styles.info}>
            <h2> Layers: </h2> {result.best_config.layers}
          </div>
          <div className={Styles.info}>
            <h2> Test loss: </h2> {result.best_config.test_loss}
          </div>
          <div className={Styles.info}>
            <h2> R2 score: </h2> {result.best_config.r2_score}
          </div>
        </div>
      </div>
      <table className={Styles.table}>
        <thead>
          <tr>
            <th>Configuration</th>
            <th>Neurons</th>
            <th>Layers</th>
            <th>Test loss:</th>
            <th>R2 score:</th>
          </tr>
          <tr>
            <td>Best config</td>
            <td>{result.best_config.neurons}</td>
            <td>{result.best_config.layers}</td>
            <td>{result.best_config.test_loss}</td>
            <td>{result.best_config.r2_score}</td>
          </tr>
        </thead>
        <tbody>
          {result.results.map((res, i) => (
            <tr key={i}>
              <td>Config {i + 1}</td>
              <td>{res.neurons}</td>
              <td>{res.layers}</td>
              <td>{res.test_loss}</td>
              <td>{res.r2_score}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={downloadOptimized} disabled={downloading}>
        {downloading ? 'Downloading...' : 'Download optimized'}{' '}
        <i className='fa-solid fa-file-arrow-down'></i>
      </button>
    </motion.div>
  );
};

export default OptimizationResults;
