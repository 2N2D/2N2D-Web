'use client';

import React, { useState, useEffect, Suspense } from 'react';
import CSVUploader from '@/components/fileUploadElements/CSVUploader';
import { deleteCsv } from '@/lib/sessionHandling/sessionUpdater';
import { motion } from 'framer-motion';
import DataTable from '@/components/data/DataTable';
import './styles.css';

function Data() {
  const [missed, setMissed] = useState<number>(0);
  const [result, setResult] = useState<any>(null);
  const [selectedView, setSelectedView] = useState<number>(0);
  const [columnType, setColumnType] = useState<number>(0);

  let Heatmap,
    MissingDataHeatmap,
    PlotDistribution = null;
  if (selectedView == 1)
    Heatmap = React.lazy(() => import('@/components/data/HeatMap'));
  if (selectedView == 2)
    MissingDataHeatmap = React.lazy(
      () => import('@/components/data/MissingValues')
    );
  if (selectedView == 3)
    PlotDistribution = React.lazy(
      () => import('@/components/data/Distribution')
    );

  function handleNewData() {
    const data = sessionStorage.getItem('csvData');
    if (!data || data.length < 4) return;
    const _result = JSON.parse(data);
    setResult(_result);
    console.log(_result);
    let missing = 0;
    for (let key in _result.summary.missingValues) {
      missing += _result.summary.missingValues[key];
    }
    setMissed(missing);
  }

  async function clearData() {
    const curSesId = sessionStorage.getItem('currentSessionId');
    if (!curSesId) return;
    await deleteCsv(parseInt(curSesId));

    setResult(null);
    sessionStorage.removeItem('csvData');
  }

  useEffect(() => {
    handleNewData();
  }, []);

  return (
    <motion.div
      className='pageData'
      transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className={'flex h-auto w-full gap-[0.1rem]'}>
        <div className={'titleArea h-full'}>
          <h1 className={'dataTitle title'}>CSV tools</h1>
        </div>
        <div className={'flex w-[50%] flex-col gap-[0.1rem]'}>
          <div className='area'>
            <div className={'dataArea w-full'}>
              <h1 className={'subtitle'}>Add dataset:</h1>
              <CSVUploader callBack={handleNewData} />
              <button className={'deleteButton'} onClick={clearData}>
                Clear Data <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
          <div className={'flex h-full gap-[0.1rem]'}>
            <div className='area h-full w-full gap-[0.55rem]'>
              <h3 className={'subtitle text-[var(--warning-color)]'}>
                Warnings
              </h3>
              <div
                className={
                  'flex h-[11rem] flex-col overflow-y-auto rounded-[0.4rem] border-1 border-[var(--border-color)] p-[0.1rem]'
                }
              >
                {result?.results &&
                result.results.encoding_feasibility.warnings.length > 0 ? (
                  result.results.encoding_feasibility.warnings.map(
                    (warn: string, i: number) => (
                      <div key={i} className={'warningItem'}>
                        <p>{warn}</p>
                      </div>
                    )
                  )
                ) : (
                  <div className={'warningItem'}>
                    <p>No warnings</p>
                  </div>
                )}
              </div>
            </div>
            <div className='area h-full w-full gap-[0.55rem]'>
              <h3 className={'subtitle text-primary'}>Recommendations</h3>
              <div
                className={
                  'bg-card-background flex h-[11rem] flex-col overflow-y-auto rounded-[0.4rem] border-1 border-[var(--border-color)] p-[0.1rem]'
                }
              >
                {result &&
                result.results &&
                result.results.encoding_feasibility.recommendations.length >
                  0 ? (
                  result.results.encoding_feasibility.recommendations.map(
                    (rec: string, i: number) => (
                      <div className={'warningItem'} key={i}>
                        <p>{rec}</p>
                      </div>
                    )
                  )
                ) : (
                  <div className={'warningItem'}>
                    <p>No recommendations</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        <div className={'flex w-[50%] flex-col gap-[0.1rem]'}>
          <div className='area w-full gap-[1rem]'>
            <h3 className={'subtitle ml-[0.5rem]'}>Encoding Feasibility</h3>
            <div className='dataSum'>
              <div className='info'>
                <h1>One hot</h1>
                <h2>
                  {result == null
                    ? '-'
                    : result.results.encoding_feasibility.is_safe_for_onehot
                      ? 'Safe'
                      : 'Unsafe'}
                </h2>
              </div>
              <div className='info'>
                <h1>Cur. Memory</h1>
                <h2>
                  {result == null
                    ? '-'
                    : result.results.encoding_feasibility.memory_estimate
                        .current_memory_mb + 'mb'}
                </h2>
              </div>
              <div className='info'>
                <h1>Est. Memory</h1>
                <h2>
                  {result == null
                    ? '-'
                    : result.results.encoding_feasibility.memory_estimate
                        .estimated_memory_mb + 'mb'}
                </h2>
              </div>
              <div className='info'>
                <h1>Overall</h1>
                <h2>
                  {result == null
                    ? '-'
                    : result.results.encoding_feasibility
                        .overall_recommendation}
                </h2>
              </div>
            </div>
          </div>
          <div className='area w-full gap-[1rem]'>
            <h3 className={'subtitle ml-[0.5rem]'}>Dataset Overview</h3>
            <div className='dataSum'>
              <div className='info'>
                <h1>File</h1>
                <h2>
                  {result == null
                    ? 'No file uploaded'
                    : result.summary.filename}
                </h2>
              </div>
              <div className='info'>
                <h1>Rows</h1>
                <h2>{result == null ? '-' : result.summary.rows}</h2>
              </div>
              <div className='info'>
                <h1>Columns</h1>
                <h2>{result == null ? '-' : result.summary.columns}</h2>
              </div>
              <div className='info'>
                <h1>Missing values</h1>
                <h2>{result == null ? '-' : missed}</h2>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className='area tableArea'>
        <DataTable result={result} />
      </div>
      <div className={'area'}>
        <div className={'viewButtons'}>
          <button
            onClick={() => {
              setSelectedView(0);
            }}
            style={
              selectedView == 0
                ? {
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--card-background)'
                  }
                : {}
            }
          >
            <i className='fa-solid fa-binary-lock'></i> Encoding Info
          </button>
          <button
            onClick={() => {
              setSelectedView(1);
            }}
            style={
              selectedView == 1
                ? {
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--card-background)'
                  }
                : {}
            }
          >
            <i className='fa-solid fa-hashtag'></i> Correlation Matrix
          </button>
          <button
            onClick={() => {
              setSelectedView(2);
            }}
            style={
              selectedView == 2
                ? {
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--card-background)'
                  }
                : {}
            }
          >
            <i className='fa-solid fa-value-absolute'></i> Missing Values
            Heatmap
          </button>
          <button
            onClick={() => {
              setSelectedView(3);
            }}
            style={
              selectedView == 3
                ? {
                    backgroundColor: 'var(--primary-color)',
                    color: 'var(--card-background)'
                  }
                : {}
            }
          >
            <i className='fa-solid fa-chart-simple'></i> Data Distribution
          </button>
        </div>
        {selectedView == 1 && result && result.results && Heatmap ? (
          <motion.div
            transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={'area flex items-center justify-center'}
          >
            <h1 className={'subtitle'}>Correlation Matrix</h1>
            <Suspense
              fallback={
                <motion.div
                  transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={
                    'flex h-[600px] flex-col items-center justify-center gap-[1rem]'
                  }
                >
                  <div className={'spinner'} />
                  <h1>
                    <b>Loading</b>
                  </h1>
                </motion.div>
              }
            >
              <motion.div
                transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={'h-full w-full'}
              >
                <Heatmap
                  matrix={result.results.visualization_data.correlation_matrix}
                />
              </motion.div>
            </Suspense>
          </motion.div>
        ) : (
          ''
        )}
        {selectedView == 2 && result && result.results && MissingDataHeatmap ? (
          <motion.div
            transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={'area flex items-center justify-center'}
          >
            <h1 className={'subtitle'}>Missing Data Heatmap</h1>
            <Suspense
              fallback={
                <motion.div
                  transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={
                    'flex h-[600px] flex-col items-center justify-center gap-[1rem]'
                  }
                >
                  <div className={'spinner'} />
                  <h1>
                    <b>Loading</b>
                  </h1>
                </motion.div>
              }
            >
              <motion.div
                transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={'h-full w-full'}
              >
                <MissingDataHeatmap
                  columns={
                    result.results.visualization_data.missing_data_heatmap
                      .columns
                  }
                  data={
                    result.results.visualization_data.missing_data_heatmap.data
                  }
                />
              </motion.div>
            </Suspense>
          </motion.div>
        ) : (
          ''
        )}
        {selectedView == 3 && result && result.results && PlotDistribution ? (
          <motion.div
            transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={'area flex items-center justify-center'}
          >
            <h1 className={'subtitle'}>Data Distribution</h1>
            <Suspense
              fallback={
                <motion.div
                  transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={
                    'flex h-[600px] flex-col items-center justify-center gap-[1rem]'
                  }
                >
                  <div className={'spinner'} />
                  <h1>
                    <b>Loading</b>
                  </h1>
                </motion.div>
              }
            >
              <motion.div
                transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={'h-full w-full'}
              >
                <PlotDistribution
                  data={result.results.visualization_data.distribution_data}
                />
              </motion.div>
            </Suspense>
          </motion.div>
        ) : (
          ''
        )}
        {selectedView == 0 && result && result.results ? (
          <motion.div
            className={'area flex h-[600px] flex-col justify-center gap-[1rem]'}
            transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className={'subtitle'}>Encoding Info</h1>
            <div
              className={
                'gap[0.5rem] flex w-fit flex-col rounded-[0.4rem] p-[0.5rem]'
              }
              style={{ backgroundColor: 'var(--background-color)' }}
            >
              <h1 className={'bold'}>
                Overall recommendation:{' '}
                <b
                  style={
                    result.results.encoding_feasibility
                      .overall_recommendation == 'safe'
                      ? { color: 'var(--success-color)' }
                      : { color: 'var(--error-color)' }
                  }
                >
                  {result.results.encoding_feasibility.overall_recommendation ==
                  'safe'
                    ? 'Safe'
                    : result.results.encoding_feasibility
                        .overall_recommendation}
                </b>
              </h1>
              <h1 className={'bold'}>
                Onehot encoding:{' '}
                <b
                  style={
                    result.results.encoding_feasibility.is_safe_for_onehot
                      ? { color: 'var(--success-color)' }
                      : { color: 'var(--error-color)' }
                  }
                >
                  {result.results.encoding_feasibility.is_safe_for_onehot
                    ? 'Safe'
                    : 'Not safe'}
                </b>
              </h1>
            </div>
            <div
              className={
                'align-center flex justify-evenly rounded-[0.4rem] p-[0.5rem]'
              }
              style={{ backgroundColor: 'var(--background-color)' }}
            >
              <div
                className={
                  'flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-[1rem]'
                }
                style={{ backgroundColor: 'var(--card-background)' }}
              >
                <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
                  Column types:
                </h1>
                <div
                  className={
                    'flex flex-col items-center justify-center gap-[0.5rem]'
                  }
                >
                  <div className={'viewButtons'}>
                    <button
                      onClick={() => {
                        setColumnType(0);
                      }}
                      className={
                        columnType == 0 ? 'bg-primary text-card-background' : ''
                      }
                    >
                      Numerical
                    </button>
                    <button
                      onClick={() => {
                        setColumnType(1);
                      }}
                      className={
                        columnType == 1 ? 'bg-primary text-card-background' : ''
                      }
                    >
                      Categorical
                    </button>
                  </div>
                  {columnType == 0 ? (
                    <div
                      className={
                        'flex h-[11rem] flex-col gap-[0.2rem] overflow-y-auto'
                      }
                    >
                      {result.results.column_types.numerical.length > 0 ? (
                        result.results.column_types.numerical.map(
                          (val: string, i: number) => (
                            <div
                              key={i}
                              className={
                                'bg-background w-full rounded-[0.4rem] p-1'
                              }
                            >
                              {val}
                            </div>
                          )
                        )
                      ) : (
                        <div>No numerical columns</div>
                      )}
                    </div>
                  ) : (
                    <div
                      className={
                        'flex h-[11rem] flex-col gap-[0.2rem] overflow-y-auto'
                      }
                    >
                      {result.results.column_types.categorical.length > 0 ? (
                        result.results.column_types.categorical.map(
                          (val: string, i: number) => (
                            <div
                              key={i}
                              className={
                                'bg-background w-full rounded-[0.4rem] p-1'
                              }
                            >
                              {val}
                            </div>
                          )
                        )
                      ) : (
                        <div>No categorical columns</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              <div
                className={
                  'flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-4'
                }
                style={{ backgroundColor: 'var(--card-background)' }}
              >
                <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
                  Categorical summary:
                </h1>
                <p>
                  Risky for onehot:{' '}
                  {result.results.encoding_feasibility.categorical_summary.risky_for_onehot.toString()}
                </p>
                <p>
                  Safe for onehot:{' '}
                  {result.results.encoding_feasibility.categorical_summary.safe_for_onehot.toString()}
                </p>
                <p>
                  Total categorical:{' '}
                  {result.results.encoding_feasibility.categorical_summary.total_categorical.toString()}
                </p>
              </div>
              <div
                className={
                  'flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-[1rem]'
                }
                style={{ backgroundColor: 'var(--card-background)' }}
              >
                <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
                  Columns estimate:
                </h1>
                <p>
                  Risky for onehot:{' '}
                  {result.results.encoding_feasibility.column_estimate.current_columns.toString()}
                </p>
                <p>
                  Safe for onehot:{' '}
                  {result.results.encoding_feasibility.column_estimate.estimated_final_columns.toString()}
                </p>
                <p>
                  Total categorical:{' '}
                  {result.results.encoding_feasibility.column_estimate.new_columns_added.toString()}
                </p>
              </div>
              <div
                className={
                  'bg-card-background flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-[1rem]'
                }
              >
                <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
                  High cardinality columns:
                </h1>
                <div>
                  {result.results.encoding_feasibility.high_cardinality_columns
                    .length > 0 ? (
                    result.results.encoding_feasibility.high_cardinality_columns.map(
                      (val: any, i: number) => (
                        <p key={i}>
                          `{val.column} ({val.unique_values} unique,{' '}
                          {val.severity})
                        </p>
                      )
                    )
                  ) : (
                    <p>No such columns</p>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          ''
        )}
      </div>
    </motion.div>
  );
}

export default Data;
