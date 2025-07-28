import { motion } from 'framer-motion';
import React, { useState } from 'react';
import { useLingui, Trans } from '@lingui/react/macro';

export default function EncodingView({ result }: { result: any }) {
  const [columnType, setColumnType] = useState<number>(0);
  const { t } = useLingui();

  return (
    <motion.div
      className={'area flex h-[600px] flex-col justify-center gap-[1rem]'}
      transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h1 className={'subtitle'}>
        <Trans>Encoding Info</Trans>
      </h1>
      <div
        className={
          'gap[0.5rem] flex w-fit flex-col rounded-[0.4rem] p-[0.5rem]'
        }
        style={{ backgroundColor: 'var(--background-color)' }}
      >
        <h1 className={'bold'}>
          <Trans>Overall recommendation:</Trans>{' '}
          <b
            style={
              result.results.encoding_feasibility.overall_recommendation ==
              'safe'
                ? { color: 'var(--success-color)' }
                : { color: 'var(--error-color)' }
            }
          >
            {result.results.encoding_feasibility.overall_recommendation ==
            'safe'
              ? t`Safe`
              : result.results.encoding_feasibility.overall_recommendation}
          </b>
        </h1>
        <h1 className={'bold'}>
          <Trans>Onehot encoding:</Trans>{' '}
          <b
            style={
              result.results.encoding_feasibility.is_safe_for_onehot
                ? { color: 'var(--success-color)' }
                : { color: 'var(--error-color)' }
            }
          >
            {result.results.encoding_feasibility.is_safe_for_onehot
              ? t`Safe`
              : t`Not safe`}
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
            <Trans>Column types:</Trans>
          </h1>
          <div
            className={'flex flex-col items-center justify-center gap-[0.5rem]'}
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
                <Trans>Numerical</Trans>
              </button>
              <button
                onClick={() => {
                  setColumnType(1);
                }}
                className={
                  columnType == 1 ? 'bg-primary text-card-background' : ''
                }
              >
                <Trans>Categorical</Trans>
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
                        className={'bg-background w-full rounded-[0.4rem] p-1'}
                      >
                        {val}
                      </div>
                    )
                  )
                ) : (
                  <div>
                    <Trans>No numerical columns</Trans>
                  </div>
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
                        className={'bg-background w-full rounded-[0.4rem] p-1'}
                      >
                        {val}
                      </div>
                    )
                  )
                ) : (
                  <div>
                    <Trans>No categorical columns</Trans>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        <div
          className={'flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-4'}
          style={{ backgroundColor: 'var(--card-background)' }}
        >
          <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
            <Trans>Categorical summary:</Trans>
          </h1>
          <p>
            <Trans>Risky for onehot:</Trans>{' '}
            {result.results.encoding_feasibility.categorical_summary.risky_for_onehot.toString()}
          </p>
          <p>
            <Trans>Safe for onehot:</Trans>{' '}
            {result.results.encoding_feasibility.categorical_summary.safe_for_onehot.toString()}
          </p>
          <p>
            <Trans>Total categorical:</Trans>{' '}
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
            <Trans>Columns estimate:</Trans>
          </h1>
          <p>
            <Trans>Risky for onehot:</Trans>{' '}
            {result.results.encoding_feasibility.column_estimate.current_columns.toString()}
          </p>
          <p>
            <Trans>Safe for onehot:</Trans>{' '}
            {result.results.encoding_feasibility.column_estimate.estimated_final_columns.toString()}
          </p>
          <p>
            <Trans>Total categorical:</Trans>{' '}
            {result.results.encoding_feasibility.column_estimate.new_columns_added.toString()}
          </p>
        </div>
        <div
          className={
            'bg-card-background flex h-[20rem] w-[20rem] flex-col rounded-[0.4rem] p-[1rem]'
          }
        >
          <h1 className={'mb-[0.5rem] text-[1.5rem] font-bold'}>
            <Trans>High cardinality columns:</Trans>
          </h1>
          <div>
            {result.results.encoding_feasibility.high_cardinality_columns
              .length > 0 ? (
              result.results.encoding_feasibility.high_cardinality_columns.map(
                (val: any, i: number) => (
                  <p key={i}>
                    <Trans>
                      {val.column} ({val.unique_values} unique, {val.severity})
                    </Trans>
                  </p>
                )
              )
            ) : (
              <p>
                <Trans>No such columns</Trans>
              </p>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
