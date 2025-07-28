'use client';

import React, { useState, useEffect, Suspense } from 'react';
import CSVUploader from '@/components/fileUploadElements/CSVUploader';
import { deleteCsv } from '@/lib/sessionHandling/sessionUpdater';
import { motion } from 'framer-motion';
import DataTable from '@/components/data/DataTable';
import './styles.css';
import DynamicInfoArea from '@/components/data/InfoArea';
import { Trans, useLingui } from '@lingui/react/macro';
import ViewSelector from '@/components/data/ViewSelector';
import WarningsRecommendations from '@/components/data/WarningsRecommendations';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import EncodingView from '@/components/data/EncodingInfoView';

function Data() {
  const [missed, setMissed] = useState<number>(0);
  const [result, setResult] = useState<any>(null);
  const [selectedView, setSelectedView] = useState<number>(0);

  const { t } = useLingui();

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
      <div className={'flex w-full gap-[0.5rem]'}>
        <div className={'titleArea'}>
          <h1 className={'dataTitle title'}>
            <Trans>CSV tools</Trans>
          </h1>
        </div>
        <div className={'flex w-[50%] flex-col gap-[0.5rem]'}>
          <div className='area'>
            <div className={'dataArea w-full'}>
              <h1 className={'subtitle'}>
                <Trans>Add dataset:</Trans>
              </h1>
              <CSVUploader callBack={handleNewData} />
              <button className={'deleteButton'} onClick={clearData}>
                <Trans>Clear Data</Trans>{' '}
                <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
          <WarningsRecommendations
            warnings={result?.results?.encoding_feasibility?.warnings || []}
            recommendations={
              result?.results?.encoding_feasibility?.recommendations || []
            }
          />
        </div>
        <div className={'flex w-[50%] flex-col gap-[0.5rem]'}>
          <DynamicInfoArea
            title={t`Dataset Overview`}
            items={[
              {
                label: t`File`,
                value:
                  result == null ? t`No file uploaded` : result.summary.filename
              },
              {
                label: t`Rows`,
                value: result == null ? '-' : result.summary.rows
              },
              {
                label: t`Columns`,
                value: result == null ? '-' : result.summary.columns
              },
              {
                label: t`Missing values`,
                value: result == null ? '-' : missed
              }
            ]}
          />
          <DynamicInfoArea
            title={t`Encoding Feasibility`}
            items={[
              {
                label: t`One hot`,
                value:
                  result == null
                    ? '-'
                    : result.results.encoding_feasibility.is_safe_for_onehot
                      ? t`Safe`
                      : t`Unsafe`
              },
              {
                label: t`Cur. Memory`,
                value:
                  result == null
                    ? '-'
                    : result.results.encoding_feasibility.memory_estimate
                        .current_memory_mb + t`mb`
              },
              {
                label: t`Est. Memory`,
                value:
                  result == null
                    ? '-'
                    : result.results.encoding_feasibility.memory_estimate
                        .estimated_memory_mb + t`mb`
              },
              {
                label: t`Overall`,
                value:
                  result == null
                    ? '-'
                    : result.results.encoding_feasibility.overall_recommendation
              }
            ]}
          />
        </div>
      </div>
      {result && result.results && (
        <motion.div
          transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className='area tableArea'
        >
          <DataTable result={result} />
        </motion.div>
      )}
      {result && result.results && (
        <motion.div
          transition={{ delay: 0.6, duration: 0.2, ease: 'easeOut' }}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={'area'}
        >
          <ViewSelector
            selectedView={selectedView}
            onViewChange={setSelectedView}
          />
          {selectedView == 1 && result && result.results && Heatmap && (
            <motion.div
              transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={'area flex items-center justify-center'}
            >
              <h1 className={'subtitle'}>
                <Trans>Correlation Matrix</Trans>
              </h1>
              <Suspense fallback={<LoadingSpinner />}>
                <motion.div
                  transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={'h-full w-full'}
                >
                  <Heatmap
                    matrix={
                      result.results.visualization_data.correlation_matrix
                    }
                  />
                </motion.div>
              </Suspense>
            </motion.div>
          )}
          {selectedView == 2 &&
            result &&
            result.results &&
            MissingDataHeatmap && (
              <motion.div
                transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={'area flex items-center justify-center'}
              >
                <h1 className={'subtitle'}>
                  <Trans>Missing Data Heatmap</Trans>
                </h1>
                <Suspense fallback={<LoadingSpinner />}>
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
                        result.results.visualization_data.missing_data_heatmap
                          .data
                      }
                    />
                  </motion.div>
                </Suspense>
              </motion.div>
            )}
          {selectedView == 3 &&
            result &&
            result.results &&
            PlotDistribution && (
              <motion.div
                transition={{ delay: 0.4, duration: 0.2, ease: 'easeOut' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={'area flex items-center justify-center'}
              >
                <h1 className={'subtitle'}>
                  <Trans>Data Distribution</Trans>
                </h1>
                <Suspense fallback={<LoadingSpinner />}>
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
            )}
          {selectedView == 0 && result && result.results && (
            <EncodingView result={result} />
          )}
        </motion.div>
      )}
    </motion.div>
  );
}

export default Data;
