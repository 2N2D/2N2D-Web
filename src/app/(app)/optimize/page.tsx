'use client';

import React, { useState, useEffect, FormEvent } from 'react';
import { getOptimizationStatus, startOptimization } from '@/lib/2n2dAPI';
import { downloadFileRequest } from '@/lib/frontend/feHandler';
import { getCurrentUser } from '@/lib/auth/authentication';
import { getSession } from '@/lib/sessionHandling/sessionManager';
import './styles.css';
import ONNXUploader from '@/components/fileUploadElements/ONNXUploader';
import CSVUploader from '@/components/fileUploadElements/CSVUploader';
import { deleteCsv, deleteOnnx } from '@/lib/sessionHandling/sessionUpdater';
import { motion } from 'framer-motion';
import OptimizationResults from '@/components/optimizationResults';
import { Trans, useLingui } from '@lingui/react/macro';

export default function Optimize() {
  const [features, setFeatures] = useState<string[]>([]);
  const [status, setStatus] = useState<string>('');
  const [progress, setProgress] = useState<number>(-1);
  const [csvFileName, setCsvFileName] = useState<string>('');
  const [onnxFileName, setOnnxFileName] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [downloading, setDownloading] = useState<boolean>(false);
  const [alert, setAlert] = useState<string | null>(null);

  const { t } = useLingui();

  async function statusUpdate() {
    const eventSource = new EventSource(
      await getOptimizationStatus(await getCurrentUser())
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data.status);
      setProgress(data.progress);
      console.log('Progress update:', data);
    };

    eventSource.onerror = (err) => {
      console.log('SSE error:', err);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }

  async function loadSession() {
    if (sessionStorage.getItem('currentSessionId')) {
      const session = await getSession(
        parseInt(sessionStorage.getItem('currentSessionId')!)
      );
      console.log(JSON.stringify(session.optResult));
      if (
        session &&
        session.optResult &&
        JSON.stringify(session.optResult).length > 2 &&
        JSON.stringify(session.optResult) != 'null' &&
        (session.optResult as any).best_config
      ) {
        setResult(session?.optResult);
        setProgress(100);
        setStatus('Optimization finished');
      } else {
        setResult(null);
        setProgress(-1);
      }

      setOnnxFileName(session?.onnxName!);
      setCsvFileName(session?.csvName!);
    }
  }

  function populateLists() {
    setFeatures([]);

    if (sessionStorage.getItem('modelName'))
      setOnnxFileName(sessionStorage.getItem('modelName')!);

    if (
      !sessionStorage.getItem('csvData') ||
      !sessionStorage.getItem('modelData')
    ) {
      return;
    }

    if (
      sessionStorage.getItem('modelData')!.length < 4 ||
      sessionStorage.getItem('csvData')!.length < 4
    )
      return;

    let csv = JSON.parse(sessionStorage.getItem('csvData')!);

    console.log(JSON.stringify(result));

    setCsvFileName(csv.summary.filename);

    let src = csv!.data;
    let feat = [];
    for (let key in src[0]) feat.push(key);

    setFeatures(feat);
  }

  useEffect(() => {
    populateLists();
    loadSession();
  }, []);

  async function optimize(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    setAlert(null);

    const formData = new FormData(e.currentTarget);
    const featuresaux = formData.getAll('sIFeatures[]');
    let Ifeatures: String[] = [];
    featuresaux.forEach((feat) => {
      Ifeatures.push(feat.toString());
    });
    const target = formData.get('target')?.toString()!;
    const encoding = formData.get('encoding')?.toString();
    const maxEpochs = Number(formData.get('epochs'));
    const strat = formData.get('strat')?.toString();

    const sesId = sessionStorage.getItem('currentSessionId');
    if (!sesId) return;
    const session = await getSession(parseInt(sesId));
    if (!session) return;

    setProgress(0);
    setStatus('Starting optimization...');
    await statusUpdate();

    const _result = await startOptimization(
      Ifeatures,
      target,
      maxEpochs,
      parseInt(sesId),
      session.csvUrl!,
      session.onnxUrl!,
      encoding!,
      strat!
    );

    if (typeof _result === 'string') {
      setAlert(_result);
      setProgress(-1);
      setStatus('Error: ' + _result);
    }

    setResult(_result);
    setProgress(100);
    setStatus('Optimization finished');
    console.log(_result);
    // loadSession();
  }

  async function downloadOptimized() {
    setDownloading(true);
    if (!result) return;

    let fileName = onnxFileName.split('.')[0] + '_optimized.onnx';
    console.log(result.url);
    await downloadFileRequest(result.url!, fileName);
    setDownloading(false);
  }

  return (
    <motion.main
      className={'pageOpt'}
      transition={{
        delay: 0.4,
        duration: 0.2,
        ease: 'easeOut',
        staggerChildren: 0.1
      }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className={'flex w-full gap-[0.5rem]'}>
        <div className={'flex flex-col gap-[0.5rem]'} style={{ width: '100%' }}>
          <form
            className={'optimizeForm area'}
            style={{ width: '100%' }}
            onSubmit={optimize}
          >
            <h2 className={'subtitle'}>
              <Trans>Optimization settings</Trans>
            </h2>
            <div className={'formGroup'}>
              <div className={'element'}>
                <label>
                  <Trans>Input Features:</Trans>
                </label>
                <ul className={'featuresList'}>
                  {features.length == 0 ? (
                    <li>
                      <Trans>
                        Upload a CSV dataset to see available features
                      </Trans>
                    </li>
                  ) : (
                    features.map((feat, i) => (
                      <li key={i}>
                        <input
                          type='checkbox'
                          value={feat}
                          name={'sIFeatures[]'}
                        />
                        <p>{feat}</p>
                      </li>
                    ))
                  )}
                </ul>
              </div>
              <div className={'element'}>
                <label>
                  <Trans>Target Feature:</Trans>
                </label>
                <select className={'targetFeature'} name={'target'}>
                  {features.length == 0 ? (
                    <option disabled>
                      Upload a CSV dataset to see available features
                    </option>
                  ) : (
                    features.map((feat, i) => (
                      <option key={i} value={feat}>
                        {feat}
                      </option>
                    ))
                  )}
                </select>
                <label>
                  <Trans>Encoding type:</Trans>
                </label>
                <select className={'targetFeature'} name={'encoding'}>
                  <option>None</option>
                  <option>label</option>
                  <option>onehot</option>
                </select>
                <label>
                  <Trans>Optimization strategy:</Trans>
                </label>
                <select className={'targetFeature'} name={'strat'}>
                  <option>brute-force</option>
                  <option>neat</option>
                  <option>genetic</option>
                </select>
                <label>
                  <Trans>Maximum Epochs Per Configuration:</Trans>
                </label>
                <input type='number' name={'epochs'} defaultValue='10' />
              </div>
            </div>

            <input
              type='submit'
              id='opt-start-optimization'
              value={t`Start Optimization`}
              disabled={
                !!(
                  progress > -1 &&
                  progress < 100 &&
                  onnxFileName &&
                  csvFileName
                )
              }
            />
            {alert != null ? (
              <div className={'alert'}>
                <h1>{alert}</h1>
              </div>
            ) : (
              ''
            )}
          </form>
          <div
            style={progress != -1 ? { width: '100%' } : { width: 0 }}
            className={`progressZone ${progress != -1 ? 'area' : ''}`}
          >
            <h1>{status}</h1>
            <div className={'barWrapper'}>
              <p>{progress}%</p>
              <div className={'loadingBar'}>
                <div
                  className={`barFiller`}
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className={'dataArea area vertical'}>
            <h1>
              <b>
                <Trans>ONNX file:</Trans>
              </b>{' '}
              {onnxFileName}
            </h1>
            <div className={'flex gap-[1rem]'}>
              <ONNXUploader callBack={populateLists} />
              <button
                className={'deleteButton'}
                onClick={async () => {
                  const curSesId = sessionStorage.getItem('currentSessionId');
                  if (!curSesId) return;
                  await deleteOnnx(parseInt(curSesId));

                  sessionStorage.removeItem('modelResponse');
                  sessionStorage.removeItem('modelData');
                  sessionStorage.removeItem('modelName');

                  await loadSession();
                  populateLists();
                }}
              >
                <Trans>Clear Data</Trans>{' '}
                <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
          <div className={'dataArea area vertical'}>
            <h1>
              <b>
                <Trans>CSV file:</Trans>
              </b>{' '}
              {csvFileName}
            </h1>
            <div className={'flex gap-[1rem]'}>
              <CSVUploader callBack={populateLists} />
              <button
                className={'deleteButton'}
                onClick={async () => {
                  const curSesId = sessionStorage.getItem('currentSessionId');
                  if (!curSesId) return;
                  await deleteCsv(parseInt(curSesId));

                  sessionStorage.removeItem('csvData');

                  await loadSession();
                  populateLists();
                }}
              >
                <Trans>Clear Data</Trans>{' '}
                <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
        </div>
        <div className={'titleArea'}>
          <h1 className={'pageTitle title'}>
            <Trans>Optimization</Trans>
          </h1>
        </div>
      </div>
      {result &&
        JSON.stringify(result).length > 2 &&
        JSON.stringify(result) != 'null' &&
        true &&
        result &&
        result.best_config && (
          <OptimizationResults
            result={result}
            progress={progress}
            downloading={downloading}
            downloadOptimized={downloadOptimized}
          />
        )}
    </motion.main>
  );
}
